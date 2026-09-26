"""End-to-end pipeline: learn on the training data, then predict the test set.

Commands
  train    normalise train data -> blocking -> features -> LightGBM -> tune threshold
           on held-out Source-1 entities (reports the challenge metric)
  predict  normalise test data -> blocking -> features -> predict -> decide ->
           write output/matching_results.tsv and output/candidate_pairs.tsv ->
           run the official validator

Examples (from the repository root):
  scripts/py -m er.run train   --train-dir ../../dataset/train --work work/full
  scripts/py -m er.run predict --test-dir  ../../dataset/test  --work work/full --out-dir output

Source-1 entities of the training data are split by a hash of their id:
  5% early stopping, 15% evaluation, ``--train-pct`` % training, rest unused.
Every Source-2/3 record always stays in the candidate pool, so evaluated
entities face exactly the competition they would face in the test set.
"""

import argparse
import json
import os
import subprocess
import sys
import time
import zlib

import lightgbm as lgb
import numpy as np
import pandas as pd

from .blocking import generate_candidates, load_prep, recall_report, true_pair_mask
from .evaluate import score_positions, sweep_thresholds
from .features import FEATURES, FeatureBuilder, iter_query_chunks
from .io import check_clean_ids, read_ground_truth, read_source
from .model import PARAMS_STAGE2, train_model
from .stage2 import S2_FEATURES, best_rows, stage2_frame
from .prepare import normalize_frame

UNUSED, TRAIN, ES, VAL = 0, 1, 2, 3
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def ensure_prep(src_dir, prefix, prep_dir, workers=None):
    """Normalise the three source files once; later runs reuse the parquet cache."""
    os.makedirs(prep_dir, exist_ok=True)
    for s in (1, 2, 3):
        out = os.path.join(prep_dir, f"s{s}.parquet")
        if os.path.exists(out):
            continue
        df = normalize_frame(read_source(os.path.join(src_dir, f"{prefix}_source{s}.tsv")), workers)
        df["src"] = s
        df.to_parquet(out, index=False)
        log(f"normalised {prefix} source {s}: {len(df)} rows")
    return prep_dir


def s1_folds(s1_ids, train_pct):
    """Fold of every Source-1 entity from a (salted) hash of its id.

    The salt keeps folds independent of the unsalted hash used by er.sample.
    """
    h = np.fromiter((zlib.crc32(b"fold:" + x.encode()) % 100 for x in s1_ids), dtype=np.int16, count=len(s1_ids))
    fold = np.full(len(h), UNUSED, dtype=np.int8)
    fold[h < 5] = ES
    fold[(h >= 5) & (h < 20)] = VAL
    fold[(h >= 20) & (h < 20 + train_pct)] = TRAIN
    return fold


def compute_features(cands, fb, path, chunk_pairs):
    """Features for all candidate pairs, written to a float32 .npy memmap."""
    x = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(len(cands), len(FEATURES)))
    done = 0
    for sl in iter_query_chunks(cands, chunk_pairs):
        x[sl] = fb.transform(cands.iloc[sl]).to_numpy(dtype=np.float32)
        done = sl.stop
        log(f"  features {done}/{len(cands)}")
    x.flush()
    return x


def predict_chunks(model, x, chunk_pairs):
    p = np.empty(x.shape[0], dtype=np.float32)
    for start in range(0, x.shape[0], chunk_pairs):
        p[start:start + chunk_pairs] = model.predict(x[start:start + chunk_pairs], num_threads=0)
    return p


# --------------------------------------------------------------------------
# train
# --------------------------------------------------------------------------

def cmd_train(args):
    os.makedirs(args.work, exist_ok=True)
    prep = ensure_prep(args.train_dir, "train", os.path.join(args.work, "prep_train"), args.workers)
    s1, q = load_prep(prep)
    truth = read_ground_truth(os.path.join(args.train_dir, "train_ground_truth.tsv"))
    log(f"loaded train: {len(s1)} S1, {len(q)} S2/S3, {sum(len(v) for v in truth.values())} true pairs")

    ks = {"comb": args.k_comb, "name": args.k_name, "addr": args.k_addr}
    cands = generate_candidates(s1, q, ks, args.max_df_frac)
    cands.to_parquet(os.path.join(args.work, "train_cands.parquet"), index=False)
    hit = true_pair_mask(cands, s1, q, truth)
    recall = recall_report(cands, hit, sum(len(v) for v in truth.values()))
    log(f"blocking: {len(cands)} pairs; " + ", ".join(recall))

    fold = s1_folds(s1["entity_id"].to_numpy(), args.train_pct)
    pair_fold = fold[cands["s1_pos"].to_numpy()]
    fb = FeatureBuilder(s1, q)
    x = compute_features(cands, fb, os.path.join(args.work, "train_features.npy"), args.chunk_pairs)
    del fb

    tr, es = pair_fold == TRAIN, pair_fold == ES
    log(f"training on {tr.sum()} pairs ({hit[tr].sum()} positive), early-stopping on {es.sum()}")
    model = train_model(x[tr], hit[tr].astype(np.int8), x[es], hit[es].astype(np.int8))
    log(f"model: {model.best_iteration} rounds")
    p = predict_chunks(model, x, args.chunk_pairs)

    # ---- stage-1 decision (reference) ----
    s1_pos_of = pd.Series(np.arange(len(s1)), index=s1["entity_id"].to_numpy())
    q_owner = {x_: s for s, ids in truth.items() for x_ in ids}
    owner_pos = pd.Series(q["entity_id"].to_numpy()).map(q_owner).map(s1_pos_of).fillna(-1).to_numpy(np.int64)
    n_true = np.array([len(truth.get(e, ())) for e in s1["entity_id"]], dtype=np.int64)
    val_mask = fold == VAL
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    best_idx, p_second = best_rows(cands["q_pos"].to_numpy(), p)
    qb, sb = cands["q_pos"].to_numpy()[best_idx], cands["s1_pos"].to_numpy()[best_idx]
    correct = sb == owner_pos[qb]
    best1 = pd.DataFrame({"q_pos": qb, "s1_pos": sb, "p": p[best_idx]})
    scores1 = sweep_thresholds(best1, correct, n_true, val_mask, grid)
    log(f"stage 1: val F0.5 {max(scores1):.5f} at threshold {grid[int(np.argmax(scores1))]}")

    # ---- stage 2: re-score each query's best candidate with context ----
    x2 = stage2_frame(cands, p, best_idx, p_second, x[best_idx], s1, q)
    bfold = fold[sb]
    tr2, es2 = bfold == UNUSED, bfold == ES
    use2 = False
    if tr2.sum() > 0:
        log(f"stage 2: training on {tr2.sum()} queries ({correct[tr2].sum()} correct best candidates)")
        model2 = train_model(x2[tr2], correct[tr2].astype(np.int8), x2[es2], correct[es2].astype(np.int8),
                             params=PARAMS_STAGE2)
        p2 = model2.predict(x2, num_threads=0).astype(np.float32)
        scores2 = sweep_thresholds(best1.assign(p=p2), correct, n_true, val_mask, grid)
        log(f"stage 2: val F0.5 {max(scores2):.5f} at threshold {grid[int(np.argmax(scores2))]} "
            f"({model2.best_iteration} rounds)")
        use2 = max(scores2) > max(scores1)
        model2.save_model(os.path.join(args.work, "model2.txt"))
    scores = scores2 if use2 else scores1
    p_final = p2 if use2 else best1["p"].to_numpy()
    t_best = float(grid[int(np.argmax(scores))])
    keep = p_final >= t_best
    chosen_s1, chosen_ok = sb[keep], correct[keep]

    meta = {
        "threshold": t_best, "stage2": use2, "ks": ks, "max_df_frac": args.max_df_frac,
        "train_pct": args.train_pct, "rounds": model.best_iteration, "features": FEATURES,
        "blocking_recall": recall, "val_f05": max(scores), "val_f05_stage1": max(scores1),
        "val_f05_by_threshold": {f"{t:.2f}": round(s, 5) for t, s in zip(grid, scores) if round(t * 100) % 5 == 0},
        "val_ceiling_after_blocking": score_positions(cands["s1_pos"].to_numpy()[hit], np.ones(hit.sum(), bool),
                                                      n_true, val_mask),
        "val_predict_nothing": score_positions(np.array([], dtype=np.int64), np.array([], dtype=bool),
                                               n_true, val_mask),
    }
    country = s1["country"].to_numpy()
    for c in sorted(set(country)):
        meta[f"val_f05_{c}"] = score_positions(chosen_s1, chosen_ok, n_true, val_mask & (country == c))
    imp = pd.Series(model.feature_importance("gain"), index=FEATURES)
    meta["feature_gain_pct"] = {k: round(v, 2) for k, v in (imp / imp.sum() * 100).sort_values(ascending=False).items()}
    if use2:
        imp2 = pd.Series(model2.feature_importance("gain"), index=S2_FEATURES)
        meta["stage2_gain_pct"] = {k: round(v, 2) for k, v in
                                   (imp2 / imp2.sum() * 100).sort_values(ascending=False).head(25).items()}
    model.save_model(os.path.join(args.work, "model.txt"))
    with open(os.path.join(args.work, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)

    # evaluation-fold decisions for error analysis
    vb = val_mask[sb]
    pd.DataFrame({
        "q_id": q["entity_id"].to_numpy()[qb[vb]], "s1_id": s1["entity_id"].to_numpy()[sb[vb]],
        "p1": best1["p"].to_numpy()[vb], "p_final": p_final[vb], "correct": correct[vb],
    }).to_parquet(os.path.join(args.work, "val_best.parquet"), index=False)
    vm = pair_fold == VAL
    pd.DataFrame({
        "q_id": q["entity_id"].to_numpy()[cands["q_pos"].to_numpy()[vm]],
        "s1_id": s1["entity_id"].to_numpy()[cands["s1_pos"].to_numpy()[vm]],
        "blk_rank": cands["blk_rank"].to_numpy()[vm], "p": p[vm], "y": hit[vm],
    }).to_parquet(os.path.join(args.work, "val_pairs.parquet"), index=False)

    summary = {k: meta[k] for k in meta if k.startswith("val_f05") and k != "val_f05_by_threshold"}
    summary.update(threshold=t_best, stage2=use2, rounds=meta["rounds"],
                   ceiling=round(meta["val_ceiling_after_blocking"], 5))
    log("TRAIN SUMMARY " + json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in summary.items()}))
    log("top features: " + ", ".join(f"{k}={v}%" for k, v in list(meta["feature_gain_pct"].items())[:8]))


# --------------------------------------------------------------------------
# predict
# --------------------------------------------------------------------------

def write_grouped(path, header, s1_ids, s1_pos, q_ids, sort_key=None):
    """One row per Source-1 id listing the ids of its pairs (all S1 rows written)."""
    s1_pos = np.asarray(s1_pos)
    order = np.lexsort((sort_key, s1_pos)) if sort_key is not None else np.argsort(s1_pos, kind="stable")
    sp, qs = s1_pos[order], np.asarray(q_ids)[order]
    bounds = np.searchsorted(sp, np.arange(len(s1_ids) + 1))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(header + "\n")
        for i, sid in enumerate(s1_ids):
            f.write(sid + "\t" + ",".join(qs[bounds[i]:bounds[i + 1]]) + "\n")
    check_clean_ids(path)


def cmd_predict(args):
    with open(os.path.join(args.work, "meta.json")) as f:
        meta = json.load(f)
    model = lgb.Booster(model_file=os.path.join(args.work, "model.txt"))
    prep = ensure_prep(args.test_dir, "test", os.path.join(args.work, "prep_test"), args.workers)
    s1, q = load_prep(prep)
    log(f"loaded test: {len(s1)} S1, {len(q)} S2/S3; threshold {meta['threshold']}")

    cands = generate_candidates(s1, q, meta["ks"], meta["max_df_frac"])
    log(f"blocking: {len(cands)} pairs")
    fb = FeatureBuilder(s1, q)
    p = np.empty(len(cands), dtype=np.float32)
    use2 = meta.get("stage2", False)
    best_x_parts = []
    qpos_all = cands["q_pos"].to_numpy()
    for sl in iter_query_chunks(cands, args.chunk_pairs):
        xc = fb.transform(cands.iloc[sl]).to_numpy(dtype=np.float32)
        p[sl] = model.predict(xc, num_threads=0)
        if use2:
            local_best, _ = best_rows(qpos_all[sl], p[sl])
            best_x_parts.append(xc[local_best])
        log(f"  scored {sl.stop}/{len(cands)}")
    del fb

    best_idx, p_second = best_rows(qpos_all, p)
    qb, sb = qpos_all[best_idx], cands["s1_pos"].to_numpy()[best_idx]
    if use2:
        model2 = lgb.Booster(model_file=os.path.join(args.work, "model2.txt"))
        x2 = stage2_frame(cands, p, best_idx, p_second, np.concatenate(best_x_parts), s1, q)
        p_final = model2.predict(x2, num_threads=0).astype(np.float32)
    else:
        p_final = p[best_idx]
    chosen = pd.DataFrame({"q_pos": qb, "s1_pos": sb, "p": p_final})
    # keep every query's final probability so other thresholds can be tried without re-running
    chosen.astype({"q_pos": np.int32, "s1_pos": np.int32}).to_parquet(
        os.path.join(args.work, "test_best.parquet"), index=False)
    threshold = args.threshold if args.threshold is not None else meta["threshold"]
    log(f"decision threshold {threshold}")
    chosen = chosen[chosen["p"] >= threshold]
    s1_ids = s1["entity_id"].to_numpy()
    q_ids = q["entity_id"].to_numpy()
    m_path = os.path.join(args.out_dir, "matching_results.tsv")
    c_path = os.path.join(args.out_dir, "candidate_pairs.tsv")
    write_grouped(m_path, "source1_entity_id\tmatched_entity_ids", s1_ids,
                  chosen["s1_pos"].to_numpy(), q_ids[chosen["q_pos"].to_numpy()], -chosen["p"].to_numpy())
    write_grouped(c_path, "source1_entity_id\tcandidate_entity_ids", s1_ids,
                  cands["s1_pos"].to_numpy(), q_ids[cands["q_pos"].to_numpy()])
    log(f"wrote {m_path} and {c_path}")

    # sanity summary: compare with the training distribution (5.6% singletons, ~3.5 matches/entity)
    n_per_s1 = np.bincount(chosen["s1_pos"].to_numpy(), minlength=len(s1))
    country = s1["country"].to_numpy()
    stats = {"pairs_matched": int(len(chosen)), "queries": int(len(q)),
             "s1_empty_pct": round(float((n_per_s1 == 0).mean() * 100), 2),
             "avg_matches_per_s1": round(float(n_per_s1.mean()), 3)}
    for c in sorted(set(country)):
        m = country == c
        stats[f"{c}_empty_pct"] = round(float((n_per_s1[m] == 0).mean() * 100), 2)
        stats[f"{c}_avg_matches"] = round(float(n_per_s1[m].mean()), 3)
    with open(os.path.join(args.work, "predict_stats.json"), "w") as f:
        json.dump(stats, f, indent=1)
    log("PREDICT SUMMARY " + json.dumps(stats))

    if args.validator and os.path.exists(args.validator):
        cmd = [sys.executable, args.validator, "--matching", m_path, "--candidate", c_path,
               "--test-dir", args.test_dir, "--check-ids"]
        log("running validator: " + " ".join(cmd))
        subprocess.run(cmd, check=False)
    else:
        log(f"validator not found at {args.validator}; skipped")


def cmd_threshold(args):
    """Rewrite matching_results.tsv from saved test probabilities with another threshold."""
    s1, q = load_prep(os.path.join(args.work, "prep_test"))
    best = pd.read_parquet(os.path.join(args.work, "test_best.parquet"))
    chosen = best[best["p"] >= args.threshold]
    m_path = os.path.join(args.out_dir, "matching_results.tsv")
    write_grouped(m_path, "source1_entity_id\tmatched_entity_ids", s1["entity_id"].to_numpy(),
                  chosen["s1_pos"].to_numpy(), q["entity_id"].to_numpy()[chosen["q_pos"].to_numpy()],
                  -chosen["p"].to_numpy())
    n_per_s1 = np.bincount(chosen["s1_pos"].to_numpy(), minlength=len(s1))
    log(f"threshold {args.threshold}: {len(chosen)} pairs, {np.mean(n_per_s1 == 0) * 100:.2f}% empty -> {m_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("train", "predict", "threshold"):
        p = sub.add_parser(name)
        p.add_argument("--work", default="work/full", help="cache/model folder")
        p.add_argument("--chunk-pairs", type=int, default=1_000_000)
        p.add_argument("--workers", type=int, default=None, help="processes for normalisation")
    tp = sub.choices["train"]
    tp.add_argument("--train-dir", default="../../dataset/train")
    tp.add_argument("--k-comb", type=int, default=10, help="candidates per query from the combined search")
    tp.add_argument("--k-name", type=int, default=5, help="... from the name-only search")
    tp.add_argument("--k-addr", type=int, default=5, help="... from the address-only search")
    tp.add_argument("--max-df-frac", type=float, default=0.002)
    tp.add_argument("--train-pct", type=int, default=25, help="%% of Source-1 entities used for training")
    pp = sub.choices["predict"]
    pp.add_argument("--test-dir", default="../../dataset/test")
    pp.add_argument("--out-dir", default="output")
    pp.add_argument("--validator", default="../../utils/validate_submission.py")
    pp.add_argument("--threshold", type=float, default=None, help="override the tuned threshold")
    thp = sub.choices["threshold"]
    thp.add_argument("--threshold", type=float, required=True)
    thp.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    {"train": cmd_train, "predict": cmd_predict, "threshold": cmd_threshold}[args.cmd](args)


if __name__ == "__main__":
    main()
