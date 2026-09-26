"""Error analysis of a training run's evaluation fold (run from the repo root on the VM: scripts/py tools/analyze_errors.py)."""
import argparse
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--work", default="work/full", help="work folder of the run")
ap.add_argument("--data-dir", default="~/data/dataset", help="folder with train/ and test/")
ap.add_argument("--train-pct", type=int, default=25, help="--train-pct used by that run")
ARGS = ap.parse_args()
import os
DATA = os.path.expanduser(ARGS.data_dir)
import json, random
import numpy as np, pandas as pd
from er.io import read_ground_truth
from er.run import s1_folds, VAL
from er.evaluate import f05_per_entity

random.seed(0)
W = ARGS.work
meta = json.load(open(f"{W}/meta.json")); t = meta["threshold"]
vp = pd.read_parquet(f"{W}/val_pairs.parquet")
truth = read_ground_truth(os.path.join(DATA, "train", "train_ground_truth.tsv"))
s1 = pd.read_parquet(f"{W}/prep_train/s1.parquet")
q = pd.concat([pd.read_parquet(f"{W}/prep_train/s{s}.parquet") for s in (2, 3)], ignore_index=True)
raw = {}
for df in (s1, q):
    raw.update(zip(df.entity_id, (df.business_name + " | " + df.business_address).str.slice(0, 95)))
fold = s1_folds(s1.entity_id.to_numpy(), ARGS.train_pct)
val_ids = list(s1.entity_id[fold == VAL])
country = dict(zip(s1.entity_id, s1.country))
q_owner = {x: s for s, ids in truth.items() for x in ids}

true_pairs = {(x, s) for s in val_ids for x in truth.get(s, ())}
in_c = set(zip(vp.q_id, vp.s1_id))
missed = [p for p in true_pairs if p not in in_c]
print(f"val S1 {len(val_ids)}, true pairs {len(true_pairs)}, blocking-missed {len(missed)} ({len(missed)/len(true_pairs)*100:.2f}%)")

vp = vp.sort_values(["q_id", "p"], ascending=[True, False])
best = vp.drop_duplicates("q_id")
sel = best[best.p >= t].copy()
sel["owner"] = sel.q_id.map(q_owner)
sel["tp"] = sel.owner == sel.s1_id
print(f"predicted {len(sel)}: TP {sel.tp.sum()}, FP-decoy {(sel.owner.isna()).sum()}, FP-wrong-entity {((~sel.tp) & sel.owner.notna()).sum()}")

def score(pred_pairs):
    pred = {}
    for x, s in pred_pairs:
        pred.setdefault(s, set()).add(x)
    n_p, n_t, n_tp = [], [], []
    for s in val_ids:
        P = pred.get(s, set()); T = set(truth.get(s, ()))
        n_p.append(len(P)); n_t.append(len(T)); n_tp.append(len(P & T))
    f = f05_per_entity(n_p, n_t, n_tp)
    return f.mean(), f

base, fb = score(zip(sel.q_id, sel.s1_id))
no_fp, _ = score(zip(sel.q_id[sel.tp], sel.s1_id[sel.tp]))
all_found = set(zip(sel.q_id[sel.tp], sel.s1_id[sel.tp])) | {p for p in true_pairs if p in in_c}
with_fn, _ = score(all_found)
print(f"score {base:.5f} | if no false positives {no_fp:.5f} (+{no_fp-base:.4f}) | + all in-shortlist misses found {with_fn:.5f} | ceiling {meta['val_ceiling_after_blocking']:.5f}")
sing = np.array([len(truth.get(s, ())) == 0 for s in val_ids])
print(f"singletons {sing.sum()} ({sing.mean()*100:.1f}%): mean F {fb[sing].mean():.4f} | non-singletons mean F {fb[~sing].mean():.4f}")
for c in ("US", "India"):
    m = np.array([country[s] == c for s in val_ids])
    print(f"  {c}: F {fb[m].mean():.4f}  singletons F {fb[m & sing].mean():.4f}  non-singletons F {fb[m & ~sing].mean():.4f}")

tpairs = vp[vp.y == 1].copy()
tpairs["best_s1"] = tpairs.q_id.map(best.set_index("q_id").s1_id)
tpairs["best_p"] = tpairs.q_id.map(best.set_index("q_id").p)
fn_low = tpairs[(tpairs.p < t) & (tpairs.best_s1 == tpairs.s1_id)]
fn_steal = tpairs[tpairs.best_s1 != tpairs.s1_id]
print(f"in-shortlist misses: low confidence {len(fn_low)}, lost to another candidate {len(fn_steal)}")
print("blk_rank of true pairs:", tpairs.blk_rank.value_counts().sort_index().to_dict())

def show(title, rows, n=12):
    print(f"\n=== {title} ===")
    for r in rows[:n]:
        print("  " + "\n  ".join(r))
        print("  --")

random.shuffle(missed)
show("BLOCKING MISSES (query  vs  true S1)", [[f"Q  {raw[x]}", f"S1 {raw[s]}"] for x, s in missed])
fpd = sel[sel.owner.isna()].sample(12, random_state=1)
show("FP DECOYS (decoy query  vs  S1 it was given to)", [[f"Q  {raw[r.q_id]}  p={r.p:.2f}", f"S1 {raw[r.s1_id]}"] for r in fpd.itertuples()])
fpw = sel[(~sel.tp) & sel.owner.notna()].sample(12, random_state=1)
show("FP WRONG ENTITY (query / given S1 / true S1)", [[f"Q  {raw[r.q_id]}  p={r.p:.2f}", f"S1 {raw[r.s1_id]}", f"OK {raw[r.owner]}"] for r in fpw.itertuples()])
fnl = fn_low.sample(12, random_state=1)
show("FN LOW CONFIDENCE (query / true S1)", [[f"Q  {raw[r.q_id]}  p={r.p:.2f} rank={r.blk_rank}", f"S1 {raw[r.s1_id]}"] for r in fnl.itertuples()])
