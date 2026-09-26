"""Second-stage re-scoring of each query's best candidate.

Stage 1 scores every (query, candidate) pair on its own. Stage 2 looks only at
each query's best candidate and adds context the pairwise model cannot see:

* the query's competition: its best vs second-best stage-1 probability;
* the candidate's competition: how many other queries chose the same Source-1
  record, how confident they are, and where this query ranks among them;
* sibling evidence: the other queries that chose the same Source-1 record are
  usually the entity's other Source-2/3 records. A query that is nearly
  identical to a confidently matched sibling is very likely a true match even
  if its own evidence is weak (e.g. an empty address); a query that differs
  from the confident siblings (another descriptor word, another number) is a
  likely decoy.

Stage 2 predicts whether the query's best candidate is its true entity; the
final decision thresholds that probability.
"""

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from .features import FEATURES

TOP_SIBLINGS = 6  # compare each query with the most confident queries of its candidate
MIN_P = 0.05      # queries below this stage-1 probability are not considered as siblings

SIB_FEATURES = ["sib_n", "sib_max_nm", "sib_max_ad", "sib_max_both", "sib_n_close", "top_sib_nm",
                "top_sib_ad", "top_sib_p", "close_sib_p"]
CTX_FEATURES = ["p1", "p1_second", "p1_margin", "s_n_best", "s_sum_p", "s_max_p_other", "s_rank_p",
                "s_n_pairs_hi"]
S2_FEATURES = CTX_FEATURES + SIB_FEATURES + FEATURES


def best_rows(q_pos, p):
    """Row index of every query's highest-probability candidate and its runner-up probability."""
    q_pos = np.asarray(q_pos)
    order = np.lexsort((-np.asarray(p), q_pos))
    qs = q_pos[order]
    first = np.ones(len(qs), dtype=bool)
    first[1:] = qs[1:] != qs[:-1]
    best_idx = order[first]
    second = np.zeros(len(best_idx), dtype=np.float32)
    # the row right after each group's first row is its runner-up (if it belongs to the same query)
    pos_first = np.flatnonzero(first)
    nxt = pos_first + 1
    ok = nxt < len(qs)
    same = np.zeros(len(pos_first), dtype=bool)
    same[ok] = qs[nxt[ok]] == qs[pos_first[ok]]
    second[same] = np.asarray(p)[order[nxt[same]]]
    return best_idx, second


def _group_rank(keys, values):
    """1-based rank of ``values`` (descending) within groups of ``keys``."""
    order = np.lexsort((-values, keys))
    ks = keys[order]
    starts = np.r_[0, np.flatnonzero(ks[1:] != ks[:-1]) + 1]
    gstart = np.repeat(starts, np.diff(np.r_[starts, len(ks)]))
    rank = np.empty(len(keys), dtype=np.int32)
    rank[order] = np.arange(len(ks)) - gstart + 1
    return rank


def stage2_frame(cands, p, best_idx, p_second, best_x, s1, q):
    """Stage-2 feature table, one row per query (its best candidate).

    ``best_idx``/``p_second`` come from ``best_rows``; ``best_x`` holds the
    stage-1 features of those rows in the same order.
    """
    qb = cands["q_pos"].to_numpy()[best_idx]
    sb = cands["s1_pos"].to_numpy()[best_idx]
    pb = np.asarray(p)[best_idx].astype(np.float32)
    n = len(best_idx)
    f = {"p1": pb, "p1_second": p_second, "p1_margin": pb - p_second}

    # candidate-side competition
    s_count = np.bincount(sb, minlength=len(s1))
    s_sum = np.bincount(sb, weights=pb, minlength=len(s1))
    f["s_n_best"] = (s_count[sb] - 1).astype(np.float32)
    f["s_sum_p"] = (s_sum[sb] - pb).astype(np.float32)
    rank = _group_rank(sb, pb)
    f["s_rank_p"] = rank.astype(np.float32)
    # best other probability for the same Source-1 record
    order = np.lexsort((-pb, sb))
    ss, ps = sb[order], pb[order]
    starts = np.r_[0, np.flatnonzero(ss[1:] != ss[:-1]) + 1]
    sizes = np.diff(np.r_[starts, len(ss)])
    gmax = ps[starts]
    gsec = np.where(sizes > 1, ps[np.minimum(starts + 1, len(ps) - 1)], 0.0)
    grp = np.empty(n, dtype=np.int64)
    grp[order] = np.repeat(np.arange(len(starts)), sizes)
    f["s_max_p_other"] = np.where(rank == 1, gsec[grp], gmax[grp]).astype(np.float32)
    hi = np.asarray(p) > 0.5
    s_hi = np.bincount(cands["s1_pos"].to_numpy()[hi], minlength=len(s1))
    f["s_n_pairs_hi"] = (s_hi[sb] - (pb > 0.5)).astype(np.float32)

    f.update(sibling_features(qb, sb, pb, q))
    frame = pd.DataFrame(f)
    x = pd.DataFrame(np.asarray(best_x, dtype=np.float32), columns=FEATURES)
    return pd.concat([frame, x], axis=1)[S2_FEATURES].astype(np.float32)


def sibling_features(qb, sb, pb, q):
    """Similarity of each query to the most confident other queries of its candidate."""
    n = len(qb)
    rows = np.arange(n)
    eligible = pb >= MIN_P
    # top siblings per Source-1 record: rank within group by probability
    rank = _group_rank(sb, pb)
    top = eligible & (rank <= TOP_SIBLINGS)
    top_tab = pd.DataFrame({"s": sb[top], "sib": rows[top], "sib_rank": rank[top], "sib_p": pb[top]})
    me = pd.DataFrame({"s": sb, "me": rows})
    pairs = me.merge(top_tab, on="s", how="inner")
    pairs = pairs[pairs["me"] != pairs["sib"]]
    out = {k: np.full(n, np.nan, dtype=np.float32) for k in SIB_FEATURES}
    out["sib_n"] = np.zeros(n, dtype=np.float32)
    if len(pairs) == 0:
        return out
    me_i = pairs["me"].to_numpy()
    sib_i = pairs["sib"].to_numpy()
    q_name = q["n_full"].to_numpy()
    q_addr = q["a_full"].to_numpy()
    nm = process.cpdist(q_name[qb[me_i]], q_name[qb[sib_i]], scorer=fuzz.token_set_ratio, workers=-1,
                        dtype=np.float32)
    ad = process.cpdist(q_addr[qb[me_i]], q_addr[qb[sib_i]], scorer=fuzz.token_set_ratio, workers=-1,
                        dtype=np.float32)
    both = np.minimum(nm, ad)
    t = pd.DataFrame({"me": me_i, "nm": nm, "ad": ad, "both": both, "close": (both >= 90).astype(np.float32),
                      "sib_rank": pairs["sib_rank"].to_numpy(), "sib_p": pairs["sib_p"].to_numpy()})
    g = t.groupby("me")
    idx = g.size().index.to_numpy()
    out["sib_n"][idx] = g.size().to_numpy()
    out["sib_max_nm"][idx] = g["nm"].max().to_numpy()
    out["sib_max_ad"][idx] = g["ad"].max().to_numpy()
    out["sib_max_both"][idx] = g["both"].max().to_numpy()
    out["sib_n_close"][idx] = g["close"].sum().to_numpy()
    # the most confident sibling (lowest sib_rank)
    tops = t.sort_values(["me", "sib_rank"]).drop_duplicates("me")
    ti = tops["me"].to_numpy()
    out["top_sib_nm"][ti] = tops["nm"].to_numpy()
    out["top_sib_ad"][ti] = tops["ad"].to_numpy()
    out["top_sib_p"][ti] = tops["sib_p"].to_numpy()
    # the most similar sibling's confidence
    closest = t.sort_values(["me", "both"], ascending=[True, False]).drop_duplicates("me")
    out["close_sib_p"][closest["me"].to_numpy()] = closest["sib_p"].to_numpy()
    return out
