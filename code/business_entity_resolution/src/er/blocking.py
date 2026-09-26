"""Candidate generation (blocking).

Every Source-2/3 record ("query") is compared only with Source-1 records of the
same country. Each record yields two bags of tokens:

  name:    n:<word>  s:<skeleton>  ns:<name without spaces>
           nb:<w1>_<w2>  sb:<k1>_<k2>   (consecutive word / skeleton pairs)
  address: a:<word>  b:<w1>_<w2> (consecutive)  c:<w1>_<w3> (skip one word)

Word pairs keep records made only of common words searchable ("gold lakshmi",
"indian agro"); skip pairs survive inserted words ("435 [439] washington").

Three IDF-weighted cosine searches are run - name only, address only, and both
combined - and their top-K lists are merged. A record whose name was replaced
still reaches its entity through the address search, and a record with an empty
or minimal address through the name search. Every merged pair then gets all
three similarity scores and its rank under each.

Tokens that are extremely common among Source-1 records ("st", state codes, big
cities) are left out of the index; they barely separate candidates and would
make the search slow.
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
try:
    from sparse_dot_topn import sp_matmul_topn
except ImportError:
    def sp_matmul_topn(top_a, top_b, top_n, threshold=0.0, sort=True, n_threads=0):
        """Graceful pure-SciPy fallback when sparse_dot_topn is not compiled on local environment."""
        b_csr = top_b.tocsr() if not sparse.isspmatrix_csr(top_b) else top_b
        n_q = top_a.shape[0]
        chunk = 2000
        rows, cols, data = [], [], []
        for start in range(0, n_q, chunk):
            end = min(start + chunk, n_q)
            batch = top_a[start:end].dot(b_csr)
            for i in range(batch.shape[0]):
                r_start, r_end = batch.indptr[i], batch.indptr[i + 1]
                if r_start == r_end:
                    continue
                c_idx = batch.indices[r_start:r_end]
                c_val = batch.data[r_start:r_end]
                if threshold > 0.0:
                    m = c_val >= threshold
                    c_idx, c_val = c_idx[m], c_val[m]
                if len(c_val) == 0:
                    continue
                if len(c_val) > top_n:
                    part = np.argpartition(-c_val, top_n)[:top_n]
                    c_idx, c_val = c_idx[part], c_val[part]
                if sort:
                    ord_ = np.argsort(-c_val)
                    c_idx, c_val = c_idx[ord_], c_val[ord_]
                rows.extend([start + i] * len(c_val))
                cols.extend(c_idx)
                data.extend(c_val)
        return sparse.csr_matrix((data, (rows, cols)), shape=(top_a.shape[0], top_b.shape[1]), dtype=np.float32)

PASSES = ("comb", "name", "addr")


def record_tokens(n_full, n_parts, n_skel, n_nospace, a_full):
    """(name tokens, address tokens) of one normalised record."""
    name = set()
    words = n_full.split()
    skels = n_skel.split()
    for w in words:
        if len(w) >= 2 or w.isdigit():
            name.add("n:" + w)
    for k in skels:
        if len(k) >= 3:
            name.add("s:" + k)
    for w1, w2 in zip(words, words[1:]):
        name.add("nb:" + w1 + "_" + w2)
    for k1, k2 in zip(skels, skels[1:]):
        name.add("sb:" + k1 + "_" + k2)
    if len(n_nospace) >= 5:
        name.add("ns:" + n_nospace)
    for part in n_parts.split("|") if n_parts else ():
        pw = part.split()
        for w in pw:
            if len(w) >= 2 or w.isdigit():
                name.add("n:" + w)
        for w1, w2 in zip(pw, pw[1:]):
            name.add("nb:" + w1 + "_" + w2)
        ns = part.replace(" ", "")
        if len(ns) >= 5:
            name.add("ns:" + ns)
    addr = set()
    aw = a_full.split()
    for w in aw:
        addr.add("a:" + w)
    for w1, w2 in zip(aw, aw[1:]):
        addr.add("b:" + w1 + "_" + w2)
    for w1, w3 in zip(aw, aw[2:]):
        addr.add("c:" + w1 + "_" + w3)
    return name, addr


def _iter_tokens(df):
    cols = ("n_full", "n_parts", "n_skel", "n_nospace", "a_full")
    for row in zip(*(df[c].to_numpy() for c in cols)):
        yield record_tokens(*row)


def _count_df(df):
    """Source-1 document frequency of every name token and address token."""
    dn, da = {}, {}
    for n, a in _iter_tokens(df):
        for t in n:
            dn[t] = dn.get(t, 0) + 1
        for t in a:
            da[t] = da.get(t, 0) + 1
    return dn, da


def _token_ids(df, vn, va):
    """CSR structures (indptr, indices) of name and address tokens found in the vocabularies."""
    n = len(df)
    ipn = np.zeros(n + 1, dtype=np.int64)
    ipa = np.zeros(n + 1, dtype=np.int64)
    fn, fa = [], []
    for i, (nt, at) in enumerate(_iter_tokens(df)):
        fn.extend(vn[t] for t in nt if t in vn)
        fa.extend(va[t] for t in at if t in va)
        ipn[i + 1] = len(fn)
        ipa[i + 1] = len(fa)
    return (ipn, np.asarray(fn, dtype=np.int32)), (ipa, np.asarray(fa, dtype=np.int32))


def _idf_matrix(indptr, indices, idf):
    return sparse.csr_matrix((idf[indices], indices, indptr), shape=(len(indptr) - 1, len(idf)))


def _l2(m):
    norms = np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    return sparse.diags((1.0 / norms).astype(np.float32)).dot(m).tocsr()


class _Index:
    """L2-normalised IDF matrices of one search pass (Source-1 and queries)."""

    def __init__(self, m1, mq):
        self.m1 = _l2(m1)
        self.mq = _l2(mq)

    def topk(self, k, n_threads):
        res = sp_matmul_topn(self.mq, self.m1.T.tocsr(), top_n=k, threshold=0.0, sort=True,
                             n_threads=n_threads).tocsr()
        counts = np.diff(res.indptr)
        q_row = np.repeat(np.arange(self.mq.shape[0], dtype=np.int64), counts)
        return q_row, res.indices.astype(np.int64)

    def pair_scores(self, q_row, s_row, chunk=5_000_000):
        out = np.empty(len(q_row), dtype=np.float32)
        for a in range(0, len(q_row), chunk):
            qi, si = q_row[a:a + chunk], s_row[a:a + chunk]
            out[a:a + chunk] = np.asarray(self.mq[qi].multiply(self.m1[si]).sum(axis=1)).ravel()
        return out


def _ranks(q_row, score):
    """1-based rank of each pair's score within its query (1 = best)."""
    order = np.lexsort((-score, q_row))
    qs = q_row[order]
    starts = np.r_[0, np.flatnonzero(qs[1:] != qs[:-1]) + 1]
    group_start = np.repeat(starts, np.diff(np.r_[starts, len(qs)]))
    rank = np.empty(len(q_row), dtype=np.int16)
    rank[order] = (np.arange(len(qs)) - group_start + 1).astype(np.int16)
    return rank


def candidates_for_country(s1, q, ks, max_df_frac, n_threads):
    """Merged candidates of the three searches for one country.

    Returns a DataFrame with q_row/s1_row (positions within ``q``/``s1``), the
    cosine score and rank of every pass, and ``found_by`` (bit mask of passes).
    """
    max_df = max(50, int(max_df_frac * len(s1)))
    dn, da = _count_df(s1)
    vn = {t: i for i, t in enumerate(t for t, c in dn.items() if c <= max_df)}
    va = {t: i for i, t in enumerate(t for t, c in da.items() if c <= max_df)}
    del dn, da
    (ipn1, ixn1), (ipa1, ixa1) = _token_ids(s1, vn, va)
    (ipnq, ixnq), (ipaq, ixaq) = _token_ids(q, vn, va)
    n_docs = len(s1) + len(q)

    def idf(ix1, ixq, size):
        df_all = np.bincount(ix1, minlength=size) + np.bincount(ixq, minlength=size)
        return (np.log((n_docs + 1) / (df_all + 1)) + 1.0).astype(np.float32)

    idf_n = idf(ixn1, ixnq, len(vn))
    idf_a = idf(ixa1, ixaq, len(va))
    del vn, va
    n1, nq = _idf_matrix(ipn1, ixn1, idf_n), _idf_matrix(ipnq, ixnq, idf_n)
    a1, aq = _idf_matrix(ipa1, ixa1, idf_a), _idf_matrix(ipaq, ixaq, idf_a)
    # name and address tokens have distinct prefixes, so the combined search is
    # simply both matrices side by side
    idx = {
        "comb": _Index(sparse.hstack([n1, a1], format="csr"), sparse.hstack([nq, aq], format="csr")),
        "name": _Index(n1, nq),
        "addr": _Index(a1, aq),
    }
    del n1, nq, a1, aq
    found = []
    for bit, p in enumerate(PASSES):
        if ks[p] > 0:
            qr, sr = idx[p].topk(ks[p], n_threads)
            found.append(pd.DataFrame({"q_row": qr, "s1_row": sr, "bit": np.int8(1 << bit)}))
    pairs = pd.concat(found, ignore_index=True)
    pairs = pairs.groupby(["q_row", "s1_row"], sort=False)["bit"].sum().reset_index()
    pairs.rename(columns={"bit": "found_by"}, inplace=True)
    qr, sr = pairs["q_row"].to_numpy(), pairs["s1_row"].to_numpy()
    for p in PASSES:
        pairs[f"{p}_cos"] = idx[p].pair_scores(qr, sr)
        pairs[f"{p}_rank"] = _ranks(qr, pairs[f"{p}_cos"].to_numpy())
    return pairs


def generate_candidates(s1, q, ks=None, max_df_frac=0.002, n_threads=None, verbose=True):
    """Candidates for all countries.

    ``ks`` maps pass name -> top-K (default comb 10, name 5, addr 5). Returned
    positions refer to the row order of ``s1``/``q``. Result columns: q_pos,
    s1_pos, found_by, <pass>_cos, <pass>_rank; sorted by q_pos then comb rank.
    """
    ks = ks or {"comb": 10, "name": 5, "addr": 5}
    n_threads = n_threads or os.cpu_count()
    s1_country = s1["country"].to_numpy()
    q_country = q["country"].to_numpy()
    parts = []
    for country in sorted(set(q_country)):
        s1_idx = np.flatnonzero(s1_country == country)
        q_idx = np.flatnonzero(q_country == country)
        if len(s1_idx) == 0 or len(q_idx) == 0:
            continue
        t0 = time.time()
        c = candidates_for_country(s1.iloc[s1_idx], q.iloc[q_idx], ks, max_df_frac, n_threads)
        c.insert(0, "q_pos", q_idx[c.pop("q_row").to_numpy()])
        c.insert(1, "s1_pos", s1_idx[c.pop("s1_row").to_numpy()])
        parts.append(c)
        if verbose:
            print(f"  blocking {country}: {len(q_idx)} queries x {len(s1_idx)} S1 -> {len(c)} pairs "
                  f"({time.time() - t0:.0f}s)", flush=True)
    cands = pd.concat(parts, ignore_index=True)
    cands = cands.sort_values(["q_pos", "comb_rank"], kind="stable").reset_index(drop=True)
    # legacy names used by the features
    cands["blk_score"] = cands["comb_cos"]
    cands["blk_rank"] = cands["comb_rank"]
    return cands


def load_prep(prep_dir):
    s1 = pd.read_parquet(os.path.join(prep_dir, "s1.parquet"))
    q = pd.concat([pd.read_parquet(os.path.join(prep_dir, f"s{s}.parquet")) for s in (2, 3)], ignore_index=True)
    return s1, q


def true_pair_mask(cands, s1, q, truth):
    """Boolean array: is each candidate pair a true match?"""
    q_owner = {x: s for s, ids in truth.items() for x in ids}
    owner = pd.Series(q["entity_id"].to_numpy()).map(q_owner).to_numpy()
    return owner[cands["q_pos"].to_numpy()] == s1["entity_id"].to_numpy()[cands["s1_pos"].to_numpy()]


def recall_report(cands, hit, n_true_pairs):
    """Share of true (query, S1) pairs kept by blocking: overall, per pass and by comb rank."""
    lines = [f"recall(all passes): {hit.sum() / n_true_pairs * 100:.2f}%"]
    fb = cands["found_by"].to_numpy()
    for bit, p in enumerate(PASSES):
        lines.append(f"{p}-only-found: {(hit & (fb == (1 << bit))).sum() / n_true_pairs * 100:.2f}%")
    ranks = cands["comb_rank"].to_numpy()
    for k in (1, 3, 10):
        lines.append(f"comb@{k}: {hit[ranks <= k].sum() / n_true_pairs * 100:.2f}%")
    return lines


def main():
    from .io import read_ground_truth
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prep", default="work/dev/prep")
    ap.add_argument("--gt", default="work/dev/train_ground_truth.tsv")
    ap.add_argument("--k-comb", type=int, default=10)
    ap.add_argument("--k-name", type=int, default=5)
    ap.add_argument("--k-addr", type=int, default=5)
    ap.add_argument("--max-df-frac", type=float, default=0.002)
    args = ap.parse_args()
    s1, q = load_prep(args.prep)
    truth = read_ground_truth(args.gt)
    t0 = time.time()
    cands = generate_candidates(s1, q, {"comb": args.k_comb, "name": args.k_name, "addr": args.k_addr},
                                args.max_df_frac)
    print(f"{len(cands)} candidate pairs ({len(cands) / len(q):.1f}/query) in {time.time() - t0:.0f}s")
    hit = true_pair_mask(cands, s1, q, truth)
    print("\n".join(recall_report(cands, hit, sum(len(v) for v in truth.values()))))


if __name__ == "__main__":
    main()
