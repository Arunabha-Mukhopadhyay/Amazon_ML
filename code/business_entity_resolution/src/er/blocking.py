"""Candidate generation (blocking).

Every Source-2/3 record ("query") is compared only with Source-1 records of the
same country. Each record becomes a bag of tokens:

    n:<word>       cleaned name words (and words of alternative name parts)
    s:<skeleton>   phonetic skeleton of each name word (transliteration-proof)
    ns:<nospace>   whole name without spaces (matches website labels/handles)
    a:<word>       cleaned address words
    b:<w1>_<w2>    consecutive address word pairs ("801_broad" is very selective)

Tokens are weighted by IDF (rare = informative) and every query keeps its top-K
Source-1 records by cosine similarity. Tokens that are extremely common among
Source-1 records ("st", state codes, big cities) are left out of the index; they
barely separate candidates and would make the search slow.

Memory: token sets are never kept for all records at once; they are turned
straight into sparse-matrix column indices.
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
from scipy import sparse
from sparse_dot_topn import sp_matmul_topn


def record_tokens(n_full, n_parts, n_skel, n_nospace, a_full):
    """Blocking tokens of one normalised record."""
    toks = set()
    words = n_full.split()
    for part in n_parts.split("|") if n_parts else ():
        words.extend(part.split())
        ns = part.replace(" ", "")
        if len(ns) >= 5:
            toks.add("ns:" + ns)
    for w in words:
        if len(w) >= 2 or w.isdigit():
            toks.add("n:" + w)
    for k in n_skel.split():
        if len(k) >= 3:
            toks.add("s:" + k)
    if len(n_nospace) >= 5:
        toks.add("ns:" + n_nospace)
    addr = a_full.split()
    for w in addr:
        toks.add("a:" + w)
    for w1, w2 in zip(addr, addr[1:]):
        toks.add("b:" + w1 + "_" + w2)
    return toks


def _iter_tokens(df):
    cols = ("n_full", "n_parts", "n_skel", "n_nospace", "a_full")
    for row in zip(*(df[c].to_numpy() for c in cols)):
        yield record_tokens(*row)


def _token_ids(df, vocab):
    """CSR structure (indptr, indices) of the tokens of ``df`` present in ``vocab``."""
    indptr = np.zeros(len(df) + 1, dtype=np.int64)
    chunks = []
    for i, toks in enumerate(_iter_tokens(df)):
        ids = [vocab[t] for t in toks if t in vocab]
        chunks.append(ids)
        indptr[i + 1] = indptr[i] + len(ids)
    indices = np.fromiter((j for ids in chunks for j in ids), dtype=np.int32, count=int(indptr[-1]))
    return indptr, indices


def _weighted(indptr, indices, idf, n_cols):
    """L2-normalised binary-TF x IDF CSR matrix."""
    data = idf[indices].astype(np.float32)
    m = sparse.csr_matrix((data, indices, indptr), shape=(len(indptr) - 1, n_cols))
    norms = np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    return sparse.diags((1.0 / norms).astype(np.float32)).dot(m).tocsr()


def candidates_for_country(s1, q, top_k, max_df_frac, n_threads):
    """Top-K Source-1 candidates for every query of one country.

    Returns arrays (q_row, s1_row, score, rank) with positions into ``s1``/``q``.
    """
    df_s1 = {}
    for toks in _iter_tokens(s1):
        for t in toks:
            df_s1[t] = df_s1.get(t, 0) + 1
    max_df = max(50, int(max_df_frac * len(s1)))
    vocab = {}
    for t, c in df_s1.items():
        if c <= max_df:
            vocab[t] = len(vocab)
    del df_s1
    ip1, ix1 = _token_ids(s1, vocab)
    ipq, ixq = _token_ids(q, vocab)
    n_cols = len(vocab)
    del vocab
    df_all = np.bincount(ix1, minlength=n_cols) + np.bincount(ixq, minlength=n_cols)
    idf = (np.log((len(s1) + len(q) + 1) / (df_all + 1)) + 1.0).astype(np.float32)
    m1 = _weighted(ip1, ix1, idf, n_cols)
    mq = _weighted(ipq, ixq, idf, n_cols)
    res = sp_matmul_topn(mq, m1.T.tocsr(), top_n=top_k, threshold=0.0, sort=True, n_threads=n_threads).tocsr()
    counts = np.diff(res.indptr)
    q_row = np.repeat(np.arange(len(q), dtype=np.int64), counts)
    starts = np.repeat(res.indptr[:-1], counts)
    rank = (np.arange(len(res.indices)) - starts + 1).astype(np.int16)
    return q_row, res.indices.astype(np.int64), res.data.astype(np.float32), rank


def generate_candidates(s1, q, top_k=10, max_df_frac=0.002, n_threads=None, verbose=True):
    """Candidates for all countries.

    ``s1``/``q`` are normalised frames; returned positions refer to their row
    order. Returns a DataFrame (q_pos, s1_pos, blk_score, blk_rank) sorted by
    q_pos then rank.
    """
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
        qr, sr, sc, rk = candidates_for_country(s1.iloc[s1_idx], q.iloc[q_idx], top_k, max_df_frac, n_threads)
        parts.append(pd.DataFrame({"q_pos": q_idx[qr], "s1_pos": s1_idx[sr], "blk_score": sc, "blk_rank": rk}))
        if verbose:
            print(f"  blocking {country}: {len(q_idx)} queries x {len(s1_idx)} S1 -> {len(qr)} pairs "
                  f"({time.time() - t0:.0f}s)", flush=True)
    cands = pd.concat(parts, ignore_index=True)
    return cands.sort_values(["q_pos", "blk_rank"], kind="stable").reset_index(drop=True)


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
    """Share of true (query, S1) pairs that survive blocking, by rank cut-off."""
    ranks = cands["blk_rank"].to_numpy()
    lines = []
    for k in (1, 2, 3, 5, 10, 20, 50):
        if k > ranks.max():
            break
        lines.append(f"recall@{k}: {hit[ranks <= k].sum() / n_true_pairs * 100:.2f}%")
    return lines


def main():
    from .io import read_ground_truth
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prep", default="work/dev/prep")
    ap.add_argument("--gt", default="work/dev/train_ground_truth.tsv")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--max-df-frac", type=float, default=0.002)
    args = ap.parse_args()
    s1, q = load_prep(args.prep)
    truth = read_ground_truth(args.gt)
    t0 = time.time()
    cands = generate_candidates(s1, q, args.k, args.max_df_frac)
    print(f"{len(cands)} candidate pairs for {len(q)} queries in {time.time() - t0:.0f}s")
    hit = true_pair_mask(cands, s1, q, truth)
    print("\n".join(recall_report(cands, hit, sum(len(v) for v in truth.values()))))


if __name__ == "__main__":
    main()
