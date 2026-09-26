"""Pair features ("clues") for (query, Source-1 candidate) pairs.

All string similarities are computed with RapidFuzz's vectorised ``cpdist``
(element-wise, multi-threaded). IDF-weighted overlaps are computed with sparse
matrices. No feature uses the country label, so the model cannot learn
country-specific shortcuts and treats unseen countries (France) the same way.

``FeatureBuilder`` precomputes everything that depends on a whole record
collection once (IDF matrices, name/address frequencies), then turns candidate
pairs into features chunk by chunk, so memory stays bounded at full scale.
"""

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler
from scipy import sparse

FEATURES = [
    "nm_ratio", "nm_tset", "nm_tsort", "nm_partial", "nm_jw", "nm_skel_tset", "nm_skel_ratio",
    "nm_nospace_ratio", "nm_nospace_partial", "nm_parts_best", "nm_wjacc", "nm_wcos",
    "nm_first_eq", "legal_eq", "q_web", "q_nonlatin", "q_nm_ntok", "s_nm_ntok",
    "ad_ratio", "ad_tset", "ad_tsort", "ad_partial", "ad_jw", "ad_wjacc", "ad_wcos",
    "house", "num_jacc", "num_q_in_s", "num_conflict", "q_a_empty", "s_a_empty",
    "s_name_freq", "s_addr_freq", "q_name_freq",
    "blk_score", "blk_rank", "blk_gap_best", "blk_gap_next", "n_close", "is_s3",
]


def _cp(a, b, scorer):
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)


class WeightedOverlap:
    """IDF-weighted Jaccard and cosine between two collections of token strings."""

    def __init__(self, q_docs, s_docs):
        vocab = {}
        for docs in (s_docs, q_docs):
            for d in docs:
                for t in d.split():
                    if t not in vocab:
                        vocab[t] = len(vocab)
        mq = self._binary(q_docs, vocab)
        ms = self._binary(s_docs, vocab)
        df = np.asarray(mq.sum(axis=0)).ravel() + np.asarray(ms.sum(axis=0)).ravel()
        n = mq.shape[0] + ms.shape[0]
        idf = (np.log((n + 1) / (df + 1)) + 1.0).astype(np.float32)
        self.mq = (mq @ sparse.diags(idf)).tocsr()       # query tokens weighted by idf
        self.ms = ms                                       # candidate tokens, binary
        self.wq = np.asarray(self.mq.sum(axis=1)).ravel()
        self.ws = np.asarray((ms @ sparse.diags(idf)).sum(axis=1)).ravel()
        self.nq = np.sqrt(np.asarray(self.mq.multiply(self.mq).sum(axis=1)).ravel())
        self.ns = np.sqrt(np.asarray((ms @ sparse.diags(idf * idf)).sum(axis=1)).ravel())

    @staticmethod
    def _binary(docs, vocab):
        indptr = [0]
        indices = []
        for d in docs:
            indices.extend({vocab[t] for t in d.split()})
            indptr.append(len(indices))
        data = np.ones(len(indices), dtype=np.float32)
        return sparse.csr_matrix((data, np.asarray(indices, dtype=np.int32), np.asarray(indptr, dtype=np.int64)),
                                 shape=(len(docs), len(vocab)))

    def score(self, qi, si):
        """(jaccard, cosine) for pairs (q_docs[qi], s_docs[si]); NaN if a side is empty."""
        a = self.mq[qi]
        b = self.ms[si]
        inter = np.asarray(a.multiply(b).sum(axis=1)).ravel()
        inter2 = np.asarray(a.multiply(a).multiply(b).sum(axis=1)).ravel()
        union = self.wq[qi] + self.ws[si] - inter
        denom = self.nq[qi] * self.ns[si]
        with np.errstate(divide="ignore", invalid="ignore"):
            jac = np.where(union > 0, inter / union, np.nan).astype(np.float32)
            cos = np.where(denom > 0, inter2 / denom, np.nan).astype(np.float32)
        return jac, cos


def number_features(q_nums, s_nums, q_house, s_house):
    """House-number state and number-set overlaps.

    house: 1 = same first number, 0 = different, NaN = one side has none.
    num_jacc: Jaccard of the number sets. q_in_s: share of the query's numbers
    found in the candidate's numbers. conflict: numbers present on only one
    side, counted on the side with fewer such numbers (capped at 5).
    """
    n = len(q_nums)
    house = np.full(n, np.nan, dtype=np.float32)
    jacc = np.full(n, np.nan, dtype=np.float32)
    q_in_s = np.full(n, np.nan, dtype=np.float32)
    conflict = np.zeros(n, dtype=np.float32)
    for i in range(n):
        qh, sh = q_house[i], s_house[i]
        if qh and sh:
            house[i] = 1.0 if qh == sh else 0.0
        qn, sn = q_nums[i], s_nums[i]
        if qn or sn:
            a, b = set(qn.split()), set(sn.split())
            inter = len(a & b)
            jacc[i] = inter / len(a | b)
            if a:
                q_in_s[i] = inter / len(a)
            conflict[i] = min(5, len(a - b), len(b - a))
    return house, jacc, q_in_s, conflict


def best_part_similarity(q_parts, s_full, s_nospace, base):
    """Max similarity between any alternative name part of the query and the
    candidate's name (token-set on words, partial ratio on no-space forms)."""
    out = base.copy()
    for i in np.flatnonzero(np.fromiter((bool(p) for p in q_parts), dtype=bool, count=len(q_parts))):
        best = out[i]
        for part in q_parts[i].split("|"):
            best = max(best, fuzz.token_set_ratio(part, s_full[i]))
            ns = part.replace(" ", "")
            if len(ns) >= 4:
                best = max(best, fuzz.partial_ratio(ns, s_nospace[i]))
        out[i] = best
    return out


class FeatureBuilder:
    """Precomputes collection-level statistics for one (Source-1, queries) pair of frames."""

    def __init__(self, s1, q):
        self.s1 = {c: s1[c].to_numpy() for c in
                   ("n_full", "n_skel", "n_nospace", "n_legal", "a_full", "a_nums", "a_house", "a_empty")}
        self.q = {c: q[c].to_numpy() for c in
                  ("entity_id", "n_full", "n_skel", "n_nospace", "n_legal", "n_parts", "n_web", "n_nonlatin",
                   "a_full", "a_nums", "a_house", "a_empty")}
        self.name_ov = WeightedOverlap(q["n_full"].tolist(), s1["n_full"].tolist())
        self.addr_ov = WeightedOverlap(q["a_full"].tolist(), s1["a_full"].tolist())
        self.q_first = np.array([s.split(" ", 1)[0] for s in q["n_full"]], dtype=object)
        self.s_first = np.array([s.split(" ", 1)[0] for s in s1["n_full"]], dtype=object)
        self.q_ntok = (q["n_full"].str.count(" ").to_numpy() + 1).astype(np.float32)
        self.s_ntok = (s1["n_full"].str.count(" ").to_numpy() + 1).astype(np.float32)
        key_n = s1["country"] + "|" + s1["n_full"]
        key_a = s1["country"] + "|" + s1["a_full"]
        name_counts = key_n.value_counts()
        self.s_name_freq = np.log1p(key_n.map(name_counts).to_numpy()).astype(np.float32)
        self.s_addr_freq = np.log1p(key_a.map(key_a.value_counts()).to_numpy()).astype(np.float32)
        q_key = (q["country"] + "|" + q["n_full"]).map(name_counts).fillna(0).to_numpy()
        self.q_name_freq = np.log1p(q_key).astype(np.float32)
        self.is_s3 = q["entity_id"].str.startswith("S3-").to_numpy().astype(np.float32)

    def transform(self, cands):
        """Features for a block of candidate pairs.

        ``cands`` has q_pos, s1_pos, blk_score, blk_rank and must contain whole
        query groups (all candidates of every query it mentions).
        """
        qi = cands["q_pos"].to_numpy()
        si = cands["s1_pos"].to_numpy()
        Q, S = self.q, self.s1
        qn, sn = Q["n_full"][qi], S["n_full"][si]
        qk, sk = Q["n_skel"][qi], S["n_skel"][si]
        qns, sns = Q["n_nospace"][qi], S["n_nospace"][si]
        qa, sa = Q["a_full"][qi], S["a_full"][si]
        f = {}
        f["nm_ratio"] = _cp(qn, sn, fuzz.ratio)
        f["nm_tset"] = _cp(qn, sn, fuzz.token_set_ratio)
        f["nm_tsort"] = _cp(qn, sn, fuzz.token_sort_ratio)
        f["nm_partial"] = _cp(qn, sn, fuzz.partial_ratio)
        f["nm_jw"] = _cp(qn, sn, JaroWinkler.normalized_similarity)
        f["nm_skel_tset"] = _cp(qk, sk, fuzz.token_set_ratio)
        f["nm_skel_ratio"] = _cp(qk, sk, fuzz.ratio)
        f["nm_nospace_ratio"] = _cp(qns, sns, fuzz.ratio)
        f["nm_nospace_partial"] = _cp(qns, sns, fuzz.partial_ratio)
        f["nm_parts_best"] = best_part_similarity(Q["n_parts"][qi], sn, sns, f["nm_tset"])
        f["nm_wjacc"], f["nm_wcos"] = self.name_ov.score(qi, si)
        f["nm_first_eq"] = (self.q_first[qi] == self.s_first[si]).astype(np.float32)
        ql, sl = Q["n_legal"][qi], S["n_legal"][si]
        f["legal_eq"] = np.where((ql == "") | (sl == ""), np.nan, (ql == sl)).astype(np.float32)
        f["q_web"] = Q["n_web"][qi].astype(np.float32)
        f["q_nonlatin"] = Q["n_nonlatin"][qi].astype(np.float32)
        f["q_nm_ntok"] = self.q_ntok[qi]
        f["s_nm_ntok"] = self.s_ntok[si]

        f["ad_ratio"] = _cp(qa, sa, fuzz.ratio)
        f["ad_tset"] = _cp(qa, sa, fuzz.token_set_ratio)
        f["ad_tsort"] = _cp(qa, sa, fuzz.token_sort_ratio)
        f["ad_partial"] = _cp(qa, sa, fuzz.partial_ratio)
        f["ad_jw"] = _cp(qa, sa, JaroWinkler.normalized_similarity)
        f["ad_wjacc"], f["ad_wcos"] = self.addr_ov.score(qi, si)
        f["house"], f["num_jacc"], f["num_q_in_s"], f["num_conflict"] = number_features(
            Q["a_nums"][qi], S["a_nums"][si], Q["a_house"][qi], S["a_house"][si])
        f["q_a_empty"] = Q["a_empty"][qi].astype(np.float32)
        f["s_a_empty"] = S["a_empty"][si].astype(np.float32)

        f["s_name_freq"] = self.s_name_freq[si]
        f["s_addr_freq"] = self.s_addr_freq[si]
        f["q_name_freq"] = self.q_name_freq[qi]

        # position of this candidate within the query's shortlist
        sc = cands["blk_score"].to_numpy(dtype=np.float32)
        rank = cands["blk_rank"].to_numpy()
        best = pd.Series(sc[rank == 1], index=qi[rank == 1])
        second = pd.Series(sc[rank == 2], index=qi[rank == 2])
        b = pd.Series(qi).map(best).to_numpy(dtype=np.float32)
        s2 = pd.Series(qi).map(second).fillna(0.0).to_numpy(dtype=np.float32)
        f["blk_score"] = sc
        f["blk_rank"] = rank.astype(np.float32)
        f["blk_gap_best"] = b - sc
        f["blk_gap_next"] = np.where(rank == 1, sc - s2, 0.0).astype(np.float32)
        close = pd.Series((sc >= b - 0.05).astype(np.float32)).groupby(qi).transform("sum")
        f["n_close"] = close.to_numpy(dtype=np.float32)
        f["is_s3"] = self.is_s3[qi]
        return pd.DataFrame({k: f[k] for k in FEATURES}, index=cands.index)


def iter_query_chunks(cands, chunk_pairs):
    """Split candidates (sorted by q_pos) into blocks of whole query groups."""
    qpos = cands["q_pos"].to_numpy()
    start = 0
    n = len(cands)
    while start < n:
        end = min(start + chunk_pairs, n)
        if end < n:
            # move the cut forward to the next query boundary
            while end < n and qpos[end] == qpos[end - 1]:
                end += 1
        yield slice(start, end)
        start = end


def build_features(cands, fb, chunk_pairs=2_000_000):
    """Features for all candidate pairs (chunked); returns float32 DataFrame."""
    parts = [fb.transform(cands.iloc[sl]) for sl in iter_query_chunks(cands, chunk_pairs)]
    return pd.concat(parts)
