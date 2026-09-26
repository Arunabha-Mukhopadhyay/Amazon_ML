"""Pair features ("clues") for (query, Source-1 candidate) pairs.

All string similarities are computed with RapidFuzz's vectorised ``cpdist``
(element-wise, multi-threaded). IDF-weighted overlaps are computed with sparse
matrices. No feature uses the country label, so the model cannot learn
country-specific shortcuts and treats unseen countries (France) the same way.

``FeatureBuilder`` precomputes everything that depends on a whole record
collection once (IDF matrices, name/address frequencies), then turns candidate
pairs into features chunk by chunk, so memory stays bounded at full scale.
"""

import math

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein
from scipy import sparse

from .normalize import skeleton

ALIGN_FEATURES = ["q_unm_n", "s_unm_n", "q_unm_max", "s_unm_max", "q_cov", "s_cov", "q_soft", "both_unm"]
MARGIN_OF = ["nm_tset", "nm_soft", "ad_tset", "ad_s_cov", "nm_s_cov", "name_cos", "addr_cos", "comb_cos"]

FEATURES = [
    "nm_ratio", "nm_tset", "nm_tsort", "nm_partial", "nm_jw", "nm_skel_tset", "nm_skel_ratio",
    "nm_nospace_ratio", "nm_nospace_partial", "nm_parts_best", "nm_wjacc", "nm_wcos",
    "nm_first_eq", "legal_eq", "q_web", "q_nonlatin", "q_nm_ntok", "s_nm_ntok",
    "ad_ratio", "ad_tset", "ad_tsort", "ad_partial", "ad_jw", "ad_wjacc", "ad_wcos",
    "house", "num_jacc", "num_q_in_s", "num_conflict", "q_a_empty", "s_a_empty",
    "house_lev", "house_prefix", "house_absdiff", "num_near", "num_s_unm",
    "s_name_freq", "s_addr_freq", "q_name_freq",
    "blk_score", "blk_rank", "blk_gap_best", "blk_gap_next", "n_close", "is_s3",
    "found_by", "name_cos", "addr_cos", "name_rank", "addr_rank", "n_cands",
] + [f"nm_{a}" for a in ALIGN_FEATURES] + [f"ad_{a}" for a in ALIGN_FEATURES] + [f"mg_{m}" for m in MARGIN_OF]


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
    house_lev / house_prefix / house_absdiff: how different the two house
    numbers are (edit distance, one a prefix of the other as in 206 vs 2065,
    log absolute difference) - typo-like differences vs a different building.
    num_jacc: Jaccard of the number sets. q_in_s: share of the query's numbers
    found in the candidate's. conflict: numbers present on only one side,
    counted on the side with fewer such numbers (capped at 5). near: query
    numbers without an exact partner but one edit away from a candidate
    number. s_unm: candidate numbers missing from the query (capped at 5).
    """
    n = len(q_nums)
    nan = lambda: np.full(n, np.nan, dtype=np.float32)  # noqa: E731
    house, h_lev, h_pre, h_diff, jacc, q_in_s = nan(), nan(), nan(), nan(), nan(), nan()
    conflict = np.zeros(n, dtype=np.float32)
    near = np.zeros(n, dtype=np.float32)
    s_unm = np.zeros(n, dtype=np.float32)
    lev = Levenshtein.distance
    for i in range(n):
        qh, sh = q_house[i], s_house[i]
        if qh and sh:
            same = qh == sh
            house[i] = 1.0 if same else 0.0
            h_lev[i] = 0.0 if same else lev(qh, sh)
            h_pre[i] = 1.0 if (not same and (qh.startswith(sh) or sh.startswith(qh))) else 0.0
            h_diff[i] = math.log1p(abs(int(qh[:9]) - int(sh[:9])))
        qn, sn = q_nums[i], s_nums[i]
        if qn or sn:
            a, b = set(qn.split()), set(sn.split())
            inter = len(a & b)
            jacc[i] = inter / len(a | b)
            if a:
                q_in_s[i] = inter / len(a)
            only_a, only_b = a - b, b - a
            conflict[i] = min(5, len(only_a), len(only_b))
            s_unm[i] = min(5, len(only_b))
            if only_a and only_b:
                near[i] = sum(1 for x in only_a if any(lev(x, y) <= 1 for y in only_b))
    return house, jacc, q_in_s, conflict, h_lev, h_pre, h_diff, near, s_unm


class TokenAligner:
    """Word-by-word fuzzy alignment between query and candidate token lists.

    Two words match when they are equal, share a phonetic skeleton, or have a
    RapidFuzz ratio >= ``thr``. For each pair it reports how many words on each
    side found no partner, how distinctive (IDF) the most distinctive unmatched
    word is, and the IDF-weighted share of each side that is covered. A word
    substituted on *both* sides ("Gold Lotus Healthcare" vs "Gold Lotus
    Technology") is the typical signature of a decoy; generic words added by
    noise ("Center", "Services") carry little IDF.

    Everything is vectorised: all token pairs of a block of record pairs are
    laid out in flat arrays and compared with ``cpdist`` in one call.
    """

    def __init__(self, q_docs, s_docs, drop_df_frac=None, skeleton_fn=None, thr=80.0):
        vocab = {}
        lists = []
        for docs in (q_docs, s_docs):
            ptr = np.zeros(len(docs) + 1, dtype=np.int64)
            flat = []
            for i, d in enumerate(docs):
                for t in dict.fromkeys(d):
                    j = vocab.get(t)
                    if j is None:
                        j = vocab[t] = len(vocab)
                    flat.append(j)
                ptr[i + 1] = len(flat)
            lists.append((ptr, np.asarray(flat, dtype=np.int32)))
        n_docs = len(q_docs) + len(s_docs)
        df = np.bincount(np.concatenate([lists[0][1], lists[1][1]]), minlength=len(vocab))
        idf = (np.log((n_docs + 1) / (df + 1)) + 1.0).astype(np.float32)
        if drop_df_frac:
            keep = df <= drop_df_frac * n_docs
            lists = [self._filter(ptr, ids, keep) for ptr, ids in lists]
        (self.q_ptr, self.q_ids), (self.s_ptr, self.s_ids) = lists
        words = list(vocab)
        self.vocab_str = np.array(words, dtype=object)
        self.tok_len = np.array([len(w) for w in words], dtype=np.int32)
        self.idf = idf
        self.thr = thr
        if skeleton_fn is not None:
            skel = {}
            self.skel_id = np.array([skel.setdefault(skeleton_fn(w), len(skel)) for w in words], dtype=np.int32)
        else:
            self.skel_id = None

    @staticmethod
    def _filter(ptr, ids, keep):
        mask = keep[ids]
        doc_of = np.repeat(np.arange(len(ptr) - 1), np.diff(ptr))
        counts = np.bincount(doc_of[mask], minlength=len(ptr) - 1)
        new_ptr = np.zeros(len(ptr), dtype=np.int64)
        new_ptr[1:] = np.cumsum(counts)
        return new_ptr, ids[mask]

    def align(self, qi, si):
        """Alignment features for record pairs (q_docs[qi], s_docs[si])."""
        n = len(qi)
        mq = self.q_ptr[qi + 1] - self.q_ptr[qi]
        ms = self.s_ptr[si + 1] - self.s_ptr[si]
        ntp = mq * ms
        pstart = np.cumsum(ntp) - ntp
        k = np.arange(int(ntp.sum()), dtype=np.int64) - np.repeat(pstart, ntp)
        msr = np.repeat(ms, ntp)
        j = k // np.maximum(msr, 1)
        l = k - j * msr
        tq = self.q_ids[np.repeat(self.q_ptr[qi], ntp) + j]
        ts = self.s_ids[np.repeat(self.s_ptr[si], ntp) + l]
        score = np.zeros(len(k), dtype=np.float32)
        eq = tq == ts
        if self.skel_id is not None:
            eq |= self.skel_id[tq] == self.skel_id[ts]
        score[eq] = 100.0
        lq, ls = self.tok_len[tq], self.tok_len[ts]
        # a ratio >= thr is impossible when the lengths differ too much
        cand = ~eq & (3 * np.minimum(lq, ls) >= 2 * np.maximum(lq, ls))
        if cand.any():
            score[cand] = process.cpdist(self.vocab_str[tq[cand]], self.vocab_str[ts[cand]],
                                         scorer=fuzz.ratio, workers=-1, dtype=np.float32)
        matched = score >= self.thr

        # per query-token instance
        qoff = np.cumsum(mq) - mq
        nq_inst = int(mq.sum())
        q_pair = np.repeat(np.arange(n), mq)
        q_j = np.arange(nq_inst) - np.repeat(qoff, mq)
        q_tok = self.q_ids[np.repeat(self.q_ptr[qi], mq) + q_j]
        q_match = np.bincount(np.repeat(qoff, ntp) + j, weights=matched, minlength=nq_inst) > 0
        q_best = np.zeros(nq_inst, dtype=np.float32)
        has = np.repeat(ms > 0, mq)
        if has.any():
            seg = (np.repeat(pstart, mq) + q_j * np.repeat(ms, mq))[has]
            q_best[has] = np.maximum.reduceat(score, seg)
        # per candidate-token instance
        soff = np.cumsum(ms) - ms
        ns_inst = int(ms.sum())
        s_pair = np.repeat(np.arange(n), ms)
        s_tok = self.s_ids[np.repeat(self.s_ptr[si], ms) + (np.arange(ns_inst) - np.repeat(soff, ms))]
        s_match = np.bincount(np.repeat(soff, ntp) + l, weights=matched, minlength=ns_inst) > 0

        wq, ws = self.idf[q_tok], self.idf[s_tok]
        q_tot = np.bincount(q_pair, weights=wq, minlength=n)
        s_tot = np.bincount(s_pair, weights=ws, minlength=n)
        q_unw = wq * ~q_match
        s_unw = ws * ~s_match
        out = {
            "q_unm_n": np.bincount(q_pair, weights=~q_match, minlength=n),
            "s_unm_n": np.bincount(s_pair, weights=~s_match, minlength=n),
            "q_unm_max": self._seg_max(q_unw, qoff, mq, n),
            "s_unm_max": self._seg_max(s_unw, soff, ms, n),
        }
        with np.errstate(divide="ignore", invalid="ignore"):
            out["q_cov"] = np.where(q_tot > 0, 1 - np.bincount(q_pair, weights=q_unw, minlength=n) / q_tot, np.nan)
            out["s_cov"] = np.where(s_tot > 0, 1 - np.bincount(s_pair, weights=s_unw, minlength=n) / s_tot, np.nan)
            out["q_soft"] = np.where(q_tot > 0, np.bincount(q_pair, weights=wq * q_best / 100, minlength=n) / q_tot,
                                     np.nan)
        out["both_unm"] = np.minimum(out["q_unm_max"], out["s_unm_max"])
        return {k: v.astype(np.float32) for k, v in out.items()}

    @staticmethod
    def _seg_max(values, off, length, n):
        res = np.zeros(n, dtype=np.float32)
        nz = length > 0
        if nz.any():
            res[nz] = np.maximum.reduceat(values, off[nz])
        return res


def _addr_words(series):
    """Non-numeric address words of at least two letters."""
    return [[w for w in a.split() if len(w) >= 2 and not w.isdigit()] for a in series]


def margins(values, q):
    """value minus the best value among the *other* candidates of the same query.

    ``q`` must be sorted so each query's rows are contiguous. A query with a
    single candidate gets its own value (no competitor). NaN counts as -inf.
    """
    v = np.where(np.isnan(values), -np.inf, values).astype(np.float64)
    starts = np.r_[0, np.flatnonzero(q[1:] != q[:-1]) + 1]
    sizes = np.diff(np.r_[starts, len(q)])
    grp = np.repeat(np.arange(len(starts)), sizes)
    gmax = np.maximum.reduceat(v, starts)
    is_max = v == gmax[grp]
    n_max = np.bincount(grp, weights=is_max)
    gsec = np.maximum.reduceat(np.where(is_max, -np.inf, v), starts)
    gsec = np.where(n_max >= 2, gmax, gsec)
    other = np.where(is_max, gsec[grp], gmax[grp])
    with np.errstate(invalid="ignore"):
        out = np.where(np.isfinite(other), v - other, v)
    out = np.where(np.isfinite(out), out, np.nan)
    return out.astype(np.float32)


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
        # word-by-word alignment: names (all core words, phonetic-skeleton aware) and
        # addresses (non-numeric words, without the most common ones such as "st")
        self.name_al = TokenAligner(q["n_full"].str.split().tolist(), s1["n_full"].str.split().tolist(),
                                    skeleton_fn=skeleton)
        self.addr_al = TokenAligner(_addr_words(q["a_full"]), _addr_words(s1["a_full"]), drop_df_frac=0.01)

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
        (f["house"], f["num_jacc"], f["num_q_in_s"], f["num_conflict"], f["house_lev"], f["house_prefix"],
         f["house_absdiff"], f["num_near"], f["num_s_unm"]) = number_features(
            Q["a_nums"][qi], S["a_nums"][si], Q["a_house"][qi], S["a_house"][si])
        f["q_a_empty"] = Q["a_empty"][qi].astype(np.float32)
        f["s_a_empty"] = S["a_empty"][si].astype(np.float32)
        for prefix, aligner in (("nm", self.name_al), ("ad", self.addr_al)):
            for k, v in aligner.align(qi, si).items():
                f[f"{prefix}_{k}"] = v
        f["nm_soft"] = f["nm_q_soft"]

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
        f["found_by"] = cands["found_by"].to_numpy(dtype=np.float32)
        for p in ("name", "addr", "comb"):
            f[f"{p}_cos"] = cands[f"{p}_cos"].to_numpy(dtype=np.float32)
        f["name_rank"] = cands["name_rank"].to_numpy(dtype=np.float32)
        f["addr_rank"] = cands["addr_rank"].to_numpy(dtype=np.float32)
        f["n_cands"] = pd.Series(qi).groupby(qi).transform("size").to_numpy(dtype=np.float32)
        # how this candidate compares with the query's other candidates
        for m in MARGIN_OF:
            f[f"mg_{m}"] = margins(np.asarray(f[m], dtype=np.float32), qi)
        return pd.DataFrame({k: np.asarray(f[k], dtype=np.float32) for k in FEATURES}, index=cands.index)


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
