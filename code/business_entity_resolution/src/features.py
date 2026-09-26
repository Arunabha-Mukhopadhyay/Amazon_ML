"""
Feature Engineering Module for Entity Resolution.

Computes 14 fine-grained similarity, conflict, and compatibility features:
1. name_exact: Exact normalized name match
2. name_compact_match: Space-stripped match (e.g. rapidgoldenoxley <-> rapid golden oxley)
3. name_jaccard: Token set Jaccard
4. name_overlap: Subset/superset token overlap
5. name_trigram_jaccard: Character trigram overlap (typo resistance)
6. name_edit_sim: Normalized Levenshtein edit distance
7. name_len_ratio: Ratio of shorter to longer name length
8. addr_exact: Exact address match
9. addr_jaccard: Address token Jaccard
10. addr_overlap: Address token overlap
11. addr_anchor_match: Shared house number or PIN code
12. addr_anchor_conflict: Both have numbers/anchors, but NONE match (crucial for Problem 1: same name, different business)
13. addr_is_empty: Missing address indicator (enables stricter name requirements for Problem 5)
14. composite_sim: Blended name and address score
"""

import math
from typing import Dict, List, Set, Tuple

try:
    from rapidfuzz.distance import Levenshtein as rf_lev
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False

FEATURE_NAMES = [
    "name_exact",
    "name_compact_match",
    "name_jaccard",
    "name_overlap",
    "name_trigram_jaccard",
    "name_edit_sim",
    "name_len_ratio",
    "addr_exact",
    "addr_jaccard",
    "addr_overlap",
    "addr_anchor_match",
    "addr_anchor_conflict",
    "addr_is_empty",
    "composite_sim",
]


def char_trigrams(s: str) -> Set[str]:
    """Extract character trigrams from string."""
    if len(s) < 3:
        return {s} if s else set()
    return {s[i:i+3] for i in range(len(s) - 2)}


def fast_levenshtein_sim(s1: str, s2: str) -> float:
    """Compute normalized Levenshtein similarity in [0, 1]."""
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    if HAS_RAPIDFUZZ:
        return rf_lev.normalized_similarity(s1, s2)

    len1, len2 = len(s1), len(s2)
    max_len = max(len1, len2)
    if abs(len1 - len2) > max_len * 0.5:
        return 1.0 - (abs(len1 - len2) / max_len)

    v0 = list(range(len2 + 1))
    v1 = [0] * (len2 + 1)

    for i in range(len1):
        v1[0] = i + 1
        c1 = s1[i]
        for j in range(len2):
            cost = 0 if c1 == s2[j] else 1
            v1[j + 1] = min(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)
        v0, v1 = v1, v0

    dist = v0[len2]
    return max(0.0, 1.0 - (dist / max_len))


def compute_pair_features(
    norm_name1: str,
    norm_addr1: str,
    anchors1: Set[str],
    norm_name2: str,
    norm_addr2: str,
    anchors2: Set[str],
) -> List[float]:
    """Compute 14 similarity and conflict features for a candidate pair."""
    # 1. Name exact match
    name_exact = 1.0 if norm_name1 and norm_name1 == norm_name2 else 0.0

    # 2. Name compact match (space-stripped)
    c1 = norm_name1.replace(" ", "")
    c2 = norm_name2.replace(" ", "")
    name_compact_match = 1.0 if c1 and c1 == c2 else 0.0

    # Tokens
    toks1 = set(norm_name1.split()) if norm_name1 else set()
    toks2 = set(norm_name2.split()) if norm_name2 else set()

    # 3. Name token Jaccard
    intersection = len(toks1 & toks2)
    union = len(toks1 | toks2)
    name_jaccard = intersection / union if union > 0 else 0.0

    # 4. Name overlap coefficient
    min_tokens = min(len(toks1), len(toks2)) if toks1 and toks2 else 0
    name_overlap = intersection / min_tokens if min_tokens > 0 else 0.0

    # 5. Trigram Jaccard
    tri1 = char_trigrams(norm_name1)
    tri2 = char_trigrams(norm_name2)
    tri_inter = len(tri1 & tri2)
    tri_union = len(tri1 | tri2)
    name_trigram_jaccard = tri_inter / tri_union if tri_union > 0 else 0.0

    # 6. Edit distance similarity
    name_edit_sim = fast_levenshtein_sim(norm_name1, norm_name2)

    # 7. Name length ratio
    l1, l2 = len(norm_name1), len(norm_name2)
    name_len_ratio = (min(l1, l2) / max(l1, l2)) if max(l1, l2) > 0 else 0.0

    # Address features
    addr_is_empty = 1.0 if not norm_addr2 else 0.0
    addr_exact = 1.0 if norm_addr1 and norm_addr1 == norm_addr2 else 0.0

    atok1 = set(norm_addr1.split()) if norm_addr1 else set()
    atok2 = set(norm_addr2.split()) if norm_addr2 else set()

    ainter = len(atok1 & atok2)
    aunion = len(atok1 | atok2)
    addr_jaccard = ainter / aunion if aunion > 0 else 0.0

    amin = min(len(atok1), len(atok2)) if atok1 and atok2 else 0
    addr_overlap = ainter / amin if amin > 0 else 0.0

    # 11. Address anchor match (shared house number / pin)
    shared_anchors = len(anchors1 & anchors2)
    addr_anchor_match = 1.0 if shared_anchors > 0 else 0.0

    # 12. Address anchor conflict (both have numbers, but they DISAGREE!)
    # Crucial for Problem 1: prevents matching different businesses with same name at different house numbers
    addr_anchor_conflict = 1.0 if (anchors1 and anchors2 and shared_anchors == 0) else 0.0

    # 14. Composite similarity
    if norm_addr1 and norm_addr2:
        composite = 0.55 * name_jaccard + 0.45 * addr_jaccard
    else:
        composite = name_jaccard

    return [
        name_exact,
        name_compact_match,
        name_jaccard,
        name_overlap,
        name_trigram_jaccard,
        name_edit_sim,
        name_len_ratio,
        addr_exact,
        addr_jaccard,
        addr_overlap,
        addr_anchor_match,
        addr_anchor_conflict,
        addr_is_empty,
        composite,
    ]
