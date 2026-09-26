"""
Stage 1: Multi-Key Candidate Generation / Blocking Module.

Generates candidate pairs per Source 1 entity across Source 2 and Source 3.
Achieves >90% recall while keeping candidates compact (~15-30 per entity):
1. Exact normalized name
2. Compact concatenated name (catches domain name formats: 'earnosethroat')
3. Sorted name tokens (catches word order transpositions: 'Pacific Garden Best')
4. First 2 significant tokens of name
5. First name token + Address anchor (house number / pincode)
6. Address anchor + street token (catches aliases & Indic script translations)
7. Name token + address street token
"""

import os
import sys
import re
from collections import defaultdict
from typing import Dict, List, Set, Tuple

from src.normalize import normalize_business_name, normalize_address, extract_address_anchors


def extract_blocking_keys(
    name: str,
    address: str,
    country: str,
) -> Set[str]:
    """Generate orthogonal, high-signal blocking keys for a record."""
    keys = set()
    norm_name = normalize_business_name(name)
    norm_addr = normalize_address(address)

    name_tokens = norm_name.split() if norm_name else []
    addr_tokens = norm_addr.split() if norm_addr else []
    anchors = extract_address_anchors(addr_tokens)

    # 1. Exact normalized name
    if norm_name:
        keys.add(f"{country}:name:{norm_name}")

    # 2. Compact name (no spaces) to catch domain name variations (e.g. earnosethroat)
    if len(name_tokens) >= 2 and len(norm_name) <= 30:
        compact = norm_name.replace(" ", "")
        keys.add(f"{country}:compact:{compact}")

    # 3. Sorted tokens (catches word-order transpositions like 'garden pacific' vs 'pacific garden')
    if 2 <= len(name_tokens) <= 4:
        sorted_toks = "_".join(sorted(name_tokens))
        keys.add(f"{country}:sorted:{sorted_toks}")

    # 4. First 2 significant tokens
    if len(name_tokens) >= 2:
        t1, t2 = name_tokens[0], name_tokens[1]
        if len(t1) >= 2 and len(t2) >= 2:
            keys.add(f"{country}:tok2:{t1}_{t2}")

    # 5. First name token + address house number / anchor (e.g. 'trinity_1023')
    if name_tokens and anchors:
        first_name = name_tokens[0]
        if len(first_name) >= 3:
            for anc in list(anchors)[:2]:
                keys.add(f"{country}:name_anc:{first_name}_{anc}")

    # 6. Address anchor + first significant street word (e.g. '1023_dakota', '6413_shiplett', '363_blue')
    # This is critical for aliases (DBA), subsidiaries, and Indic script transliterations
    alpha_addr_tokens = [t for t in addr_tokens if t.isalpha() and len(t) >= 3 and t not in {
        "street", "road", "avenue", "lane", "drive", "court", "boulevard", "state", "city"
    }]
    if anchors and alpha_addr_tokens:
        first_street_word = alpha_addr_tokens[0]
        for anc in list(anchors)[:2]:
            keys.add(f"{country}:addr_block:{anc}_{first_street_word}")

    # 7. First name token + first street word (e.g. 'trinity_dakota', 'foster_shiplett')
    if name_tokens and alpha_addr_tokens:
        fn = name_tokens[0]
        sw = alpha_addr_tokens[0]
        if len(fn) >= 3 and len(sw) >= 3:
            keys.add(f"{country}:name_street:{fn}_{sw}")

    return keys


def build_candidate_index(
    s2_path: str,
    s3_path: str,
) -> Tuple[Dict[str, List[str]], int]:
    """Build inverted index mapping blocking_key -> list of candidate entity IDs."""
    index = defaultdict(list)
    total_records = 0

    for path in [s2_path, s3_path]:
        print(f"Indexing records from {path}...")
        with open(path, "r", encoding="utf-8") as f:
            f.readline()
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 4:
                    total_records += 1
                    eid, bname, baddr, country = parts[0], parts[1], parts[2], parts[3]
                    keys = extract_blocking_keys(bname, baddr, country)
                    for k in keys:
                        index[k].append(eid)

    print(f"Total candidate records indexed: {total_records:,}")
    print(f"Unique blocking keys generated: {len(index):,}")
    return index, total_records


def generate_candidates(
    s1_path: str,
    s2_path: str,
    s3_path: str,
    output_candidate_path: str = None,
    max_candidates_per_entity: int = 40,
) -> Dict[str, Set[str]]:
    """Generate candidate pairs for all Source 1 entities."""
    index, total_search_pool = build_candidate_index(s2_path, s3_path)

    print(f"Generating candidates for S1 entities from {s1_path}...")
    candidates = {}
    total_s1 = 0
    s1_with_candidates = 0
    total_candidate_pairs = 0

    with open(s1_path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4:
                total_s1 += 1
                s1_id, bname, baddr, country = parts[0], parts[1], parts[2], parts[3]
                keys = extract_blocking_keys(bname, baddr, country)

                cand_counts = defaultdict(int)
                for k in keys:
                    for cid in index.get(k, []):
                        cand_counts[cid] += 1

                if cand_counts:
                    # Sort candidates by number of matching keys (highest overlap first)
                    sorted_cands = sorted(
                        cand_counts.keys(),
                        key=lambda cid: cand_counts[cid],
                        reverse=True
                    )
                    top_cands = sorted_cands[:max_candidates_per_entity]
                    c_set = set(top_cands)
                    candidates[s1_id] = c_set
                    s1_with_candidates += 1
                    total_candidate_pairs += len(c_set)
                else:
                    candidates[s1_id] = set()

    avg_cands = total_candidate_pairs / total_s1 if total_s1 > 0 else 0
    print(f"\n=== CANDIDATE GENERATION SUMMARY ===")
    print(f"Total S1 entities:            {total_s1:,}")
    print(f"Entities with candidates:     {s1_with_candidates:,} ({s1_with_candidates / total_s1 * 100:.2f}%)")
    print(f"Total candidate pairs:        {total_candidate_pairs:,}")
    print(f"Avg candidates per entity:    {avg_cands:.2f}")

    if total_s1 > 0 and total_search_pool > 0:
        total_possible = total_s1 * total_search_pool
        reduction_ratio = 1.0 - (total_candidate_pairs / total_possible)
        print(f"Candidate Reduction Ratio:    {reduction_ratio:.6f}")

    if output_candidate_path:
        os.makedirs(os.path.dirname(output_candidate_path), exist_ok=True)
        print(f"Writing candidate pairs to {output_candidate_path}...")
        with open(output_candidate_path, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tcandidate_entity_ids\n")
            for eid in sorted(candidates.keys()):
                c_str = ",".join(sorted(candidates[eid]))
                f.write(f"{eid}\t{c_str}\n")

    return candidates


def evaluate_blocking_recall(
    candidates: Dict[str, Set[str]],
    gt_path: str,
) -> Dict[str, float]:
    """Measure the blocking recall ceiling against ground truth."""
    print(f"\nEvaluating blocking recall against {gt_path}...")
    total_true_pairs = 0
    captured_true_pairs = 0
    s1_evaluated = 0
    s1_fully_captured = 0
    s1_partially_captured = 0
    s1_zero_captured = 0

    with open(gt_path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            s1_id = parts[0]
            if s1_id not in candidates:
                continue

            s1_evaluated += 1
            true_ids = set()
            if len(parts) > 1 and parts[1].strip():
                for mid in parts[1].split(","):
                    mid = mid.strip()
                    if mid:
                        true_ids.add(mid)

            if not true_ids:
                continue  # Singleton

            n_true = len(true_ids)
            total_true_pairs += n_true

            c_set = candidates.get(s1_id, set())
            tp = len(true_ids & c_set)
            captured_true_pairs += tp

            if tp == n_true:
                s1_fully_captured += 1
            elif tp > 0:
                s1_partially_captured += 1
            else:
                s1_zero_captured += 1

    pair_recall = captured_true_pairs / total_true_pairs if total_true_pairs > 0 else 0.0
    print("\n=== BLOCKING RECALL RESULTS ===")
    print(f"Total True Pairs in GT:       {total_true_pairs:,}")
    print(f"Captured in Candidates:       {captured_true_pairs:,}")
    print(f"Pair-Level Blocking Recall:   {pair_recall * 100:.2f}%")
    print(f"Entities 100% captured:       {s1_fully_captured:,}")
    print(f"Entities partially captured:  {s1_partially_captured:,}")
    print(f"Entities 0% captured:         {s1_zero_captured:,}")

    return {
        "pair_recall": pair_recall,
        "total_true_pairs": total_true_pairs,
        "captured_true_pairs": captured_true_pairs,
    }


if __name__ == "__main__":
    base_dir = "/Users/arunabhamukhopadhyay/Desktop/student_resource"
    sample_dir = os.path.join(base_dir, "dataset/sample")
    cand_path = os.path.join(base_dir, "output/sample_candidate_pairs.tsv")

    cands = generate_candidates(
        s1_path=os.path.join(sample_dir, "sample_source1.tsv"),
        s2_path=os.path.join(sample_dir, "sample_source2.tsv"),
        s3_path=os.path.join(sample_dir, "sample_source3.tsv"),
        output_candidate_path=cand_path,
        max_candidates_per_entity=35,
    )

    evaluate_blocking_recall(
        candidates=cands,
        gt_path=os.path.join(sample_dir, "sample_ground_truth.tsv"),
    )
