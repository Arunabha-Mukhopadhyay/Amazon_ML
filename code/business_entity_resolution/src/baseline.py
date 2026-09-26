"""
Naive Exact-Match Baseline.

Matches records that share the EXACT same (country, normalized_business_name).
Serves as the sanity check for the scoring and submission validation pipeline.
"""

import os
import sys
import csv
from collections import defaultdict
from typing import Dict, List, Set, Tuple

from src.normalize import normalize_business_name
from src.metric import evaluate_predictions


def run_exact_match_baseline(
    s1_path: str,
    s2_path: str,
    s3_path: str,
    gt_path: str = None,
    output_matching_path: str = None,
    output_candidate_path: str = None,
) -> Dict[str, float]:
    """Run exact match on normalized name + country."""
    print("Building candidate index from Source 2 and Source 3...")
    # Map (country, normalized_name) -> list of entity_ids
    index = defaultdict(list)
    s2_count = 0
    s3_count = 0

    # Index Source 2
    with open(s2_path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4:
                eid, bname, _, country = parts[0], parts[1], parts[2], parts[3]
                norm_name = normalize_business_name(bname)
                if norm_name:
                    index[(country, norm_name)].append(eid)
                    s2_count += 1

    # Index Source 3
    with open(s3_path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4:
                eid, bname, _, country = parts[0], parts[1], parts[2], parts[3]
                norm_name = normalize_business_name(bname)
                if norm_name:
                    index[(country, norm_name)].append(eid)
                    s3_count += 1

    print(f"Indexed {s2_count:,} S2 records and {s3_count:,} S3 records into {len(index):,} unique keys.")

    print(f"Generating matches for Source 1 records from {s1_path}...")
    predictions = {}
    total_s1 = 0
    matched_s1 = 0
    total_preds = 0

    with open(s1_path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4:
                total_s1 += 1
                eid, bname, _, country = parts[0], parts[1], parts[2], parts[3]
                norm_name = normalize_business_name(bname)
                matched_ids = index.get((country, norm_name), [])
                # Ensure deduplicated list preserving order
                seen = set()
                unique_matches = []
                for mid in matched_ids:
                    if mid not in seen:
                        seen.add(mid)
                        unique_matches.append(mid)

                predictions[eid] = set(unique_matches)
                if unique_matches:
                    matched_s1 += 1
                    total_preds += len(unique_matches)

    print(f"Processed {total_s1:,} Source 1 entities.")
    print(f"Entities with >= 1 match: {matched_s1:,} ({matched_s1 / total_s1 * 100:.2f}%)")
    print(f"Total predicted pairs: {total_preds:,}")

    # Write output files if paths provided
    if output_matching_path:
        os.makedirs(os.path.dirname(output_matching_path), exist_ok=True)
        print(f"Writing matching results to {output_matching_path}...")
        with open(output_matching_path, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
            for eid in sorted(predictions.keys()):
                m_str = ",".join(sorted(predictions[eid]))
                f.write(f"{eid}\t{m_str}\n")

    if output_candidate_path:
        os.makedirs(os.path.dirname(output_candidate_path), exist_ok=True)
        print(f"Writing candidate pairs to {output_candidate_path}...")
        with open(output_candidate_path, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tcandidate_entity_ids\n")
            for eid in sorted(predictions.keys()):
                m_str = ",".join(sorted(predictions[eid]))
                f.write(f"{eid}\t{m_str}\n")

    # Evaluate against ground truth if provided
    metrics = None
    if gt_path and os.path.isfile(gt_path):
        print(f"Loading ground truth from {gt_path}...")
        ground_truth = {}
        with open(gt_path, "r", encoding="utf-8") as f:
            f.readline()
            for line in f:
                parts = line.rstrip("\n").split("\t")
                eid = parts[0]
                m_set = set()
                if len(parts) > 1 and parts[1].strip():
                    for mid in parts[1].split(","):
                        mid = mid.strip()
                        if mid:
                            m_set.add(mid)
                ground_truth[eid] = m_set

        # Only evaluate on entities present in s1_path
        eval_gt = {eid: ground_truth.get(eid, set()) for eid in predictions}
        metrics = evaluate_predictions(eval_gt, predictions)
        print("\n=== EXACT-MATCH BASELINE EVALUATION RESULTS ===")
        print(f"Macro F_0.5 Score:     {metrics['macro_f05']:.4f}")
        print(f"Macro Precision:       {metrics['macro_precision']:.4f}")
        print(f"Macro Recall:          {metrics['macro_recall']:.4f}")
        print(f"Evaluated Entities:    {metrics['num_entities']:,}")
        print(f"Singletons:            {metrics['total_singletons']:,}")
        print(f"Singleton Accuracy:    {metrics['singleton_accuracy']:.4f}")

    return metrics


if __name__ == "__main__":
    base_dir = "/Users/arunabhamukhopadhyay/Desktop/student_resource"

    # Test baseline on our sample dataset first
    sample_dir = os.path.join(base_dir, "dataset/sample")
    out_matching = os.path.join(base_dir, "output/sample_matching_results.tsv")
    out_candidate = os.path.join(base_dir, "output/sample_candidate_pairs.tsv")

    run_exact_match_baseline(
        s1_path=os.path.join(sample_dir, "sample_source1.tsv"),
        s2_path=os.path.join(sample_dir, "sample_source2.tsv"),
        s3_path=os.path.join(sample_dir, "sample_source3.tsv"),
        gt_path=os.path.join(sample_dir, "sample_ground_truth.tsv"),
        output_matching_path=out_matching,
        output_candidate_path=out_candidate,
    )
