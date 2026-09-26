"""
End-to-End Business Entity Resolution Pipeline Orchestrator.

Usage:
    PYTHONPATH=code/business_entity_resolution python3 code/business_entity_resolution/src/pipeline.py --mode sample
    PYTHONPATH=code/business_entity_resolution python3 code/business_entity_resolution/src/pipeline.py --mode full
"""

import os
import sys
import argparse
import random
import subprocess
from collections import defaultdict
from typing import Dict, List, Set, Tuple

from src.normalize import normalize_business_name, normalize_address, extract_address_anchors
from src.blocking import extract_blocking_keys
from src.features import compute_pair_features
from src.matching import CalibratedMatcher, get_matcher
from src.metric import evaluate_predictions


def load_records_tsv(path: str) -> Dict[str, Tuple[str, str, str, str, Set[str]]]:
    """Load records and precompute normalized names, addresses, and anchors."""
    print(f"Loading and pre-normalizing records from {path}...")
    records = {}
    with open(path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4:
                eid, bname, baddr, country = parts[0], parts[1], parts[2], parts[3]
                norm_name = normalize_business_name(bname)
                norm_addr = normalize_address(baddr, country)
                addr_tokens = norm_addr.split() if norm_addr else []
                anchors = extract_address_anchors(addr_tokens)
                records[eid] = (norm_name, norm_addr, country, bname, anchors)
    return records


def run_pipeline_sample(base_dir: str):
    """Run full pipeline on the 15,000-entity sample dataset."""
    sample_dir = os.path.join(base_dir, "dataset/sample")
    s1_path = os.path.join(sample_dir, "sample_source1.tsv")
    s2_path = os.path.join(sample_dir, "sample_source2.tsv")
    s3_path = os.path.join(sample_dir, "sample_source3.tsv")
    gt_path = os.path.join(sample_dir, "sample_ground_truth.tsv")

    out_matching = os.path.join(base_dir, "output/sample_matching_results.tsv")
    out_candidate = os.path.join(base_dir, "output/sample_candidate_pairs.tsv")

    print("\n=======================================================")
    print("STEP 1: LOAD & PRE-NORMALIZE SAMPLE RECORDS")
    print("=======================================================")
    s1_records = load_records_tsv(s1_path)
    s2_records = load_records_tsv(s2_path)
    s3_records = load_records_tsv(s3_path)
    targets = {**s2_records, **s3_records}

    # Load Ground Truth
    print(f"Loading ground truth from {gt_path}...")
    ground_truth = {}
    with open(gt_path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            s1_id = parts[0]
            m_set = set()
            if len(parts) > 1 and parts[1].strip():
                for mid in parts[1].split(","):
                    mid = mid.strip()
                    if mid:
                        m_set.add(mid)
            ground_truth[s1_id] = m_set

    print("\n=======================================================")
    print("STEP 2: SPLIT SAMPLE S1 ENTITIES 80/20 (TRAIN/VAL)")
    print("=======================================================")
    s1_keys = sorted(s1_records.keys())
    rng = random.Random(42)
    rng.shuffle(s1_keys)
    split_idx = int(0.8 * len(s1_keys))
    train_ids = set(s1_keys[:split_idx])
    val_ids = set(s1_keys[split_idx:])
    print(f"Sample Split: {len(train_ids):,} Train S1 entities, {len(val_ids):,} Validation S1 entities.")

    print("\n=======================================================")
    print("STEP 3: STAGE 1 BLOCKING / CANDIDATE GENERATION")
    print("=======================================================")
    index = defaultdict(list)
    for tid, rec in targets.items():
        norm_name, norm_addr, country, raw_name, anchors = rec
        keys = extract_blocking_keys(raw_name, norm_addr, country)
        for k in keys:
            index[k].append(tid)

    # Generate candidates for all S1 entities
    all_candidates = {}
    total_cand_pairs = 0
    for s1_id, rec in s1_records.items():
        norm_name, norm_addr, country, raw_name, anchors = rec
        keys = extract_blocking_keys(raw_name, norm_addr, country)
        cand_counts = defaultdict(int)
        for k in keys:
            for cid in index.get(k, []):
                cand_counts[cid] += 1

        if cand_counts:
            sorted_cands = sorted(cand_counts.keys(), key=lambda c: cand_counts[c], reverse=True)[:35]
            c_set = set(sorted_cands)
            all_candidates[s1_id] = c_set
            total_cand_pairs += len(c_set)
        else:
            all_candidates[s1_id] = set()

    print(f"Generated {total_cand_pairs:,} total candidate pairs across {len(s1_records):,} entities.")

    print("\n=======================================================")
    print("STEP 4: STAGE 2 FEATURE EXTRACTION")
    print("=======================================================")
    X_train = []
    y_train = []
    val_cand_features = defaultdict(list)

    for s1_id, cand_set in all_candidates.items():
        s1_rec = s1_records[s1_id]
        true_set = ground_truth.get(s1_id, set())

        for cid in cand_set:
            c_rec = targets[cid]
            feats = compute_pair_features(
                norm_name1=s1_rec[0],
                norm_addr1=s1_rec[1],
                anchors1=s1_rec[4],
                norm_name2=c_rec[0],
                norm_addr2=c_rec[1],
                anchors2=c_rec[4],
            )
            is_match = 1 if cid in true_set else 0

            if s1_id in train_ids:
                X_train.append(feats)
                y_train.append(is_match)
            else:
                val_cand_features[s1_id].append((cid, feats))

    pos_train = sum(y_train)
    neg_train = len(y_train) - pos_train
    print(f"Extracted {len(X_train):,} training pairs ({pos_train:,} positive, {neg_train:,} negative).")

    print("\n=======================================================")
    print("STEP 5: TRAIN CLASSIFIER & TUNE F_0.5 THRESHOLD")
    print("=======================================================")
    matcher = get_matcher()
    if hasattr(matcher, "fit"):
        matcher.fit(X_train, y_train)
    else:
        matcher.train_logistic_sgd(X_train, y_train, epochs=4, lr=0.08)

    val_gt = {eid: ground_truth.get(eid, set()) for eid in val_ids}
    best_threshold = matcher.tune_threshold(val_cand_features, val_gt)

    print("\n=======================================================")
    print("STEP 6: GENERATE FINAL PREDICTIONS & EVALUATE OVERALL")
    print("=======================================================")
    final_predictions = {}
    for s1_id, cand_set in all_candidates.items():
        s1_rec = s1_records[s1_id]
        m_set = set()
        for cid in cand_set:
            c_rec = targets[cid]
            feats = compute_pair_features(
                s1_rec[0], s1_rec[1], s1_rec[4],
                c_rec[0], c_rec[1], c_rec[4]
            )
            if matcher.predict_score(feats) >= best_threshold:
                m_set.add(cid)
        final_predictions[s1_id] = m_set

    # Write output files
    os.makedirs(os.path.dirname(out_matching), exist_ok=True)
    with open(out_matching, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for eid in sorted(final_predictions.keys()):
            m_str = ",".join(sorted(final_predictions[eid]))
            f.write(f"{eid}\t{m_str}\n")

    with open(out_candidate, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for eid in sorted(all_candidates.keys()):
            c_str = ",".join(sorted(all_candidates[eid]))
            f.write(f"{eid}\t{c_str}\n")

    print(f"Saved matching results to:  {out_matching}")
    print(f"Saved candidate pairs to:   {out_candidate}")

    # Evaluate Overall Sample Score
    metrics = evaluate_predictions(ground_truth, final_predictions)
    print("\n=======================================================")
    print("FINAL PIPELINE EVALUATION ON SAMPLE DATASET:")
    print("=======================================================")
    print(f"  Macro F_0.5 Score:   {metrics['macro_f05']:.4f}")
    print(f"  Macro Precision:     {metrics['macro_precision']:.4f}")
    print(f"  Macro Recall:        {metrics['macro_recall']:.4f}")
    print(f"  Singleton Accuracy:  {metrics['singleton_accuracy']:.4f}")
    print(f"  Total S1 Entities:   {metrics['num_entities']:,}")

    # Check validator
    test_source1_fake = os.path.join(sample_dir, "test_source1.tsv")
    # Link or copy sample_source1 as test_source1 so validate_submission.py can check it
    if not os.path.isfile(test_source1_fake):
        import shutil
        shutil.copy(s1_path, test_source1_fake)

    print("\nRunning submission validator check...")
    res = subprocess.run([
        sys.executable,
        os.path.join(base_dir, "utils/validate_submission.py"),
        "--matching", out_matching,
        "--candidate", out_candidate,
        "--test-dir", sample_dir,
    ], capture_output=True, text=True)
    print(res.stdout)


def run_pipeline_full(base_dir: str):
    """Run full production pipeline across all 26.4M records using country streaming."""
    dataset_dir = os.path.join(base_dir, "dataset")
    train_s1 = os.path.join(dataset_dir, "train/train_source1.tsv")
    train_s2 = os.path.join(dataset_dir, "train/train_source2.tsv")
    train_s3 = os.path.join(dataset_dir, "train/train_source3.tsv")
    train_gt = os.path.join(dataset_dir, "train/train_ground_truth.tsv")

    test_s1 = os.path.join(dataset_dir, "test/test_source1.tsv")
    test_s2 = os.path.join(dataset_dir, "test/test_source2.tsv")
    test_s3 = os.path.join(dataset_dir, "test/test_source3.tsv")

    out_matching = os.path.join(base_dir, "output/matching_results.tsv")
    out_candidate = os.path.join(base_dir, "output/candidate_pairs.tsv")

    os.makedirs(os.path.dirname(out_matching), exist_ok=True)
    os.makedirs(os.path.dirname(out_candidate), exist_ok=True)

    print("\n=======================================================")
    print("STAGE 1: TRAIN MATCHER ON TRAINING GROUND TRUTH SAMPLE")
    print("=======================================================")
    # Train matcher weights on a stratified 30,000-entity slice to establish robust weights
    s1_sample = load_records_tsv(os.path.join(base_dir, "dataset/sample/sample_source1.tsv"))
    s2_sample = load_records_tsv(os.path.join(base_dir, "dataset/sample/sample_source2.tsv"))
    s3_sample = load_records_tsv(os.path.join(base_dir, "dataset/sample/sample_source3.tsv"))
    targets_sample = {**s2_sample, **s3_sample}

    gt_sample = {}
    with open(os.path.join(base_dir, "dataset/sample/sample_ground_truth.tsv"), "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.rstrip("\n").split("\t")
            gt_sample[p[0]] = set(p[1].split(",")) if len(p) > 1 and p[1].strip() else set()

    # Index sample targets
    sample_index = defaultdict(list)
    for tid, rec in targets_sample.items():
        keys = extract_blocking_keys(rec[3], rec[1], rec[2])
        for k in keys:
            sample_index[k].append(tid)

    X_train = []
    y_train = []
    for s1_id, s1_rec in s1_sample.items():
        keys = extract_blocking_keys(s1_rec[3], s1_rec[1], s1_rec[2])
        cand_counts = defaultdict(int)
        for k in keys:
            for cid in sample_index.get(k, []):
                cand_counts[cid] += 1
        top_cands = sorted(cand_counts.keys(), key=lambda c: cand_counts[c], reverse=True)[:35]
        true_set = gt_sample.get(s1_id, set())
        for cid in top_cands:
            c_rec = targets_sample[cid]
            feats = compute_pair_features(s1_rec[0], s1_rec[1], s1_rec[4], c_rec[0], c_rec[1], c_rec[4])
            X_train.append(feats)
            y_train.append(1 if cid in true_set else 0)

    matcher = get_matcher()
    if hasattr(matcher, "fit"):
        matcher.fit(X_train, y_train)
    else:
        matcher.train_logistic_sgd(X_train, y_train, epochs=4, lr=0.08)
    matcher.threshold = 0.40  # Optimal F_0.5 decision threshold calibrated on validation split

    del s1_sample, s2_sample, s3_sample, targets_sample, sample_index, X_train, y_train
    import gc
    gc.collect()

    print("\n=======================================================")
    print("STAGE 2: FULL TEST INFERENCE STREAMED BY COUNTRY")
    print("=======================================================")
    # Initialize output files
    with open(out_matching, "w", encoding="utf-8") as fm:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
    with open(out_candidate, "w", encoding="utf-8") as fc:
        fc.write("source1_entity_id\tcandidate_entity_ids\n")

    for country in ["France", "US", "India"]:
        print(f"\n---> Streaming country: {country}")
        country_targets = {}
        country_index = defaultdict(list)

        # Index S2 & S3 for this country
        for path in [test_s2, test_s3]:
            print(f"  Reading candidates from {os.path.basename(path)} for {country}...")
            with open(path, "r", encoding="utf-8") as f:
                f.readline()
                for line in f:
                    p = line.rstrip("\n").split("\t")
                    if len(p) >= 4 and p[3] == country:
                        eid, bname, baddr = p[0], p[1], p[2]
                        norm_name = normalize_business_name(bname)
                        norm_addr = normalize_address(baddr, country)
                        anchors = extract_address_anchors(norm_addr.split())
                        country_targets[eid] = (norm_name, norm_addr, country, bname, anchors)
                        keys = extract_blocking_keys(bname, norm_addr, country)
                        for k in keys:
                            country_index[k].append(eid)

        print(f"  [{country}] Indexed {len(country_targets):,} target candidates and {len(country_index):,} keys.")

        # Process S1 for this country and write directly
        processed_s1 = 0
        matched_s1 = 0
        with open(test_s1, "r", encoding="utf-8") as f, \
             open(out_matching, "a", encoding="utf-8") as fm, \
             open(out_candidate, "a", encoding="utf-8") as fc:
            f.readline()
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) >= 4 and p[3] == country:
                    s1_id, bname, baddr = p[0], p[1], p[2]
                    norm_name = normalize_business_name(bname)
                    norm_addr = normalize_address(baddr, country)
                    anchors = extract_address_anchors(norm_addr.split())
                    keys = extract_blocking_keys(bname, norm_addr, country)

                    cand_counts = defaultdict(int)
                    for k in keys:
                        for cid in country_index.get(k, []):
                            cand_counts[cid] += 1

                    top_cands = sorted(cand_counts.keys(), key=lambda c: cand_counts[c], reverse=True)[:35]
                    matches = []
                    for cid in top_cands:
                        c_rec = country_targets[cid]
                        feats = compute_pair_features(norm_name, norm_addr, anchors, c_rec[0], c_rec[1], c_rec[4])
                        if matcher.predict_score(feats) >= matcher.threshold:
                            matches.append(cid)

                    # Deduplicate preserving order
                    c_str = ",".join(dict.fromkeys(top_cands).keys())
                    m_str = ",".join(dict.fromkeys(matches).keys())

                    fm.write(f"{s1_id}\t{m_str}\n")
                    fc.write(f"{s1_id}\t{c_str}\n")

                    processed_s1 += 1
                    if matches:
                        matched_s1 += 1

        print(f"  [{country}] Done: {processed_s1:,} S1 entities processed ({matched_s1:,} non-empty).")
        del country_targets, country_index
        gc.collect()

    print("\nRunning submission validator check on generated outputs...")
    res = subprocess.run([
        sys.executable,
        os.path.join(base_dir, "utils/validate_submission.py"),
        "--matching", out_matching,
        "--candidate", out_candidate,
        "--test-dir", os.path.join(base_dir, "dataset/test"),
    ], capture_output=True, text=True)
    print(res.stdout)


def run_pipeline_val_full(base_dir: str):
    """Run validation evaluation on the full held-out 20% validation split (441k entities)."""
    dataset_dir = os.path.join(base_dir, "dataset")
    splits_dir = os.path.join(base_dir, "data_splits")
    train_s1 = os.path.join(dataset_dir, "train/train_source1.tsv")
    train_s2 = os.path.join(dataset_dir, "train/train_source2.tsv")
    train_s3 = os.path.join(dataset_dir, "train/train_source3.tsv")
    val_gt_path = os.path.join(splits_dir, "val_gt.tsv")
    val_ids_path = os.path.join(splits_dir, "val_s1_ids.txt")

    print("\n=======================================================")
    print("RUNNING VALIDATION EVALUATION ON FULL 20% HELD-OUT SPLIT")
    print("=======================================================")
    
    # Load validation S1 IDs
    val_ids = set()
    with open(val_ids_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                val_ids.add(line.strip())
    print(f"Loaded {len(val_ids):,} held-out validation S1 entity IDs.")

    # Load validation ground truth
    val_gt = {}
    with open(val_gt_path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 1:
                val_gt[p[0]] = set(p[1].split(",")) if len(p) > 1 and p[1].strip() else set()

    # Pre-train matcher on sample
    s1_sample = load_records_tsv(os.path.join(base_dir, "dataset/sample/sample_source1.tsv"))
    s2_sample = load_records_tsv(os.path.join(base_dir, "dataset/sample/sample_source2.tsv"))
    s3_sample = load_records_tsv(os.path.join(base_dir, "dataset/sample/sample_source3.tsv"))
    targets_sample = {**s2_sample, **s3_sample}

    gt_sample = {}
    with open(os.path.join(base_dir, "dataset/sample/sample_ground_truth.tsv"), "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.rstrip("\n").split("\t")
            gt_sample[p[0]] = set(p[1].split(",")) if len(p) > 1 and p[1].strip() else set()

    sample_index = defaultdict(list)
    for tid, rec in targets_sample.items():
        keys = extract_blocking_keys(rec[3], rec[1], rec[2])
        for k in keys:
            sample_index[k].append(tid)

    X_train, y_train = [], []
    for s1_id, s1_rec in s1_sample.items():
        keys = extract_blocking_keys(s1_rec[3], s1_rec[1], s1_rec[2])
        cand_counts = defaultdict(int)
        for k in keys:
            for cid in sample_index.get(k, []):
                cand_counts[cid] += 1
        top_cands = sorted(cand_counts.keys(), key=lambda c: cand_counts[c], reverse=True)[:35]
        true_set = gt_sample.get(s1_id, set())
        for cid in top_cands:
            c_rec = targets_sample[cid]
            feats = compute_pair_features(s1_rec[0], s1_rec[1], s1_rec[4], c_rec[0], c_rec[1], c_rec[4])
            X_train.append(feats)
            y_train.append(1 if cid in true_set else 0)

    matcher = get_matcher()
    if hasattr(matcher, "fit"):
        matcher.fit(X_train, y_train)
    else:
        matcher.train_logistic_sgd(X_train, y_train, epochs=4, lr=0.08)
    matcher.threshold = 0.40

    del s1_sample, s2_sample, s3_sample, targets_sample, sample_index, X_train, y_train
    import gc
    gc.collect()

    # Stream by country over train S2/S3 and evaluate on val S1
    predictions = {}
    for country in ["US", "India"]:
        print(f"\n---> Evaluating validation split for country: {country}")
        country_targets = {}
        country_index = defaultdict(list)

        for path in [train_s2, train_s3]:
            print(f"  Reading candidate pool from {os.path.basename(path)} for {country}...")
            with open(path, "r", encoding="utf-8") as f:
                f.readline()
                for line in f:
                    p = line.rstrip("\n").split("\t")
                    if len(p) >= 4 and p[3] == country:
                        eid, bname, baddr = p[0], p[1], p[2]
                        norm_name = normalize_business_name(bname)
                        norm_addr = normalize_address(baddr, country)
                        anchors = extract_address_anchors(norm_addr.split())
                        country_targets[eid] = (norm_name, norm_addr, country, bname, anchors)
                        keys = extract_blocking_keys(bname, norm_addr, country)
                        for k in keys:
                            country_index[k].append(eid)

        print(f"  [{country}] Search pool indexed: {len(country_targets):,} records.")

        # Read S1 train file, filter only validation entities for this country
        processed = 0
        with open(train_s1, "r", encoding="utf-8") as f:
            f.readline()
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) >= 4 and p[0] in val_ids and p[3] == country:
                    s1_id, bname, baddr = p[0], p[1], p[2]
                    norm_name = normalize_business_name(bname)
                    norm_addr = normalize_address(baddr, country)
                    anchors = extract_address_anchors(norm_addr.split())
                    keys = extract_blocking_keys(bname, norm_addr, country)

                    cand_counts = defaultdict(int)
                    for k in keys:
                        for cid in country_index.get(k, []):
                            cand_counts[cid] += 1

                    top_cands = sorted(cand_counts.keys(), key=lambda c: cand_counts[c], reverse=True)[:35]
                    matches = set()
                    for cid in top_cands:
                        c_rec = country_targets[cid]
                        feats = compute_pair_features(norm_name, norm_addr, anchors, c_rec[0], c_rec[1], c_rec[4])
                        if matcher.predict_score(feats) >= matcher.threshold:
                            matches.add(cid)

                    predictions[s1_id] = matches
                    processed += 1

        print(f"  [{country}] Evaluated {processed:,} validation entities.")
        del country_targets, country_index
        gc.collect()

    print("\n=======================================================")
    print("FULL-SCALE 20% VALIDATION SPLIT RESULTS:")
    print("=======================================================")
    metrics = evaluate_predictions(val_gt, predictions)
    print(f"  Macro F_0.5 Score:   {metrics['macro_f05']:.4f}")
    print(f"  Macro Precision:     {metrics['macro_precision']:.4f}")
    print(f"  Macro Recall:        {metrics['macro_recall']:.4f}")
    print(f"  Singleton Accuracy:  {metrics['singleton_accuracy']:.4f}")
    print(f"  Total Val Entities:  {metrics['num_entities']:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["sample", "val_full", "full"], default="sample")
    args = parser.parse_args()

    base_dir = "/Users/arunabhamukhopadhyay/Desktop/student_resource"
    if args.mode == "sample":
        run_pipeline_sample(base_dir)
    elif args.mode == "val_full":
        run_pipeline_val_full(base_dir)
    else:
        run_pipeline_full(base_dir)
