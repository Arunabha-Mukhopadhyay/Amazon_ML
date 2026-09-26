"""
Train / Validation Split Generator.

Splits Source 1 entities strictly 80/20 by entity_id:
- S1 entities are split into train_s1_ids and val_s1_ids
- train_ground_truth is partitioned into train_gt and val_gt
- Source 2 and Source 3 files are never split (they remain the full candidate search pool)
- Also provides a utility to build a fast local development sample (e.g., 15k entities)
"""

import os
import random
import csv
from typing import Set, Dict, List, Tuple


def generate_train_val_split(
    s1_path: str,
    gt_path: str,
    output_dir: str,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[int, int]:
    """Split Source 1 entities and ground truth into 80/20 train/validation sets."""
    os.makedirs(output_dir, exist_ok=True)
    rng = random.Random(seed)

    print(f"Reading S1 entities from {s1_path}...")
    s1_entities = []
    with open(s1_path, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            s1_entities.append(parts[0])

    rng.shuffle(s1_entities)
    split_idx = int(len(s1_entities) * train_ratio)
    train_ids = set(s1_entities[:split_idx])
    val_ids = set(s1_entities[split_idx:])

    train_ids_file = os.path.join(output_dir, "train_s1_ids.txt")
    val_ids_file = os.path.join(output_dir, "val_s1_ids.txt")

    with open(train_ids_file, "w", encoding="utf-8") as f:
        for eid in sorted(train_ids):
            f.write(f"{eid}\n")

    with open(val_ids_file, "w", encoding="utf-8") as f:
        for eid in sorted(val_ids):
            f.write(f"{eid}\n")

    print(f"Wrote {len(train_ids):,} train IDs to {train_ids_file}")
    print(f"Wrote {len(val_ids):,} val IDs to {val_ids_file}")

    # Now partition ground truth
    train_gt_file = os.path.join(output_dir, "train_gt.tsv")
    val_gt_file = os.path.join(output_dir, "val_gt.tsv")

    train_count = 0
    val_count = 0

    with open(gt_path, "r", encoding="utf-8") as f, \
         open(train_gt_file, "w", encoding="utf-8") as f_train, \
         open(val_gt_file, "w", encoding="utf-8") as f_val:
        header = f.readline()
        f_train.write(header)
        f_val.write(header)

        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            s1_id = parts[0]
            if s1_id in train_ids:
                f_train.write(line)
                train_count += 1
            elif s1_id in val_ids:
                f_val.write(line)
                val_count += 1

    print(f"Wrote {train_count:,} rows to {train_gt_file}")
    print(f"Wrote {val_count:,} rows to {val_gt_file}")
    return len(train_ids), len(val_ids)


def create_sample_dataset(
    dataset_dir: str,
    sample_dir: str,
    sample_size: int = 15000,
    seed: int = 42,
):
    """Create a self-contained miniature dataset for rapid local testing.

    Includes sample_s1 (15k entities), all their ground truth matches, plus
    a 5x pool of negative/unmatched S2 and S3 candidates.
    """
    os.makedirs(sample_dir, exist_ok=True)
    rng = random.Random(seed)

    # 1. Read S1 records
    s1_file = os.path.join(dataset_dir, "train/train_source1.tsv")
    gt_file = os.path.join(dataset_dir, "train/train_ground_truth.tsv")

    print(f"Sampling {sample_size:,} entities from {s1_file}...")
    s1_records = []
    with open(s1_file, "r", encoding="utf-8") as f:
        s1_header = f.readline()
        for line in f:
            if line.strip():
                s1_records.append(line)

    sampled_s1 = rng.sample(s1_records, min(sample_size, len(s1_records)))
    sampled_s1_ids = {line.split("\t")[0] for line in sampled_s1}

    sample_s1_file = os.path.join(sample_dir, "sample_source1.tsv")
    with open(sample_s1_file, "w", encoding="utf-8") as f:
        f.write(s1_header)
        for line in sampled_s1:
            f.write(line)

    # 2. Extract ground truth for sampled S1
    sample_gt_file = os.path.join(sample_dir, "sample_ground_truth.tsv")
    matched_target_ids = set()

    with open(gt_file, "r", encoding="utf-8") as f, \
         open(sample_gt_file, "w", encoding="utf-8") as out_gt:
        gt_header = f.readline()
        out_gt.write(gt_header)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if parts[0] in sampled_s1_ids:
                out_gt.write(line)
                if len(parts) > 1 and parts[1].strip():
                    for mid in parts[1].split(","):
                        mid = mid.strip()
                        if mid:
                            matched_target_ids.add(mid)

    s2_needed = {m for m in matched_target_ids if m.startswith("S2-")}
    s3_needed = {m for m in matched_target_ids if m.startswith("S3-")}
    print(f"Sample has {len(matched_target_ids):,} true target matches (S2: {len(s2_needed):,}, S3: {len(s3_needed):,})")

    # 3. Extract S2 records (all true matches + extra random negatives to make realistic search pool)
    for src_num, needed_ids in [(2, s2_needed), (3, s3_needed)]:
        src_file = os.path.join(dataset_dir, f"train/train_source{src_num}.tsv")
        out_file = os.path.join(sample_dir, f"sample_source{src_num}.tsv")
        print(f"Extracting sample for Source {src_num}...")

        with open(src_file, "r", encoding="utf-8") as f, \
             open(out_file, "w", encoding="utf-8") as out_f:
            header = f.readline()
            out_f.write(header)

            extra_negatives = []
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if parts[0] in needed_ids:
                    out_f.write(line)
                elif len(extra_negatives) < sample_size * 4:
                    if rng.random() < 0.05:  # sub-sample distractors
                        extra_negatives.append(line)

            for line in extra_negatives:
                out_f.write(line)

    print(f"Sample dataset successfully created at: {sample_dir}")


if __name__ == "__main__":
    base_dir = "/Users/arunabhamukhopadhyay/Desktop/student_resource"
    split_dir = os.path.join(base_dir, "data_splits")
    generate_train_val_split(
        s1_path=os.path.join(base_dir, "dataset/train/train_source1.tsv"),
        gt_path=os.path.join(base_dir, "dataset/train/train_ground_truth.tsv"),
        output_dir=split_dir,
    )
    create_sample_dataset(
        dataset_dir=os.path.join(base_dir, "dataset"),
        sample_dir=os.path.join(base_dir, "dataset/sample"),
        sample_size=15000,
    )
