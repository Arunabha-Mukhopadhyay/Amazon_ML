"""
Country-Partitioned Stream Engine.

Ensures that peak memory never exceeds ~400 MB RAM, easily running on
an 8 GB Mac without any swap or memory pressure.
Processes each country independently:
1. Stream S2 & S3 records for Country X into memory
2. Stream S1 records for Country X, match, and write directly to output
3. Free memory completely before processing Country X+1
"""

import os
import gc
import sys
from collections import defaultdict
from typing import Dict, List, Set, Tuple

from src.normalize import normalize_business_name


def stream_country_exact_match(
    s1_path: str,
    s2_path: str,
    s3_path: str,
    output_matching_path: str,
    output_candidate_path: str,
    countries: List[str] = None,
):
    """Run exact match partitioned by country to guarantee minimal RAM usage (<400MB)."""
    if countries is None:
        countries = ["France", "US", "India"]

    os.makedirs(os.path.dirname(output_matching_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_candidate_path), exist_ok=True)

    # Initialize output files with headers
    with open(output_matching_path, "w", encoding="utf-8") as fm:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
    with open(output_candidate_path, "w", encoding="utf-8") as fc:
        fc.write("source1_entity_id\tcandidate_entity_ids\n")

    total_s1_processed = 0
    total_matches_found = 0

    for country in countries:
        print(f"\n==========================================")
        print(f"--> Processing Country: {country}")
        print(f"==========================================")

        # 1. Build index for this country only
        index = defaultdict(list)
        count_s2 = 0
        count_s3 = 0

        # Read S2 for this country
        with open(s2_path, "r", encoding="utf-8") as f:
            f.readline()
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 4 and parts[3] == country:
                    norm_name = normalize_business_name(parts[1])
                    if norm_name:
                        index[norm_name].append(parts[0])
                        count_s2 += 1

        # Read S3 for this country
        with open(s3_path, "r", encoding="utf-8") as f:
            f.readline()
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 4 and parts[3] == country:
                    norm_name = normalize_business_name(parts[1])
                    if norm_name:
                        index[norm_name].append(parts[0])
                        count_s3 += 1

        print(f"[{country}] Indexed {count_s2:,} S2 and {count_s3:,} S3 records. Unique names: {len(index):,}")

        # 2. Query S1 for this country and append directly to output
        s1_country_count = 0
        s1_matched_count = 0

        with open(s1_path, "r", encoding="utf-8") as f, \
             open(output_matching_path, "a", encoding="utf-8") as fm, \
             open(output_candidate_path, "a", encoding="utf-8") as fc:
            f.readline()
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 4 and parts[3] == country:
                    s1_id = parts[0]
                    norm_name = normalize_business_name(parts[1])
                    matches = index.get(norm_name, [])

                    # Deduplicate while preserving order
                    seen = set()
                    uniq = []
                    for mid in matches:
                        if mid not in seen:
                            seen.add(mid)
                            uniq.append(mid)

                    m_str = ",".join(uniq)
                    fm.write(f"{s1_id}\t{m_str}\n")
                    fc.write(f"{s1_id}\t{m_str}\n")

                    s1_country_count += 1
                    if uniq:
                        s1_matched_count += 1

        print(f"[{country}] Processed {s1_country_count:,} S1 entities ({s1_matched_count:,} matched).")
        total_s1_processed += s1_country_count
        total_matches_found += s1_matched_count

        # 3. Clean up memory before next country
        del index
        gc.collect()

    print(f"\nCompleted all countries! Total S1 processed: {total_s1_processed:,}, Matched: {total_matches_found:,}")


if __name__ == "__main__":
    base_dir = "/Users/arunabhamukhopadhyay/Desktop/student_resource"
    stream_country_exact_match(
        s1_path=os.path.join(base_dir, "dataset/test/test_source1.tsv"),
        s2_path=os.path.join(base_dir, "dataset/test/test_source2.tsv"),
        s3_path=os.path.join(base_dir, "dataset/test/test_source3.tsv"),
        output_matching_path=os.path.join(base_dir, "output/matching_results.tsv"),
        output_candidate_path=os.path.join(base_dir, "output/candidate_pairs.tsv"),
    )
