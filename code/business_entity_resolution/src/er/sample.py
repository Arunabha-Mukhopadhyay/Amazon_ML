"""Create a small, deterministic development sample of the training data.

A fixed fraction of Source-1 entities is selected by hashing their id. The
sample keeps every Source-2/3 record matched to a selected entity, plus the same
fraction of the true decoys (records that match no entity at all). Records whose
entity was not selected are dropped: in the real data every matched record's
entity is present, so keeping them would inflate the decoy ratio.

Usage:
    scripts/py -m er.sample --data-dir ../../dataset --out-dir work/dev --percent 5
"""

import argparse
import os
import time
import zlib


def pick(entity_id, percent):
    """Deterministically select ``percent`` % of ids."""
    return zlib.crc32(entity_id.encode()) % 100 < percent


def copy_rows(src, dst, keep):
    """Copy the header plus every line whose first field satisfies ``keep``."""
    n = 0
    with open(src, encoding="utf-8") as fin, open(dst, "w", encoding="utf-8", newline="\n") as fout:
        fout.write(fin.readline())
        for line in fin:
            if keep(line.partition("\t")[0]):
                fout.write(line)
                n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="../../dataset")
    ap.add_argument("--out-dir", default="work/dev")
    ap.add_argument("--percent", type=int, default=5)
    args = ap.parse_args()
    t0 = time.time()
    train = os.path.join(args.data_dir, "train")
    os.makedirs(args.out_dir, exist_ok=True)

    selected = set()
    selected_matches = set()
    other_matched_picked = set()  # matched to an unselected entity, and would pass pick()
    gt_out = os.path.join(args.out_dir, "train_ground_truth.tsv")
    with open(os.path.join(train, "train_ground_truth.tsv"), encoding="utf-8") as fin, \
            open(gt_out, "w", encoding="utf-8", newline="\n") as fout:
        fout.write(fin.readline())
        for line in fin:
            s1, _, rest = line.rstrip("\n").partition("\t")
            ids = rest.split(",") if rest else []
            if pick(s1, args.percent):
                selected.add(s1)
                fout.write(line)
                selected_matches.update(ids)
            else:
                other_matched_picked.update(x for x in ids if pick(x, args.percent))

    def keep_other(x):
        if x in selected_matches:
            return True
        return pick(x, args.percent) and x not in other_matched_picked

    n1 = copy_rows(os.path.join(train, "train_source1.tsv"),
                   os.path.join(args.out_dir, "train_source1.tsv"), lambda x: x in selected)
    n2 = copy_rows(os.path.join(train, "train_source2.tsv"),
                   os.path.join(args.out_dir, "train_source2.tsv"), keep_other)
    n3 = copy_rows(os.path.join(train, "train_source3.tsv"),
                   os.path.join(args.out_dir, "train_source3.tsv"), keep_other)
    print(f"dev sample: S1={n1} S2={n2} S3={n3} ({time.time() - t0:.0f}s) -> {args.out_dir}")


if __name__ == "__main__":
    main()
