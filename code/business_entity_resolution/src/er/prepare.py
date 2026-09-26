"""Normalise every record of a split once and cache the result as parquet.

For each source file this produces one row per record with the cleaned name
and address fields used by blocking and feature extraction.

Usage:
    scripts/py -m er.prepare --in-dir work/dev --prefix train --out-dir work/dev/prep
    scripts/py -m er.prepare --in-dir ../../dataset/test --prefix test --out-dir work/test/prep
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from .io import read_source
from .normalize import normalize_address, normalize_name

CHUNK = 20000


def _nonlatin(text):
    """True if the text has letters outside the Latin alphabets."""
    return any(ord(c) > 0x24F and c.isalpha() for c in text)


def _normalize_chunk(rows):
    """Normalise a list of (name, address, country) tuples."""
    out = []
    for name, addr, country in rows:
        n = normalize_name(name)
        a = normalize_address(addr, country)
        out.append((
            n["full"], n["nospace"], n["skel"], n["legal"],
            "|".join(dict.fromkeys(n["parts"])), n["web"], int(_nonlatin(name)),
            a["full"], " ".join(a["nums"]), a["house"], int(not a["tokens"]),
        ))
    return out


COLUMNS = [
    "n_full", "n_nospace", "n_skel", "n_legal", "n_parts", "n_web", "n_nonlatin",
    "a_full", "a_nums", "a_house", "a_empty",
]


def normalize_frame(df, workers=None):
    """Return ``df`` with the normalised columns appended."""
    rows = list(zip(df["business_name"], df["business_address"], df["country"]))
    chunks = [rows[i:i + CHUNK] for i in range(0, len(rows), CHUNK)]
    workers = workers or os.cpu_count()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        results = [r for part in ex.map(_normalize_chunk, chunks) for r in part]
    norm = pd.DataFrame(results, columns=COLUMNS, index=df.index)
    for c in ("n_web", "n_nonlatin", "a_empty"):
        norm[c] = norm[c].astype("int8")
    return pd.concat([df, norm], axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", required=True, help="folder with <prefix>_source{1,2,3}.tsv")
    ap.add_argument("--prefix", required=True, choices=["train", "test"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    for s in (1, 2, 3):
        t0 = time.time()
        df = read_source(os.path.join(args.in_dir, f"{args.prefix}_source{s}.tsv"))
        df = normalize_frame(df, args.workers)
        df["src"] = s
        df.to_parquet(os.path.join(args.out_dir, f"s{s}.parquet"), index=False)
        print(f"source {s}: {len(df)} rows normalised in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
