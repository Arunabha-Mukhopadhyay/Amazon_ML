"""Safe reading and writing of the challenge's tab-separated files.

Every field is read as plain text: pandas' automatic missing-value detection is
switched off (so business names such as "NA" or addresses containing "NULL" are
kept verbatim) and quote handling is disabled (names may contain quote marks).
Output files are written with "\\n" line endings and no quoting, exactly as the
submission validator expects.
"""

import csv
import os

import pandas as pd

SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]


def read_source(path):
    """Read a *_source{1,2,3}.tsv file as a DataFrame of strings."""
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        quoting=csv.QUOTE_NONE,
        engine="c",
    )
    if list(df.columns) != SOURCE_COLUMNS:
        raise ValueError(f"{path}: unexpected columns {list(df.columns)}")
    return df


def read_ground_truth(path):
    """Read train_ground_truth.tsv into {source1_id: [matched ids]}."""
    truth = {}
    with open(path, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        if header != ["source1_entity_id", "matched_entity_ids"]:
            raise ValueError(f"{path}: unexpected header {header}")
        for line in f:
            s1, _, rest = line.rstrip("\n").partition("\t")
            truth[s1] = rest.split(",") if rest else []
    return truth


def write_id_lists(path, header, s1_ids, mapping):
    """Write one row per Source-1 id: "<s1>\\t<id>,<id>,...".

    ``s1_ids`` fixes the row set and order (every Source-1 entity appears exactly
    once, empty when it has no ids); ``mapping`` is {s1_id: iterable of ids}.
    Ids are de-duplicated while keeping their order.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(header) + "\n")
        for s1 in s1_ids:
            ids = list(dict.fromkeys(mapping.get(s1, ())))
            f.write(s1 + "\t" + ",".join(ids) + "\n")


def check_clean_ids(path):
    """Fail loudly if any id in a written file carries stray whitespace/\\r."""
    bad = 0
    with open(path, encoding="utf-8", newline="") as f:
        next(f)
        for line in f:
            if "\r" in line or " " in line:
                bad += 1
    if bad:
        raise ValueError(f"{path}: {bad} rows contain '\\r' or spaces")
