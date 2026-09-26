"""Inspect exam (test) records of one country: matched groups, unmatched records, frequent words."""
import argparse
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--work", default="work/full", help="work folder of the run")
ap.add_argument("--data-dir", default="~/data/dataset", help="folder with train/ and test/")
ap.add_argument("--train-pct", type=int, default=25, help="--train-pct used by that run")
ARGS = ap.parse_args()
import os
DATA = os.path.expanduser(ARGS.data_dir)
import collections, random, re
import pandas as pd
random.seed(0)
W = os.path.join(ARGS.work, "prep_test")
s1 = pd.read_parquet(f"{W}/s1.parquet"); q = pd.concat([pd.read_parquet(f"{W}/s{s}.parquet") for s in (2, 3)], ignore_index=True)
raw = dict(zip(s1.entity_id, (s1.business_name + " | " + s1.business_address).str.slice(0, 100)))
raw.update(zip(q.entity_id, (q.business_name + " | " + q.business_address).str.slice(0, 100)))
fr1 = set(s1.entity_id[s1.country == "France"])
matched = {}
with open(os.path.join("output", "matching_results.tsv")) as f:
    next(f)
    for line in f:
        s, _, rest = line.rstrip("\n").partition("\t")
        if s in fr1:
            matched[s] = rest.split(",") if rest else []
groups = random.sample([s for s in matched if matched[s]], 6)
print("=== FRENCH GROUPS WE MATCHED ===")
for s in groups:
    print("S1", raw[s])
    for x in matched[s][:5]:
        print("   ", x[:2], raw[x])
got = {x for v in matched.values() for x in v}
fq = q[q.country == "France"]
un = fq[~fq.entity_id.isin(got)]
print(f"\nFrench queries {len(fq)}, unmatched {len(un)} ({len(un)/len(fq)*100:.1f}%)")
for s in (2, 3):
    part = fq[fq.src == s]
    print(f"  S{s}: unmatched {(~part.entity_id.isin(got)).mean()*100:.1f}%")
print("=== UNMATCHED FRENCH QUERIES (sample) ===")
for r in un.sample(14, random_state=2).itertuples():
    print("  ", r.entity_id[:2], raw[r.entity_id], "  ->", r.n_full, "|", r.a_full[:60])
# frequent words in French addresses (raw, lowercase)
cnt = collections.Counter()
for a in fq.business_address.sample(200000, random_state=0):
    cnt.update(re.findall(r"[a-zà-ÿ\.]+", a.lower()))
print("\nfrequent French address words:", cnt.most_common(70))
cnt2 = collections.Counter()
for n in fq.business_name.sample(200000, random_state=0):
    cnt2.update(re.findall(r"[a-zà-ÿ\.]+", n.lower()))
print("\nfrequent French name words:", cnt2.most_common(50))
