"""Best decision threshold when decoy mistakes are weighted up (test has ~1.9x more decoys per entity than training)."""
import argparse
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--work", default="work/full", help="work folder of the run")
ap.add_argument("--data-dir", default="~/data/dataset", help="folder with train/ and test/")
ap.add_argument("--train-pct", type=int, default=25, help="--train-pct used by that run")
ARGS = ap.parse_args()
import os
DATA = os.path.expanduser(ARGS.data_dir)
import numpy as np, pandas as pd
from er.io import read_ground_truth
from er.run import s1_folds, VAL
from er.evaluate import f05_per_entity
W = ARGS.work
vb = pd.read_parquet(f"{W}/val_best.parquet")
truth = read_ground_truth(os.path.join(DATA, "train", "train_ground_truth.tsv"))
s1 = pd.read_parquet(f"{W}/prep_train/s1.parquet", columns=["entity_id", "country"])
fold = s1_folds(s1.entity_id.to_numpy(), ARGS.train_pct)
val = s1[fold == VAL]
q_owner = {x: s for s, ids in truth.items() for x in ids}
vb["decoy"] = vb.q_id.map(q_owner).isna()
n_true = np.array([len(truth.get(s, ())) for s in val.entity_id])
print(f"val best rows {len(vb)}, decoy share among them {vb.decoy.mean()*100:.1f}%")
def score(t, w, mask=None):
    sel = vb[vb.p_final >= t]
    g = sel.assign(wt=np.where(sel.decoy, w, 1.0), tp=sel.correct.astype(float)).groupby("s1_id")[["wt", "tp"]].sum()
    npred = g.wt.reindex(val.entity_id, fill_value=0).to_numpy()
    ntp = g.tp.reindex(val.entity_id, fill_value=0).to_numpy()
    f = f05_per_entity(npred, n_true, ntp)
    return f.mean() if mask is None else f[mask].mean()
grid = np.round(np.arange(0.30, 0.99, 0.02), 2)
for w in (1.0, 1.5, 1.9, 2.5):
    sc = [score(t, w) for t in grid]
    i = int(np.argmax(sc))
    print(f"decoy weight {w}: best threshold {grid[i]}  weighted F0.5 {sc[i]:.5f}  | plain F0.5 at it {score(grid[i], 1.0):.5f}")
for c in ("US", "India"):
    m = (val.country == c).to_numpy()
    print(c, {t: round(score(t, 1.0, m), 5) for t in (0.5, 0.6, 0.7, 0.8, 0.9)})
