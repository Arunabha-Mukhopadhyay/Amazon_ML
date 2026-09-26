"""The challenge metric: macro-averaged F-beta (beta = 0.5) per Source-1 entity.

For each Source-1 entity:
  * no true matches and nothing predicted       -> 1.0
  * no true matches but something predicted     -> 0.0
  * true matches but nothing predicted          -> 0.0
  * otherwise F0.5 = 1.25 P R / (0.25 P + R)     (0 when no predicted id is correct)
The score is the mean over all Source-1 entities in the evaluation set.
"""

import numpy as np


def f05_per_entity(n_pred, n_true, n_tp):
    """Vectorised per-entity F0.5 from counts."""
    n_pred = np.asarray(n_pred, dtype=np.float64)
    n_true = np.asarray(n_true, dtype=np.float64)
    n_tp = np.asarray(n_tp, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(n_pred > 0, n_tp / n_pred, 0.0)
        r = np.where(n_true > 0, n_tp / n_true, 0.0)
        f = np.where(n_tp > 0, 1.25 * p * r / (0.25 * p + r), 0.0)
    return np.where((n_true == 0) & (n_pred == 0), 1.0, f)


def macro_f05(pred, truth, s1_ids):
    """Score predictions {s1: iterable of ids} against truth {s1: list of ids}."""
    n_pred, n_true, n_tp = [], [], []
    for s in s1_ids:
        p = set(pred.get(s, ()))
        t = set(truth.get(s, ()))
        n_pred.append(len(p))
        n_true.append(len(t))
        n_tp.append(len(p & t))
    return float(f05_per_entity(n_pred, n_true, n_tp).mean())


def score_positions(s1_pos, correct, n_true, eval_mask):
    """Macro F0.5 from predicted pairs given as positions.

    s1_pos: Source-1 position of each predicted pair; correct: whether the pair
    is a true match; n_true: true-match count per Source-1 position;
    eval_mask: which Source-1 positions are scored.
    """
    n_s1 = len(n_true)
    n_pred = np.bincount(s1_pos, minlength=n_s1)
    n_tp = np.bincount(s1_pos[correct], minlength=n_s1)
    return float(f05_per_entity(n_pred[eval_mask], n_true[eval_mask], n_tp[eval_mask]).mean())


def sweep_thresholds(best, correct, n_true, eval_mask, grid):
    """Score every threshold in ``grid`` for a best-candidate-per-query table."""
    s1_pos = best["s1_pos"].to_numpy()
    p = best["p"].to_numpy()
    return [score_positions(s1_pos[p >= t], correct[p >= t], n_true, eval_mask) for t in grid]
