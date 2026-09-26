"""LightGBM pair classifier and the decision step.

The classifier outputs P(query and candidate are the same business). The
decision step then gives every query to at most one Source-1 entity: its
highest-probability candidate, and only when that probability clears a
threshold. This uses the data's structure (each Source-2/3 record belongs to at
most one Source-1 entity) and keeps precision high, which F0.5 rewards.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd

PARAMS = {
    "objective": "binary",
    "learning_rate": 0.1,
    "num_leaves": 255,
    "min_data_in_leaf": 500,
    "max_bin": 127,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "seed": 42,
    "deterministic": True,
    "force_row_wise": True,
    "num_threads": 0,
    "verbose": -1,
}


PARAMS_STAGE2 = dict(PARAMS, learning_rate=0.05, num_leaves=127, min_data_in_leaf=200)


def train_model(x_tr, y_tr, x_es, y_es, max_rounds=1500, early_stop=50, params=None):
    """Train with early stopping on a held-out (early-stopping) fold."""
    dtr = lgb.Dataset(x_tr, label=y_tr, free_raw_data=True)
    des = lgb.Dataset(x_es, label=y_es, reference=dtr, free_raw_data=True)
    return lgb.train(
        params or PARAMS, dtr, num_boost_round=max_rounds, valid_sets=[des], valid_names=["es"],
        callbacks=[lgb.early_stopping(early_stop, verbose=False), lgb.log_evaluation(100)],
    )


def best_per_query(q_pos, s1_pos, p):
    """For every query keep only its highest-probability candidate.

    Returns a DataFrame (q_pos, s1_pos, p) with one row per query.
    """
    q_pos = np.asarray(q_pos)
    order = np.lexsort((-np.asarray(p), q_pos))
    qs = q_pos[order]
    first = np.ones(len(qs), dtype=bool)
    first[1:] = qs[1:] != qs[:-1]
    idx = order[first]
    return pd.DataFrame({"q_pos": q_pos[idx], "s1_pos": np.asarray(s1_pos)[idx], "p": np.asarray(p)[idx]})
