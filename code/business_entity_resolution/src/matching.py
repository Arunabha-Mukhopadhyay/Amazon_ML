"""
Stage 2: Pairwise Matching Classifier & F_0.5 Threshold Tuner.

Implements LightGBM (Gradient Boosted Decision Trees) as the primary matching engine,
with an automatic fallback to CalibratedMatcher when running in zero-dependency environments.

Features:
- Handles non-linear feature interactions (e.g. address conflicts overriding name similarity)
- Threshold sweep directly maximizing Macro F_0.5 on held-out validation entities
- High-throughput batch inference
"""

import os
import math
import random
from typing import Dict, List, Set, Tuple

from src.features import compute_pair_features, FEATURE_NAMES
from src.metric import evaluate_predictions, compute_entity_f05

# Check LightGBM availability
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


class LightGBMMatcher:
    """Gradient Boosted Decision Tree Matcher using LightGBM."""

    def __init__(self, n_estimators: int = 120, learning_rate: float = 0.05, max_depth: int = 6):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.model = None
        self.threshold = 0.40

    def fit(self, X: List[List[float]], y: List[int]):
        """Train LightGBM binary classifier on candidate pair features."""
        print(f"Training LightGBM on {len(X):,} candidate pairs with {len(FEATURE_NAMES)} features...")
        dtrain = lgb.Dataset(X, label=y, feature_name=FEATURE_NAMES)
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "num_leaves": 31,
            "feature_fraction": 0.85,
            "verbose": -1,
            "n_jobs": -1,
            "seed": 42,
        }
        self.model = lgb.train(
            params,
            dtrain,
            num_boost_round=self.n_estimators,
        )
        print("LightGBM training complete.")

    def predict_score(self, features: List[float]) -> float:
        """Predict match probability for a single pair."""
        if self.model is None:
            return 0.0
        return float(self.model.predict([features])[0])

    def predict_batch(self, X: List[List[float]]) -> List[float]:
        """Predict match probabilities for a batch of pairs."""
        if self.model is None or not X:
            return [0.0] * len(X)
        return list(self.model.predict(X))

    def tune_threshold(
        self,
        validation_pairs: Dict[str, List[Tuple[str, List[float]]]],
        val_ground_truth: Dict[str, Set[str]],
        thresholds: List[float] = None,
    ) -> float:
        """Sweep decision threshold to directly maximize Macro F_0.5."""
        if thresholds is None:
            thresholds = [i / 100.0 for i in range(25, 90, 3)]

        # Batch predict all candidate pairs
        all_feats = []
        pair_mapping = []
        for s1_id, cand_list in validation_pairs.items():
            for cid, feats in cand_list:
                all_feats.append(feats)
                pair_mapping.append((s1_id, cid))

        if all_feats:
            scores = self.predict_batch(all_feats)
        else:
            scores = []

        # Group scored candidates by s1_id
        scored_by_s1 = {s1_id: [] for s1_id in validation_pairs}
        for (s1_id, cid), score in zip(pair_mapping, scores):
            scored_by_s1[s1_id].append((cid, score))

        best_f05 = -1.0
        best_thresh = 0.40
        best_metrics = {}

        print(f"\nSweeping {len(thresholds)} threshold values for Macro F_0.5 optimization...")
        for th in thresholds:
            preds = {}
            for s1_id, cands in scored_by_s1.items():
                m_set = {cid for cid, score in cands if score >= th}
                preds[s1_id] = m_set

            metrics = evaluate_predictions(val_ground_truth, preds)
            f05 = metrics["macro_f05"]
            if f05 > best_f05:
                best_f05 = f05
                best_thresh = th
                best_metrics = metrics

        print(f"\n--> Optimal LightGBM Threshold: {best_thresh:.2f}")
        print(f"--> Best Macro F_0.5:           {best_metrics['macro_f05']:.4f}")
        print(f"--> Macro Precision:            {best_metrics['macro_precision']:.4f}")
        print(f"--> Macro Recall:               {best_metrics['macro_recall']:.4f}")
        print(f"--> Singleton Accuracy:         {best_metrics['singleton_accuracy']:.4f}")

        self.threshold = best_thresh
        return best_thresh


class CalibratedMatcher:
    """Calibrated pairwise similarity matcher (pure Python, stdlib-only)."""

    def __init__(self, weights: List[float] = None, bias: float = -2.2):
        if weights is None:
            self.weights = [
                3.0,  # name_exact
                2.5,  # name_compact_match
                2.2,  # name_jaccard
                1.5,  # name_overlap
                2.0,  # name_trigram_jaccard
                2.5,  # name_edit_sim
                0.8,  # name_len_ratio
                2.2,  # addr_exact
                1.8,  # addr_jaccard
                1.0,  # addr_overlap
                1.8,  # addr_anchor_match
                -3.5, # addr_anchor_conflict
                -0.8, # addr_is_empty
                2.5,  # composite_sim
            ]
        else:
            self.weights = list(weights)
        self.bias = bias
        self.threshold = 0.40

    def predict_score(self, features: List[float]) -> float:
        """Compute match probability via sigmoid."""
        z = self.bias + sum(w * f for w, f in zip(self.weights, features))
        if z < -20.0:
            return 0.0
        if z > 20.0:
            return 1.0
        return 1.0 / (1.0 + math.exp(-z))

    def train_logistic_sgd(
        self,
        X: List[List[float]],
        y: List[int],
        epochs: int = 4,
        lr: float = 0.06,
        reg: float = 0.001,
    ):
        """Train weights via regularized SGD."""
        n_samples = len(X)
        n_features = len(self.weights)
        indices = list(range(n_samples))

        print(f"Training CalibratedMatcher with SGD on {n_samples:,} candidate pairs...")
        for epoch in range(epochs):
            random.shuffle(indices)
            loss = 0.0
            for idx in indices:
                feat = X[idx]
                target = y[idx]
                pred = self.predict_score(feat)
                err = pred - target
                loss += - (target * math.log(max(pred, 1e-9)) + (1 - target) * math.log(max(1 - pred, 1e-9)))

                for j in range(n_features):
                    self.weights[j] -= lr * (err * feat[j] + reg * self.weights[j])
                self.bias -= lr * err

            avg_loss = loss / n_samples
            if (epoch + 1) % 1 == 0 or epoch == epochs - 1:
                print(f"  Epoch {epoch + 1}/{epochs} - LogLoss: {avg_loss:.4f}")

    def tune_threshold(
        self,
        validation_pairs: Dict[str, List[Tuple[str, List[float]]]],
        val_ground_truth: Dict[str, Set[str]],
        thresholds: List[float] = None,
    ) -> float:
        """Sweep probability threshold to directly maximize Macro F_0.5."""
        if thresholds is None:
            thresholds = [i / 100.0 for i in range(25, 90, 3)]

        best_f05 = -1.0
        best_thresh = 0.40
        best_metrics = {}

        print(f"\nSweeping {len(thresholds)} threshold values for Macro F_0.5 optimization...")
        for th in thresholds:
            preds = {}
            for s1_id, cand_list in validation_pairs.items():
                m_set = set()
                for cid, feats in cand_list:
                    score = self.predict_score(feats)
                    if score >= th:
                        m_set.add(cid)
                preds[s1_id] = m_set

            metrics = evaluate_predictions(val_ground_truth, preds)
            f05 = metrics["macro_f05"]
            if f05 > best_f05:
                best_f05 = f05
                best_thresh = th
                best_metrics = metrics

        print(f"\n--> Optimal Threshold Found: {best_thresh:.2f}")
        print(f"--> Best Macro F_0.5:        {best_metrics['macro_f05']:.4f}")
        print(f"--> Macro Precision:         {best_metrics['macro_precision']:.4f}")
        print(f"--> Macro Recall:            {best_metrics['macro_recall']:.4f}")
        print(f"--> Singleton Accuracy:      {best_metrics['singleton_accuracy']:.4f}")

        self.threshold = best_thresh
        return best_thresh


def get_matcher(use_lightgbm: bool = True):
    """Factory function returning LightGBMMatcher if available, else CalibratedMatcher."""
    if use_lightgbm and HAS_LIGHTGBM:
        print("[Matcher] Initializing LightGBMMatcher (Gradient Boosted Trees).")
        return LightGBMMatcher()
    else:
        if use_lightgbm and not HAS_LIGHTGBM:
            print("[Matcher] LightGBM not installed. Using built-in CalibratedMatcher.")
        else:
            print("[Matcher] Using built-in CalibratedMatcher.")
        return CalibratedMatcher()
