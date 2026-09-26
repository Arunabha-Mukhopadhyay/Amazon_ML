"""
Evaluation Metric: Macro-Averaged F_0.5 Score (beta = 0.5)

Formula:
    F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)

Computed per Source 1 entity, then macro-averaged across all Source 1 entities.
Singletons:
    - True empty, predicted empty -> 1.0
    - True empty, predicted non-empty -> 0.0
    - True non-empty, predicted empty -> 0.0
"""

from typing import Dict, List, Set, Union


def compute_entity_f05(true_ids: Set[str], pred_ids: Set[str]) -> float:
    """Compute F_0.5 score for a single Source 1 entity."""
    # Singleton cases
    if len(true_ids) == 0:
        return 1.0 if len(pred_ids) == 0 else 0.0
    if len(pred_ids) == 0:
        return 0.0

    tp = len(true_ids & pred_ids)
    if tp == 0:
        return 0.0

    precision = tp / len(pred_ids)
    recall = tp / len(true_ids)

    denom = 0.25 * precision + recall
    if denom == 0:
        return 0.0

    return (1.25 * precision * recall) / denom


def evaluate_predictions(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]],
) -> Dict[str, float]:
    """Compute macro-averaged F_0.5, macro precision, and macro recall.

    Args:
        ground_truth: dict mapping s1_id -> set of true matching s2/s3 ids
        predictions: dict mapping s1_id -> set of predicted matching s2/s3 ids

    Returns:
        dict with macro_f05, macro_precision, macro_recall, num_entities, singleton_accuracy
    """
    total_f05 = 0.0
    total_p = 0.0
    total_r = 0.0
    singleton_correct = 0
    total_singletons = 0
    non_singletons = 0

    entities = set(ground_truth.keys())

    for s1_id in entities:
        true_set = ground_truth.get(s1_id, set())
        pred_set = predictions.get(s1_id, set())

        score = compute_entity_f05(true_set, pred_set)
        total_f05 += score

        if len(true_set) == 0:
            total_singletons += 1
            if len(pred_set) == 0:
                singleton_correct += 1
                total_p += 1.0
                total_r += 1.0
            else:
                total_p += 0.0
                total_r += 0.0
        else:
            non_singletons += 1
            if len(pred_set) == 0:
                total_p += 0.0
                total_r += 0.0
            else:
                tp = len(true_set & pred_set)
                p = tp / len(pred_set)
                r = tp / len(true_set)
                total_p += p
                total_r += r

    n = len(entities)
    macro_f05 = total_f05 / n if n > 0 else 0.0
    macro_p = total_p / n if n > 0 else 0.0
    macro_r = total_r / n if n > 0 else 0.0
    singleton_acc = (
        singleton_correct / total_singletons if total_singletons > 0 else 1.0
    )

    return {
        "macro_f05": macro_f05,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "num_entities": n,
        "total_singletons": total_singletons,
        "singleton_accuracy": singleton_acc,
    }
