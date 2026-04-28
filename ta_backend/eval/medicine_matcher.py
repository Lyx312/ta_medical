import re
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from eval.config import DEFAULT_THRESHOLDS, EvalThresholds


DOSE_PATTERN = re.compile(
    r"\b(\d+(?:[.,]\d+)?\s*(?:mg|ml|miligram|mililiter|iu|iu\.?|tablet|kapsul|kaplet|bij|botol|sachet|gram|g\.?|cc))\b",
    re.IGNORECASE,
)


def normalize_medicine_name(name: str | None) -> frozenset[str]:
    if not name:
        return frozenset()
    normalized = name.lower().strip()
    normalized = DOSE_PATTERN.sub("", normalized)
    normalized = re.sub(r"[/+\\-_,&]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    tokens = normalized.split()
    return frozenset(tokens)


def name_jaccard(a: str | None, b: str | None) -> float:
    set_a = normalize_medicine_name(a)
    set_b = normalize_medicine_name(b)
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def quantity_proximity(pred_qty: float | int | None, gt_qty: float | int | None) -> float:
    if pred_qty is None and gt_qty is None:
        return 1.0
    if pred_qty is None or gt_qty is None:
        return 0.0
    if gt_qty == 0:
        return 1.0 if pred_qty == 0 else 0.0
    diff = abs(float(pred_qty) - float(gt_qty))
    return max(0.0, 1.0 - diff / max(float(gt_qty), 1.0))


def combined_similarity(
    pred: dict[str, Any],
    gt: dict[str, Any],
    thresholds: EvalThresholds = DEFAULT_THRESHOLDS,
) -> float:
    name_sim = name_jaccard(pred.get("name"), gt.get("name"))
    qty_prox = quantity_proximity(pred.get("quantity"), gt.get("quantity"))
    return thresholds.MEDICINE_NAME_WEIGHT * name_sim + thresholds.MEDICINE_QTY_PROXIMITY_WEIGHT * qty_prox


MedicineMatch = tuple[int, int, float]


def match_medicines(
    predictions: list[dict[str, Any]],
    ground_truths: list[dict[str, Any]],
    thresholds: EvalThresholds = DEFAULT_THRESHOLDS,
) -> tuple[list[MedicineMatch], list[int], list[int]]:
    if not predictions and not ground_truths:
        return [], [], []
    if not predictions:
        return [], [], list(range(len(ground_truths)))
    if not ground_truths:
        return [], list(range(len(predictions))), []

    n_pred = len(predictions)
    n_gt = len(ground_truths)
    cost_matrix = np.zeros((n_pred, n_gt), dtype=float)

    for i, pred in enumerate(predictions):
        for j, gt in enumerate(ground_truths):
            sim = combined_similarity(pred, gt, thresholds)
            cost_matrix[i, j] = 1.0 - sim

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched: list[MedicineMatch] = []
    unmatched_pred: list[int] = []
    unmatched_gt: list[int] = []

    for i, j in zip(row_ind, col_ind):
        sim = combined_similarity(predictions[i], ground_truths[j], thresholds)
        if sim >= thresholds.MEDICINE_NAME_SIM_THRESHOLD:
            matched.append((i, j, sim))
        else:
            unmatched_pred.append(i)
            unmatched_gt.append(j)

    matched_pred = {match[0] for match in matched}
    matched_gt = {match[1] for match in matched}
    assigned_pred = set(row_ind.tolist())
    assigned_gt = set(col_ind.tolist())

    for i in range(n_pred):
        if i not in matched_pred and i not in assigned_pred:
            unmatched_pred.append(i)

    for j in range(n_gt):
        if j not in matched_gt and j not in assigned_gt:
            unmatched_gt.append(j)

    return matched, unmatched_pred, unmatched_gt
