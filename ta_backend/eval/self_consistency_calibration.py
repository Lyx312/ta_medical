from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.services.ai_ensemble_helpers import (
    DEFAULT_SELF_CONSISTENCY_CALIBRATION_ARTIFACT_PATH,
    classify_other_charge_kind,
    coerce_numeric_value,
    extract_other_charge_descriptor_signature,
    normalize_candidate_key,
    slugify_key,
)
from app.services.ai_ensemble_config import (
    CALIBRATION_VERSION,
    SELF_CONSISTENCY_CALIBRATION_FEATURES,
)
from eval.evaluator import CASE_INSENSITIVE_FIELDS, MedicalClaimEvaluator, _exact_match, _case_insensitive_match, _list_exact_match, _num_match
from eval.ground_truth import GroundTruthError, load_ground_truth
from eval.result_store import load_latest_snapshot, read_json, resolve_run_record_path


@dataclass(frozen=True)
class CalibrationRow:
    claim_id: str
    field_path: str
    features: dict[str, float]
    label: int


def sigmoid(value: float) -> float:
    clipped = max(min(value, 35.0), -35.0)
    return 1.0 / (1.0 + math.exp(-clipped))


def predict_probability(
    feature_names: tuple[str, ...],
    coefficients: list[float],
    intercept: float,
    features: dict[str, float],
) -> float:
    logit = intercept
    for index, feature_name in enumerate(feature_names):
        logit += coefficients[index] * float(features.get(feature_name, 0.0))
    return sigmoid(logit)


def fit_logistic_calibrator(
    rows: list[CalibrationRow],
    *,
    iterations: int = 4000,
    learning_rate: float = 0.15,
    l2_penalty: float = 0.01,
) -> tuple[list[float], float]:
    coefficients = [0.0 for _ in SELF_CONSISTENCY_CALIBRATION_FEATURES]
    intercept = 0.0
    if not rows:
        return coefficients, intercept

    for _ in range(iterations):
        coefficient_grads = [0.0 for _ in coefficients]
        intercept_grad = 0.0
        row_count = float(len(rows))
        for row in rows:
            probability = predict_probability(
                SELF_CONSISTENCY_CALIBRATION_FEATURES,
                coefficients,
                intercept,
                row.features,
            )
            error = probability - float(row.label)
            intercept_grad += error
            for index, feature_name in enumerate(SELF_CONSISTENCY_CALIBRATION_FEATURES):
                coefficient_grads[index] += error * float(row.features.get(feature_name, 0.0))

        intercept -= learning_rate * (intercept_grad / row_count)
        for index in range(len(coefficients)):
            regularized_grad = (coefficient_grads[index] / row_count) + (l2_penalty * coefficients[index])
            coefficients[index] -= learning_rate * regularized_grad

    return coefficients, intercept


def compute_accuracy(probabilities: list[float], labels: list[int]) -> float:
    if not labels:
        return 0.0
    correct = sum(1 for probability, label in zip(probabilities, labels) if (1 if probability >= 0.5 else 0) == label)
    return correct / len(labels)


def compute_brier_score(probabilities: list[float], labels: list[int]) -> float:
    if not labels:
        return 0.0
    return sum((probability - label) ** 2 for probability, label in zip(probabilities, labels)) / len(labels)


def compute_ece(probabilities: list[float], labels: list[int], *, bin_count: int = 10) -> float:
    if not labels:
        return 0.0
    total = len(labels)
    ece = 0.0
    for bin_index in range(bin_count):
        bin_start = bin_index / bin_count
        bin_end = (bin_index + 1) / bin_count
        bin_items = [
            (probability, label)
            for probability, label in zip(probabilities, labels)
            if (bin_start <= probability < bin_end) or (bin_index == bin_count - 1 and probability == 1.0)
        ]
        if not bin_items:
            continue
        avg_confidence = sum(probability for probability, _ in bin_items) / len(bin_items)
        avg_accuracy = sum(label for _, label in bin_items) / len(bin_items)
        ece += (len(bin_items) / total) * abs(avg_accuracy - avg_confidence)
    return ece


def round_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    rounded: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, float):
            rounded[key] = round(value, 6)
        else:
            rounded[key] = value
    return rounded


def _load_prediction_artifacts(results_dir: Path, claim_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    latest_snapshot = load_latest_snapshot(results_dir, claim_id)
    if latest_snapshot is None:
        raise GroundTruthError(f"No latest snapshot found for claim '{claim_id}' in {results_dir}")
    run_record_path = resolve_run_record_path(results_dir, claim_id, latest_snapshot)
    if run_record_path is None:
        raise GroundTruthError(f"No run record path found for claim '{claim_id}'")
    run_record = read_json(run_record_path)
    if run_record is None:
        raise GroundTruthError(f"Run record could not be read for claim '{claim_id}': {run_record_path}")
    scan_log = run_record.get("scan_log")
    if not isinstance(scan_log, dict):
        raise GroundTruthError(f"Run record has no scan_log export for claim '{claim_id}'")
    extraction_meta = scan_log.get("extraction_meta_payload")
    if not isinstance(extraction_meta, dict):
        raise GroundTruthError(f"Run record has no extraction_meta_payload for claim '{claim_id}'")
    field_confidence = extraction_meta.get("field_confidence")
    predicted = scan_log.get("extraction_payload")
    if not isinstance(field_confidence, dict) or not isinstance(predicted, dict):
        raise GroundTruthError(f"Run record is missing field confidence or extraction payload for claim '{claim_id}'")
    return predicted, field_confidence


def _build_path_prefix(list_name: str, item: dict[str, Any]) -> str:
    if list_name == "diagnoses":
        item_key = normalize_candidate_key("diagnoses.description", item.get("description"))
        return f"diagnoses[{slugify_key(item_key)}]"
    if list_name == "medicines":
        item_key = normalize_candidate_key("medicines.name", item.get("name"))
        return f"medicines[{slugify_key(item_key)}]"
    amount = coerce_numeric_value(item.get("amount"))
    amount_key = "__null__" if amount is None else str(amount)
    charge_kind = classify_other_charge_kind(item.get("category"), item.get("description"))
    descriptor_key = extract_other_charge_descriptor_signature(item.get("category"), item.get("description"))
    item_key = f"{charge_kind.casefold()}:{amount_key}:{descriptor_key}"
    return f"other_charges[{slugify_key(item_key)}]"


def _strict_match_maps(predicted: dict[str, Any], ground_truth: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    evaluator = MedicalClaimEvaluator()

    diagnosis_matches = evaluator._evaluate_diagnoses_strict(
        predicted.get("medical_context", {}).get("diagnoses", []),
        ground_truth.get("medical_context", {}).get("diagnoses", []),
    ).get("matches", [])
    medicine_matches = evaluator._evaluate_medicines(
        predicted.get("medicines", []),
        ground_truth.get("medicines", []),
    ).get("per_field", {}).get("medicines", {}).get("strict_matches", [])
    other_charge_matches = evaluator._evaluate_other_charges(
        predicted.get("other_charges", []),
        ground_truth.get("other_charges", []),
    ).get("per_field", {}).get("other_charges", {}).get("strict_matches", [])

    return {
        "diagnoses": {
            _build_path_prefix("diagnoses", match["predicted"]): match["ground_truth"]
            for match in diagnosis_matches
            if isinstance(match, dict)
        },
        "medicines": {
            _build_path_prefix("medicines", match["predicted"]): match["ground_truth"]
            for match in medicine_matches
            if isinstance(match, dict)
        },
        "other_charges": {
            _build_path_prefix("other_charges", match["predicted"]): match["ground_truth"]
            for match in other_charge_matches
            if isinstance(match, dict)
        },
    }


def _compare_scalar_field(field_path: str, predicted_value: Any, ground_truth_value: Any) -> bool:
    if isinstance(ground_truth_value, (list, tuple)):
        return _list_exact_match(predicted_value, ground_truth_value)
    if isinstance(ground_truth_value, (int, float)):
        return _num_match(predicted_value, ground_truth_value)
    if field_path in CASE_INSENSITIVE_FIELDS:
        return _case_insensitive_match(predicted_value, ground_truth_value)
    return _exact_match(predicted_value, ground_truth_value)


def _lookup_nested_value(payload: dict[str, Any], field_path: str) -> Any:
    current: Any = payload
    for part in field_path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _field_label(
    *,
    field_path: str,
    predicted: dict[str, Any],
    ground_truth: dict[str, Any],
    evaluator_score: dict[str, Any],
    strict_maps: dict[str, dict[str, dict[str, Any]]],
) -> int | None:
    per_field = evaluator_score.get("per_field", {})
    if field_path in per_field and isinstance(per_field[field_path], dict):
        metrics = per_field[field_path]
        strict_f1 = metrics.get("strict_f1")
        if isinstance(strict_f1, (int, float)):
            return 1 if strict_f1 >= 1.0 else 0
        f1 = metrics.get("f1")
        if isinstance(f1, (int, float)):
            return 1 if f1 >= 1.0 else 0

    if "[" not in field_path:
        predicted_value = _lookup_nested_value(predicted, field_path)
        ground_truth_value = _lookup_nested_value(ground_truth, field_path)
        return 1 if _compare_scalar_field(field_path, predicted_value, ground_truth_value) else 0

    list_name, remainder = field_path.split("[", 1)
    path_suffix = remainder.split("].", 1)
    if len(path_suffix) != 2:
        return None
    path_prefix = f"{list_name}[{path_suffix[0]}]"
    field_name = path_suffix[1]
    matched_gt_item = strict_maps.get(list_name, {}).get(path_prefix)
    if not isinstance(matched_gt_item, dict):
        return 0

    predicted_items = predicted.get("medical_context", {}).get("diagnoses", []) if list_name == "diagnoses" else predicted.get(list_name, [])
    predicted_item = next(
        (item for item in predicted_items if isinstance(item, dict) and _build_path_prefix(list_name, item) == path_prefix),
        None,
    )
    if not isinstance(predicted_item, dict):
        return 0

    normalized_field_path = f"{list_name}.{field_name}"
    return 1 if _compare_scalar_field(normalized_field_path, predicted_item.get(field_name), matched_gt_item.get(field_name)) else 0


def collect_calibration_rows(dataset_dir: Path, results_dir: Path) -> list[CalibrationRow]:
    rows: list[CalibrationRow] = []
    evaluator = MedicalClaimEvaluator()
    claim_dirs = sorted(path for path in dataset_dir.iterdir() if path.is_dir() and path.name != "results")
    for claim_dir in claim_dirs:
        gt_data, _image_paths = load_ground_truth(claim_dir)
        predicted, field_confidence = _load_prediction_artifacts(results_dir, claim_dir.name)
        score = evaluator.evaluate(predicted, gt_data["expected"])
        strict_maps = _strict_match_maps(predicted, gt_data["expected"])

        for field_path, meta in field_confidence.items():
            if not isinstance(meta, dict):
                continue
            label = _field_label(
                field_path=field_path,
                predicted=predicted,
                ground_truth=gt_data["expected"],
                evaluator_score=score,
                strict_maps=strict_maps,
            )
            if label is None:
                continue
            features = {
                feature_name: float(meta.get(feature_name, 0.0))
                for feature_name in SELF_CONSISTENCY_CALIBRATION_FEATURES
            }
            rows.append(
                CalibrationRow(
                    claim_id=claim_dir.name,
                    field_path=field_path,
                    features=features,
                    label=label,
                )
            )
    return rows


def build_claim_folds(claim_ids: list[str], fold_count: int) -> list[list[str]]:
    normalized_fold_count = max(2, min(fold_count, len(claim_ids)))
    folds: list[list[str]] = [[] for _ in range(normalized_fold_count)]
    for index, claim_id in enumerate(sorted(claim_ids)):
        folds[index % normalized_fold_count].append(claim_id)
    return folds


def evaluate_cross_validation(rows: list[CalibrationRow], *, fold_count: int = 5) -> dict[str, Any]:
    claim_ids = sorted({row.claim_id for row in rows})
    folds = build_claim_folds(claim_ids, fold_count)
    fold_reports: list[dict[str, Any]] = []
    all_probabilities: list[float] = []
    all_labels: list[int] = []

    for fold_index, validation_claim_ids in enumerate(folds):
        validation_claim_id_set = set(validation_claim_ids)
        train_rows = [row for row in rows if row.claim_id not in validation_claim_id_set]
        validation_rows = [row for row in rows if row.claim_id in validation_claim_id_set]
        coefficients, intercept = fit_logistic_calibrator(train_rows)

        probabilities = [
            predict_probability(
                SELF_CONSISTENCY_CALIBRATION_FEATURES,
                coefficients,
                intercept,
                row.features,
            )
            for row in validation_rows
        ]
        labels = [row.label for row in validation_rows]
        all_probabilities.extend(probabilities)
        all_labels.extend(labels)
        fold_reports.append(
            round_metrics(
                {
                    "fold_index": fold_index,
                    "validation_claim_ids": validation_claim_ids,
                    "row_count": len(validation_rows),
                    "accuracy": compute_accuracy(probabilities, labels),
                    "brier_score": compute_brier_score(probabilities, labels),
                    "ece": compute_ece(probabilities, labels),
                }
            )
        )

    return {
        "folds": fold_reports,
        "aggregate": round_metrics(
            {
                "row_count": len(all_labels),
                "accuracy": compute_accuracy(all_probabilities, all_labels),
                "brier_score": compute_brier_score(all_probabilities, all_labels),
                "ece": compute_ece(all_probabilities, all_labels),
            }
        ),
    }


def write_artifact(path: Path, rows: list[CalibrationRow]) -> dict[str, Any]:
    coefficients, intercept = fit_logistic_calibrator(rows)
    payload = {
        "calibration_version": CALIBRATION_VERSION,
        "feature_names": list(SELF_CONSISTENCY_CALIBRATION_FEATURES),
        "coefficients": [round(value, 10) for value in coefficients],
        "intercept": round(intercept, 10),
        "training_claim_count": len({row.claim_id for row in rows}),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as artifact_file:
        json.dump(payload, artifact_file, indent=2, ensure_ascii=False)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit self-consistency confidence calibration artifact from labeled eval claims.")
    parser.add_argument("--folder", required=True, help="Dataset root containing claim subfolders and ground-truth JSON files.")
    parser.add_argument("--results-dir", default=None, help="Results directory with cached latest.json and run logs. Defaults to <folder>/results.")
    parser.add_argument("--artifact-out", default=str(DEFAULT_SELF_CONSISTENCY_CALIBRATION_ARTIFACT_PATH), help="Path to write runtime calibration artifact JSON.")
    parser.add_argument("--report-out", default=None, help="Optional path to write cross-validation report JSON.")
    parser.add_argument("--folds", type=int, default=5, help="Claim-level fold count for cross-validation.")
    args = parser.parse_args()

    dataset_dir = Path(args.folder)
    results_dir = Path(args.results_dir) if args.results_dir else dataset_dir / "results"
    rows = collect_calibration_rows(dataset_dir, results_dir)
    if not rows:
        raise SystemExit("No calibration rows found. Ensure eval results exist under the results directory.")

    report = evaluate_cross_validation(rows, fold_count=args.folds)
    artifact_payload = write_artifact(Path(args.artifact_out), rows)

    output = {
        "calibration_version": CALIBRATION_VERSION,
        "feature_names": list(SELF_CONSISTENCY_CALIBRATION_FEATURES),
        "row_count": len(rows),
        "claim_count": len({row.claim_id for row in rows}),
        "cross_validation": report,
        "artifact": artifact_payload,
    }

    if args.report_out:
        report_path = Path(args.report_out)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as report_file:
            json.dump(output, report_file, indent=2, ensure_ascii=False)

    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
