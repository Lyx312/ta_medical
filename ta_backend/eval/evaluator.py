from __future__ import annotations

import re
from typing import Any, Callable

from app.services.standard_vocabulary import (
    normalize_facility_name,
    normalize_to_evaluation_document_type,
    normalize_to_evaluation_other_charge_category,
)
from eval.config import DEFAULT_THRESHOLDS, EvalThresholds
from eval.medicine_matcher import match_medicines, normalize_medicine_name

EVALUATOR_SCHEMA_VERSION = "medical_claim_extraction_eval_v2"
PRIMARY_METRIC = "strict_kie_micro_f1"

CASE_INSENSITIVE_FIELDS = frozenset({
    "patient_info.name",
    "patient_info.gender",
    "summary.currency",
    "medicines.unit",
})

SECTION_ORDER = (
    "summary",
    "patient_info",
    "medical_context",
    "medicines",
    "other_charges",
    "financial_recap",
)

CRITICAL_FIELD_PATHS = (
    "summary.medical_facility_name",
    "summary.service_date",
    "summary.transaction_date",
    "summary.currency",
    "summary.total_amount",
    "patient_info.name",
    "financial_recap.total_medicines",
    "financial_recap.total_other_charges",
    "financial_recap.grand_total",
)

FINANCIAL_EXACT_FIELD_PATHS = (
    "summary.total_amount",
    "financial_recap.total_medicines",
    "financial_recap.total_other_charges",
    "financial_recap.grand_total",
)


def _normalize_text(text: str | None) -> list[str]:
    if not text:
        return []
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip().split()


def _normalize_exact_text(value: Any) -> str | None:
    if _is_empty(value):
        return None
    return re.sub(r"\s+", " ", str(value).strip()).casefold()


def _token_f1(pred_tokens: list[str], gt_tokens: list[str]) -> float:
    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0
    common = set(pred_tokens) & set(gt_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _token_overlap(pred_tokens: list[str], gt_tokens: list[str]) -> float:
    if not pred_tokens or not gt_tokens:
        return 0.0
    return len(set(pred_tokens) & set(gt_tokens)) / min(len(set(pred_tokens)), len(set(gt_tokens)))


def _levenshtein_distance(pred: str, gt: str) -> int:
    m, n = len(pred), len(gt)
    if m == 0:
        return n
    if n == 0:
        return m

    previous = list(range(n + 1))
    current = [0] * (n + 1)
    for i in range(1, m + 1):
        current[0] = i
        for j in range(1, n + 1):
            if pred[i - 1] == gt[j - 1]:
                current[j] = previous[j - 1]
            else:
                current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + 1)
        previous, current = current, previous
    return previous[n]


def _normalized_similarity(pred: str | None, gt: str | None) -> float:
    if not pred and not gt:
        return 1.0
    if not pred or not gt:
        return 0.0
    pred_text = str(pred).casefold().strip()
    gt_text = str(gt).casefold().strip()
    max_len = max(len(pred_text), len(gt_text), 1)
    return max(0.0, 1.0 - (_levenshtein_distance(pred_text, gt_text) / max_len))


def _compute_cer(pred: str | None, gt: str | None) -> float:
    """Compute Character Error Rate (CER). Lower is better."""
    if not pred and not gt:
        return 0.0
    if not pred or not gt:
        return 1.0
    pred_text = str(pred)
    gt_text = str(gt)
    if not gt_text:
        return 1.0 if pred_text else 0.0
    return round(_levenshtein_distance(pred_text, gt_text) / len(gt_text), 4)


def _compute_anls(pred: str | None, gt: str | None, threshold: float = 0.5) -> float:
    """Compute DocVQA-style ANLS using normalized Levenshtein distance."""
    if not pred and not gt:
        return 1.0
    if not pred or not gt:
        return 0.0
    pred_text = str(pred)
    gt_text = str(gt)
    max_len = max(len(pred_text), len(gt_text), 1)
    normalized_distance = _levenshtein_distance(pred_text.casefold(), gt_text.casefold()) / max_len
    if normalized_distance >= threshold:
        return 0.0
    return round(1.0 - normalized_distance, 4)


def _exact_match(pred: Any, gt: Any) -> bool:
    if pred is None and gt is None:
        return True
    if pred is None or gt is None:
        return False
    return str(pred).strip() == str(gt).strip()


def _case_insensitive_match(pred: Any, gt: Any) -> bool:
    if pred is None and gt is None:
        return True
    if pred is None or gt is None:
        return False
    return str(pred).strip().casefold() == str(gt).strip().casefold()


def _fuzzy_name_match(pred: str | None, gt: str | None) -> float:
    if not pred and not gt:
        return 1.0
    if not pred or not gt:
        return 0.0
    pred_tokens = _normalize_text(pred)
    gt_tokens = _normalize_text(gt)
    if not pred_tokens and not gt_tokens:
        return 1.0
    return _token_f1(pred_tokens, gt_tokens)


def _normalize_clinical_text(value: Any) -> str | None:
    if _is_empty(value):
        return None
    text = str(value).casefold()
    text = re.sub(r"\(\s*\+\s*\)", " ", text)
    text = re.sub(r"\bpositif\b", " ", text)
    text = re.sub(r"[+/;,|]+", " ", text)
    text = re.sub(r"[^\w\s.-]", " ", text)
    return re.sub(r"\s+", " ", text).strip() or None


def _clinical_text_match(pred: Any, gt: Any) -> bool:
    if _normalize_exact_text(pred) == _normalize_exact_text(gt):
        return True
    pred_normalized = _normalize_clinical_text(pred)
    gt_normalized = _normalize_clinical_text(gt)
    if pred_normalized is None and gt_normalized is None:
        return True
    if pred_normalized is None or gt_normalized is None:
        return False
    if pred_normalized == gt_normalized:
        return True
    pred_tokens = pred_normalized.split()
    gt_tokens = gt_normalized.split()
    return _token_f1(pred_tokens, gt_tokens) >= 0.82 or _compute_anls(pred_normalized, gt_normalized, threshold=0.35) >= 0.78


FACILITY_ALIAS_TOKENS = {
    "ubaya": ("universitas", "surabaya"),
}


def _facility_tokens(value: Any) -> list[str]:
    normalized = normalize_facility_name(value) if not _is_empty(value) else None
    if not normalized:
        return []
    tokens = _normalize_text(normalized)
    expanded: list[str] = []
    for token in tokens:
        expanded.extend(FACILITY_ALIAS_TOKENS.get(token, (token,)))
    return [token for token in expanded if token not in {"klinik", "clinic", "pratama", "utama"}]


def _facility_match(pred_value: Any, gt_value: Any) -> bool:
    pred_normalized = normalize_facility_name(pred_value) if not _is_empty(pred_value) else None
    gt_normalized = normalize_facility_name(gt_value) if not _is_empty(gt_value) else None
    if pred_normalized == gt_normalized:
        return True
    pred_tokens = _facility_tokens(pred_value)
    gt_tokens = _facility_tokens(gt_value)
    if not pred_tokens or not gt_tokens:
        return False
    return _token_overlap(pred_tokens, gt_tokens) >= 0.80 and _token_f1(pred_tokens, gt_tokens) >= 0.65


def _list_exact_match(pred: Any, gt: Any) -> bool:
    pred_list = pred if isinstance(pred, list) else []
    gt_list = gt if isinstance(gt, list) else []
    if not pred_list and not gt_list:
        return True
    return sorted(str(item).strip() for item in pred_list) == sorted(str(item).strip() for item in gt_list)


def _num_match(pred: Any, gt: Any, tolerance: float = 0.0) -> bool:
    p = _to_number(pred)
    g = _to_number(gt)
    if p is None and g is None:
        return True
    if p is None or g is None:
        return False
    if tolerance == 0.0:
        return abs(p - g) < 1e-9
    return abs(p - g) <= tolerance


def _to_number(value: Any) -> float | None:
    if _is_empty(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _number_key(value: Any) -> str | None:
    number = _to_number(value)
    if number is None:
        return None
    if number.is_integer():
        return str(int(number))
    return str(number)


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0
    return False


def _is_empty_item(item: Any) -> bool:
    if not isinstance(item, dict):
        return _is_empty(item)
    return all(_is_empty(value) for value in item.values())


def _metric_dict(tp: float, fp: float, fn: float, **extra: Any) -> dict[str, Any]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    result: dict[str, Any] = {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": round(tp, 4),
        "fp": round(fp, 4),
        "fn": round(fn, 4),
        "applicable": (tp + fp + fn) > 0,
    }
    result.update(extra)
    return result


def _with_strict_aliases(metrics: dict[str, Any]) -> dict[str, Any]:
    result = dict(metrics)
    result.update(
        strict_precision=result.get("precision", 0.0),
        strict_recall=result.get("recall", 0.0),
        strict_f1=result.get("f1", 0.0),
        strict_tp=result.get("tp", 0),
        strict_fp=result.get("fp", 0),
        strict_fn=result.get("fn", 0),
    )
    return result


def _add_counts(target: dict[str, float], source: dict[str, Any]) -> None:
    target["tp"] = target.get("tp", 0.0) + float(source.get("tp", 0.0))
    target["fp"] = target.get("fp", 0.0) + float(source.get("fp", 0.0))
    target["fn"] = target.get("fn", 0.0) + float(source.get("fn", 0.0))


def _merge_metric_counts(metrics: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    counts = {"tp": 0.0, "fp": 0.0, "fn": 0.0}
    for metric in metrics:
        _add_counts(counts, metric)
    return _with_strict_aliases(_metric_dict(counts["tp"], counts["fp"], counts["fn"], **extra))


def _build_lcs_index_pairs(pred: list[str], gt: list[str]) -> list[tuple[int, int]]:
    if not pred or not gt:
        return []

    dp = [[0] * (len(gt) + 1) for _ in range(len(pred) + 1)]
    for pred_index in range(len(pred) - 1, -1, -1):
        for gt_index in range(len(gt) - 1, -1, -1):
            if pred[pred_index] == gt[gt_index]:
                dp[pred_index][gt_index] = dp[pred_index + 1][gt_index + 1] + 1
            else:
                dp[pred_index][gt_index] = max(dp[pred_index + 1][gt_index], dp[pred_index][gt_index + 1])

    matches: list[tuple[int, int]] = []
    pred_index = 0
    gt_index = 0
    while pred_index < len(pred) and gt_index < len(gt):
        if pred[pred_index] == gt[gt_index]:
            matches.append((pred_index, gt_index))
            pred_index += 1
            gt_index += 1
        elif dp[pred_index + 1][gt_index] >= dp[pred_index][gt_index + 1]:
            pred_index += 1
        else:
            gt_index += 1
    return matches


def _normalize_detected_document_sequence(values: list[Any]) -> list[str]:
    return [normalize_to_evaluation_document_type(str(value)) or "other" for value in values if not _is_empty(value)]


def _build_unmatched_document_items(
    raw_values: list[Any],
    normalized_values: list[str],
    matched_indices: set[int],
) -> list[dict[str, Any]]:
    return [
        {
            "index": index,
            "raw_value": raw_values[index] if index < len(raw_values) else None,
            "normalized_value": normalized_values[index],
        }
        for index in range(len(normalized_values))
        if index not in matched_indices
    ]


def _normalize_other_charge_category_for_eval(category: Any, description: Any = None) -> str | None:
    canonical = normalize_to_evaluation_other_charge_category(str(category) if category is not None else None)
    if canonical is not None:
        return canonical
    return normalize_to_evaluation_other_charge_category(str(description) if description is not None else None)


def _medicine_key(item: dict[str, Any]) -> str | None:
    tokens = normalize_medicine_name(item.get("name"))
    if not tokens:
        return None
    return " ".join(sorted(tokens))


def _medicine_name_similarity(pred: str | None, gt: str | None) -> float:
    pred_tokens = normalize_medicine_name(pred)
    gt_tokens = normalize_medicine_name(gt)
    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0
    token_sim = len(pred_tokens & gt_tokens) / len(pred_tokens | gt_tokens)
    char_sim = _normalized_similarity(" ".join(sorted(pred_tokens)), " ".join(sorted(gt_tokens)))
    return max(token_sim, char_sim)


def _match_medicines_conservative(
    pred: list[dict[str, Any]],
    gt: list[dict[str, Any]],
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    exact_matches, exact_unmatched_pred, exact_unmatched_gt = _match_by_key(pred, gt, _medicine_key)
    matched: list[tuple[int, int, float]] = [(pred_index, gt_index, 1.0) for pred_index, gt_index in exact_matches]
    matched_pred: set[int] = {pred_index for pred_index, _gt_index in exact_matches}
    matched_gt: set[int] = {gt_index for _pred_index, gt_index in exact_matches}

    candidates: list[tuple[float, int, int]] = []
    for pred_index in exact_unmatched_pred:
        pred_item = pred[pred_index]
        for gt_index in exact_unmatched_gt:
            gt_item = gt[gt_index]
            name_sim = _medicine_name_similarity(pred_item.get("name"), gt_item.get("name"))
            if name_sim < 0.78:
                continue
            candidates.append((name_sim, pred_index, gt_index))

    candidates.sort(reverse=True)
    for similarity, pred_index, gt_index in candidates:
        if pred_index in matched_pred or gt_index in matched_gt:
            continue
        matched.append((pred_index, gt_index, round(similarity, 4)))
        matched_pred.add(pred_index)
        matched_gt.add(gt_index)

    unmatched_pred = [index for index in range(len(pred)) if index not in matched_pred]
    unmatched_gt = [index for index in range(len(gt)) if index not in matched_gt]
    matched.sort(key=lambda item: item[0])
    return matched, unmatched_pred, unmatched_gt


def _diagnosis_key(item: dict[str, Any]) -> str | None:
    return _normalize_exact_text(item.get("description"))


def _other_charge_key(item: dict[str, Any]) -> str | None:
    category = _normalize_other_charge_category_for_eval(item.get("category"), item.get("description"))
    amount = _number_key(item.get("amount"))
    if category is None or amount is None:
        return None
    return f"{category.casefold()}:{amount}"


def _filter_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in items if isinstance(item, dict) and not _is_empty_item(item)]


def _match_by_key(
    pred: list[dict[str, Any]],
    gt: list[dict[str, Any]],
    key_fn: Callable[[dict[str, Any]], str | None],
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    gt_key_to_indices: dict[str, list[int]] = {}
    for gt_index, gt_item in enumerate(gt):
        key = key_fn(gt_item)
        if key is not None:
            gt_key_to_indices.setdefault(key, []).append(gt_index)

    matched: list[tuple[int, int]] = []
    matched_gt: set[int] = set()
    unmatched_pred: list[int] = []
    for pred_index, pred_item in enumerate(pred):
        key = key_fn(pred_item)
        if key is None:
            unmatched_pred.append(pred_index)
            continue
        candidates = [index for index in gt_key_to_indices.get(key, []) if index not in matched_gt]
        if candidates:
            gt_index = candidates[0]
            matched.append((pred_index, gt_index))
            matched_gt.add(gt_index)
        else:
            unmatched_pred.append(pred_index)

    matched_pred = {pred_index for pred_index, _ in matched}
    unmatched_gt = [index for index in range(len(gt)) if index not in matched_gt]
    unmatched_pred.extend(index for index in range(len(pred)) if index not in matched_pred and index not in unmatched_pred)
    return matched, unmatched_pred, unmatched_gt


class MedicalClaimEvaluator:
    def __init__(self, thresholds: EvalThresholds = DEFAULT_THRESHOLDS):
        self.t = thresholds

    def evaluate(
        self,
        predicted: dict[str, Any],
        ground_truth: dict[str, Any],
    ) -> dict[str, Any]:
        section_results: dict[str, dict[str, Any]] = {}
        per_field: dict[str, dict[str, Any]] = {}
        text_entries: list[dict[str, Any]] = []
        entity_metrics: dict[str, dict[str, Any]] = {}
        details: dict[str, dict[str, Any]] = {}

        for section_name, evaluator in (
            ("summary", self._evaluate_summary),
            ("patient_info", self._evaluate_patient_info),
            ("medical_context", self._evaluate_medical_context),
            ("medicines", self._evaluate_medicines),
            ("other_charges", self._evaluate_other_charges),
            ("financial_recap", self._evaluate_financial_recap),
        ):
            result = evaluator(
                predicted.get(section_name, {} if section_name not in {"medicines", "other_charges"} else []),
                ground_truth.get(section_name, {} if section_name not in {"medicines", "other_charges"} else []),
            )
            section_results[section_name] = result["section"]
            per_field.update(result.get("per_field", {}))
            text_entries.extend(result.get("text_entries", []))
            entity_metrics.update(result.get("entity_metrics", {}))
            if result.get("details"):
                details[section_name] = result["details"]

        strict_kie_micro = self._combine_sections(section_results)
        strict_kie_section_macro_f1 = self._macro_f1(section_results.values())
        field_exact_accuracy = self._compute_exact_accuracy(per_field, CRITICAL_FIELD_PATHS)
        financial_exact_accuracy = self._compute_exact_accuracy(per_field, FINANCIAL_EXACT_FIELD_PATHS)
        relaxed_text = self._aggregate_text_entries(text_entries)
        edit_burden_score = self._compute_edit_burden(per_field, entity_metrics, text_entries)
        verify_only_reasons = self._collect_verify_only_fail_reasons(section_results, per_field, entity_metrics, text_entries)
        blocking_fail_reasons = verify_only_reasons["blocking_fail_reasons"]
        advisory_fail_reasons = verify_only_reasons["advisory_fail_reasons"]
        pass_verify_only = self._compute_verify_only_gate(
            edit_burden_score,
            blocking_fail_reasons,
        )

        return {
            "schema_version": EVALUATOR_SCHEMA_VERSION,
            "primary_metric": PRIMARY_METRIC,
            "strict_kie_micro": strict_kie_micro,
            "strict_kie_section_macro_f1": strict_kie_section_macro_f1,
            "per_section": section_results,
            "per_field": per_field,
            "entity_metrics": entity_metrics,
            "field_exact_accuracy": field_exact_accuracy,
            "financial_exact_accuracy": financial_exact_accuracy,
            "relaxed_text": relaxed_text,
            "diagnostics": {
                "usability": {
                    "edit_burden_score": round(edit_burden_score, 4),
                    "pass_verify_only": pass_verify_only,
                    "blocking_fail_reasons": blocking_fail_reasons,
                    "advisory_fail_reasons": advisory_fail_reasons,
                    "critical_fail_reasons": blocking_fail_reasons,
                }
            },
            "details": details,
            "medicine_details": details.get("medicines", {}),
            "other_charges_details": details.get("other_charges", {}),
        }

    def _evaluate_summary(self, pred: dict[str, Any], gt: dict[str, Any]) -> dict[str, Any]:
        pred_dict = pred if isinstance(pred, dict) else {}
        gt_dict = gt if isinstance(gt, dict) else {}
        field_metrics: dict[str, dict[str, Any]] = {}
        text_entries: list[dict[str, Any]] = []

        detected = self._evaluate_detected_documents(
            pred_dict.get("detected_documents", []),
            gt_dict.get("detected_documents", []),
        )
        field_metrics["summary.detected_documents"] = detected

        for field_name, match_kind in (
            ("medical_facility_name", "facility"),
            ("service_date", "exact"),
            ("transaction_date", "exact"),
            ("currency", "case_insensitive"),
            ("total_amount", "numeric"),
        ):
            field_path = f"summary.{field_name}"
            metric = self._evaluate_scalar_value(
                field_path,
                pred_dict.get(field_name),
                gt_dict.get(field_name),
                match_kind,
            )
            field_metrics[field_path] = metric
            if field_name == "medical_facility_name":
                entry = self._text_entry(field_path, pred_dict.get(field_name), gt_dict.get(field_name))
                if entry is not None:
                    text_entries.append(entry)

        return {
            "section": _merge_metric_counts(list(field_metrics.values())),
            "per_field": field_metrics,
            "text_entries": text_entries,
        }

    def _evaluate_detected_documents(self, pred: list, gt: list) -> dict[str, Any]:
        pred_raw = [value for value in (pred if isinstance(pred, list) else []) if not _is_empty(value)]
        gt_raw = [value for value in (gt if isinstance(gt, list) else []) if not _is_empty(value)]
        pred_types = _normalize_detected_document_sequence(pred_raw)
        gt_types = _normalize_detected_document_sequence(gt_raw)

        matched_pairs = _build_lcs_index_pairs(pred_types, gt_types)
        matched_pred_indices = {pred_index for pred_index, _ in matched_pairs}
        matched_gt_indices = {gt_index for _, gt_index in matched_pairs}
        metric = _metric_dict(
            len(matched_pairs),
            len(pred_types) - len(matched_pairs),
            len(gt_types) - len(matched_pairs),
            predicted_sequence=pred_types,
            ground_truth_sequence=gt_types,
            matched_sequence=[pred_types[pred_index] for pred_index, _ in matched_pairs],
            sequence_exact=pred_types == gt_types,
            matches=[
                {
                    "predicted_index": pred_index,
                    "ground_truth_index": gt_index,
                    "predicted_raw": pred_raw[pred_index],
                    "ground_truth_raw": gt_raw[gt_index],
                    "normalized_value": pred_types[pred_index],
                }
                for pred_index, gt_index in matched_pairs
            ],
            unmatched_predicted=_build_unmatched_document_items(pred_raw, pred_types, matched_pred_indices),
            unmatched_ground_truth=_build_unmatched_document_items(gt_raw, gt_types, matched_gt_indices),
        )
        return _with_strict_aliases(metric)

    def _evaluate_detected_documents_strict(self, pred: list, gt: list) -> dict[str, Any]:
        return self._evaluate_detected_documents(pred, gt)

    def _evaluate_patient_info(self, pred: dict[str, Any], gt: dict[str, Any]) -> dict[str, Any]:
        pred_dict = pred if isinstance(pred, dict) else {}
        gt_dict = gt if isinstance(gt, dict) else {}
        field_metrics: dict[str, dict[str, Any]] = {}
        text_entries: list[dict[str, Any]] = []
        for field_name, match_kind, is_text in (
            ("name", "case_insensitive", True),
            ("date_of_birth", "exact", False),
            ("age", "numeric", False),
            ("gender", "case_insensitive", False),
            ("address", "text_exact", True),
        ):
            field_path = f"patient_info.{field_name}"
            metric = self._evaluate_scalar_value(
                field_path,
                pred_dict.get(field_name),
                gt_dict.get(field_name),
                match_kind,
            )
            field_metrics[field_path] = metric
            if is_text:
                entry = self._text_entry(field_path, pred_dict.get(field_name), gt_dict.get(field_name))
                if entry is not None:
                    text_entries.append(entry)
        return {
            "section": _merge_metric_counts(list(field_metrics.values())),
            "per_field": field_metrics,
            "text_entries": text_entries,
        }

    def _evaluate_medical_context(self, pred: dict[str, Any], gt: dict[str, Any]) -> dict[str, Any]:
        pred_dict = pred if isinstance(pred, dict) else {}
        gt_dict = gt if isinstance(gt, dict) else {}
        field_metrics: dict[str, dict[str, Any]] = {}
        text_entries: list[dict[str, Any]] = []

        for field_name in ("anamnesis", "therapy_note"):
            field_path = f"medical_context.{field_name}"
            metric = self._evaluate_scalar_value(
                field_path,
                pred_dict.get(field_name),
                gt_dict.get(field_name),
                "clinical_text",
            )
            field_metrics[field_path] = metric
            entry = self._text_entry(field_path, pred_dict.get(field_name), gt_dict.get(field_name))
            if entry is not None:
                text_entries.append(entry)

        diagnosis_result = self._evaluate_diagnoses(
            pred_dict.get("diagnoses", []),
            gt_dict.get("diagnoses", []),
        )
        field_metrics.update(diagnosis_result["per_field"])
        text_entries.extend(diagnosis_result.get("text_entries", []))

        return {
            "section": _merge_metric_counts(list(field_metrics.values())),
            "per_field": field_metrics,
            "text_entries": text_entries,
            "entity_metrics": {"diagnoses": diagnosis_result["entity_metric"]},
            "details": {"diagnoses": diagnosis_result["details"]},
        }

    def _evaluate_diagnoses(self, pred: list[dict[str, Any]], gt: list[dict[str, Any]]) -> dict[str, Any]:
        pred_items = _filter_items(pred if isinstance(pred, list) else [])
        gt_items = _filter_items(gt if isinstance(gt, list) else [])
        matched, unmatched_pred, unmatched_gt = _match_by_key(pred_items, gt_items, _diagnosis_key)
        entity_metric = self._build_entity_metric(
            pred_items,
            gt_items,
            matched,
            unmatched_pred,
            unmatched_gt,
        )
        field_metrics: list[dict[str, Any]] = [entity_metric]
        field_counts: dict[str, dict[str, float]] = {}
        text_entries: list[dict[str, Any]] = []
        strict_matches: list[dict[str, Any]] = []

        for pred_index, gt_index in matched:
            pred_item = pred_items[pred_index]
            gt_item = gt_items[gt_index]
            strict_matches.append({"predicted": pred_item, "ground_truth": gt_item})
            doctor_metric = self._evaluate_scalar_value(
                "medical_context.diagnoses.doctor_name",
                pred_item.get("doctor_name"),
                gt_item.get("doctor_name"),
                "text_exact",
            )
            _add_counts(field_counts.setdefault("medical_context.diagnoses.doctor_name", {}), doctor_metric)
            field_metrics.append(doctor_metric)
            for field_name in ("description", "doctor_name"):
                entry = self._text_entry(
                    f"medical_context.diagnoses.{field_name}",
                    pred_item.get(field_name),
                    gt_item.get(field_name),
                )
                if entry is not None:
                    text_entries.append(entry)

        per_field = {
            "medical_context.diagnoses": _with_strict_aliases(
                {
                    **entity_metric,
                    "matches": strict_matches,
                    "strict_matches": strict_matches,
                    "unmatched_predicted": [pred_items[index] for index in unmatched_pred],
                    "unmatched_ground_truth": [gt_items[index] for index in unmatched_gt],
                }
            )
        }
        for field_path, counts in field_counts.items():
            per_field[field_path] = _with_strict_aliases(_metric_dict(counts.get("tp", 0.0), counts.get("fp", 0.0), counts.get("fn", 0.0)))

        return {
            "entity_metric": entity_metric,
            "per_field": per_field,
            "text_entries": text_entries,
            "details": {
                "matches": strict_matches,
                "unmatched_predicted": [pred_items[index] for index in unmatched_pred],
                "unmatched_ground_truth": [gt_items[index] for index in unmatched_gt],
            },
        }

    def _evaluate_diagnoses_strict(self, pred: list[dict[str, Any]], gt: list[dict[str, Any]]) -> dict[str, Any]:
        result = self._evaluate_diagnoses(pred, gt)
        metric = result["entity_metric"]
        details = result["details"]
        return {
            **metric,
            "matches": details.get("matches", []),
            "unmatched_predicted": details.get("unmatched_predicted", []),
            "unmatched_ground_truth": details.get("unmatched_ground_truth", []),
        }

    def _evaluate_medicines(self, pred: list[dict[str, Any]], gt: list[dict[str, Any]]) -> dict[str, Any]:
        pred_items = _filter_items(pred if isinstance(pred, list) else [])
        gt_items = _filter_items(gt if isinstance(gt, list) else [])
        matched, unmatched_pred, unmatched_gt = _match_medicines_conservative(pred_items, gt_items)
        entity_metric = self._build_entity_metric(
            pred_items,
            gt_items,
            matched,
            unmatched_pred,
            unmatched_gt,
        )
        field_metrics: list[dict[str, Any]] = [entity_metric]
        field_counts: dict[str, dict[str, float]] = {}
        text_entries: list[dict[str, Any]] = []
        strict_matches: list[dict[str, Any]] = []

        for pred_index, gt_index, similarity in matched:
            pred_item = pred_items[pred_index]
            gt_item = gt_items[gt_index]
            strict_matches.append({"predicted": pred_item, "ground_truth": gt_item, "similarity": similarity})
            for field_name, match_kind, is_text in (
                ("quantity", "numeric", False),
                ("unit", "case_insensitive", False),
                ("dosage_instruction", "text_exact", True),
                ("unit_price", "numeric", False),
                ("total_price", "numeric", False),
            ):
                field_path = f"medicines.{field_name}"
                metric = self._evaluate_scalar_value(
                    field_path,
                    pred_item.get(field_name),
                    gt_item.get(field_name),
                    match_kind,
                )
                _add_counts(field_counts.setdefault(field_path, {}), metric)
                field_metrics.append(metric)
                if is_text:
                    entry = self._text_entry(field_path, pred_item.get(field_name), gt_item.get(field_name))
                    if entry is not None:
                        text_entries.append(entry)
            entry = self._text_entry("medicines.name", pred_item.get("name"), gt_item.get("name"))
            if entry is not None:
                text_entries.append(entry)

        for pred_index, gt_index, _similarity in match_medicines(pred_items, gt_items, self.t)[0]:
            entry = self._text_entry("medicines.name.relaxed", pred_items[pred_index].get("name"), gt_items[gt_index].get("name"))
            if entry is not None:
                text_entries.append(entry)

        per_field = {
            "medicines": _with_strict_aliases(
                {
                    **entity_metric,
                    "matches": strict_matches,
                    "strict_matches": strict_matches,
                    "unmatched_predicted": [pred_items[index] for index in unmatched_pred],
                    "unmatched_ground_truth": [gt_items[index] for index in unmatched_gt],
                }
            )
        }
        for field_path, counts in field_counts.items():
            per_field[field_path] = _with_strict_aliases(_metric_dict(counts.get("tp", 0.0), counts.get("fp", 0.0), counts.get("fn", 0.0)))

        section_metric = _merge_metric_counts(field_metrics)
        details = {
            "matches": [
                {
                    "predicted": pred_items[pred_index],
                    "ground_truth": gt_items[gt_index],
                    "similarity": similarity,
                }
                for pred_index, gt_index, similarity in matched
            ],
            "unmatched_predicted": [pred_items[index] for index in unmatched_pred],
            "unmatched_ground_truth": [gt_items[index] for index in unmatched_gt],
            "matched_count": len(matched),
            "total_predicted": len(pred_items),
            "total_ground_truth": len(gt_items),
        }
        return {
            "section": section_metric,
            "per_field": per_field,
            "text_entries": text_entries,
            "entity_metrics": {"medicines": entity_metric},
            "details": details,
        }

    def _evaluate_other_charges(self, pred: list[dict[str, Any]], gt: list[dict[str, Any]]) -> dict[str, Any]:
        pred_items = _filter_items(pred if isinstance(pred, list) else [])
        gt_items = _filter_items(gt if isinstance(gt, list) else [])
        matched, unmatched_pred, unmatched_gt = _match_by_key(pred_items, gt_items, _other_charge_key)
        unmatched_pred = self._prune_amount_null_duplicate_other_charges(pred_items, matched, unmatched_pred)
        entity_metric = self._build_entity_metric(
            pred_items,
            gt_items,
            matched,
            unmatched_pred,
            unmatched_gt,
        )
        field_metrics: list[dict[str, Any]] = [entity_metric]
        field_counts: dict[str, dict[str, float]] = {}
        text_entries: list[dict[str, Any]] = []
        strict_matches: list[dict[str, Any]] = []

        for pred_index, gt_index in matched:
            pred_item = pred_items[pred_index]
            gt_item = gt_items[gt_index]
            pred_category = _normalize_other_charge_category_for_eval(pred_item.get("category"), pred_item.get("description"))
            gt_category = _normalize_other_charge_category_for_eval(gt_item.get("category"), gt_item.get("description"))
            strict_matches.append(
                {
                    "predicted": pred_item,
                    "ground_truth": gt_item,
                    "predicted_category_canonical": pred_category,
                    "ground_truth_category_canonical": gt_category,
                }
            )
            description_metric = self._evaluate_scalar_value(
                "other_charges.description",
                pred_item.get("description"),
                gt_item.get("description"),
                "text_exact",
            )
            _add_counts(field_counts.setdefault("other_charges.description", {}), description_metric)
            field_metrics.append(description_metric)
            entry = self._text_entry("other_charges.description", pred_item.get("description"), gt_item.get("description"))
            if entry is not None:
                text_entries.append(entry)

        per_field = {
            "other_charges": _with_strict_aliases(
                {
                    **entity_metric,
                    "matches": strict_matches,
                    "strict_matches": strict_matches,
                    "unmatched_predicted": [pred_items[index] for index in unmatched_pred],
                    "unmatched_ground_truth": [gt_items[index] for index in unmatched_gt],
                }
            )
        }
        for field_path, counts in field_counts.items():
            per_field[field_path] = _with_strict_aliases(_metric_dict(counts.get("tp", 0.0), counts.get("fp", 0.0), counts.get("fn", 0.0)))

        section_metric = _merge_metric_counts(field_metrics)
        details = {
            "matches": [
                {
                    "predicted": match["predicted"],
                    "ground_truth": match["ground_truth"],
                    "similarity": 1.0,
                    "predicted_category_canonical": match.get("predicted_category_canonical"),
                    "ground_truth_category_canonical": match.get("ground_truth_category_canonical"),
                }
                for match in strict_matches
            ],
            "unmatched_predicted": [pred_items[index] for index in unmatched_pred],
            "unmatched_ground_truth": [gt_items[index] for index in unmatched_gt],
        }
        return {
            "section": section_metric,
            "per_field": per_field,
            "text_entries": text_entries,
            "entity_metrics": {"other_charges": entity_metric},
            "details": details,
        }

    def _evaluate_financial_recap(self, pred: dict[str, Any], gt: dict[str, Any]) -> dict[str, Any]:
        pred_dict = pred if isinstance(pred, dict) else {}
        gt_dict = gt if isinstance(gt, dict) else {}
        field_metrics: dict[str, dict[str, Any]] = {}
        for field_name in ("total_medicines", "total_other_charges", "grand_total"):
            field_path = f"financial_recap.{field_name}"
            field_metrics[field_path] = self._evaluate_scalar_value(
                field_path,
                pred_dict.get(field_name),
                gt_dict.get(field_name),
                "numeric",
            )
        return {
            "section": _merge_metric_counts(list(field_metrics.values())),
            "per_field": field_metrics,
            "text_entries": [],
        }

    def _evaluate_scalar_value(self, field_path: str, pred_value: Any, gt_value: Any, match_kind: str) -> dict[str, Any]:
        pred_empty = _is_empty(pred_value)
        gt_empty = _is_empty(gt_value)
        if pred_empty and gt_empty:
            return _with_strict_aliases(
                _metric_dict(
                    0,
                    0,
                    0,
                    predicted=pred_value,
                    ground_truth=gt_value,
                    exact_match=True,
                    status="not_applicable",
                )
            )

        if self._values_match(field_path, pred_value, gt_value, match_kind):
            return _with_strict_aliases(
                _metric_dict(
                    1,
                    0,
                    0,
                    predicted=pred_value,
                    ground_truth=gt_value,
                    exact_match=True,
                    status="match",
                )
            )
        if pred_empty:
            tp, fp, fn, status_value = 0, 0, 1, "missing"
        elif gt_empty:
            tp, fp, fn, status_value = 0, 1, 0, "extra"
        else:
            tp, fp, fn, status_value = 0, 1, 1, "mismatch"
        return _with_strict_aliases(
            _metric_dict(
                tp,
                fp,
                fn,
                predicted=pred_value,
                ground_truth=gt_value,
                exact_match=False,
                status=status_value,
            )
        )

    def _values_match(self, field_path: str, pred_value: Any, gt_value: Any, match_kind: str) -> bool:
        if match_kind == "numeric":
            return _num_match(pred_value, gt_value)
        if match_kind == "case_insensitive" or field_path in CASE_INSENSITIVE_FIELDS:
            return _case_insensitive_match(pred_value, gt_value)
        if match_kind == "facility":
            return _facility_match(pred_value, gt_value)
        if match_kind == "clinical_text":
            return _clinical_text_match(pred_value, gt_value)
        if match_kind == "text_exact":
            return _normalize_exact_text(pred_value) == _normalize_exact_text(gt_value)
        return _exact_match(pred_value, gt_value)

    def _prune_amount_null_duplicate_other_charges(
        self,
        pred_items: list[dict[str, Any]],
        matched: list[tuple[int, int]],
        unmatched_pred: list[int],
    ) -> list[int]:
        matched_pred_items = [pred_items[pred_index] for pred_index, _gt_index in matched]
        kept_unmatched: list[int] = []
        for pred_index in unmatched_pred:
            item = pred_items[pred_index]
            if not self._is_amount_null_duplicate_other_charge(item, matched_pred_items):
                kept_unmatched.append(pred_index)
        return kept_unmatched

    def _is_amount_null_duplicate_other_charge(
        self,
        item: dict[str, Any],
        matched_items: list[dict[str, Any]],
    ) -> bool:
        if _to_number(item.get("amount")) is not None:
            return False
        category = _normalize_other_charge_category_for_eval(item.get("category"), item.get("description"))
        description_tokens = _normalize_text(str(item.get("description") or item.get("category") or ""))
        for matched_item in matched_items:
            matched_category = _normalize_other_charge_category_for_eval(
                matched_item.get("category"),
                matched_item.get("description"),
            )
            if category != matched_category:
                continue
            matched_description_tokens = _normalize_text(
                str(matched_item.get("description") or matched_item.get("category") or "")
            )
            if not description_tokens or _token_overlap(description_tokens, matched_description_tokens) > 0:
                return True
        return False

    def _text_entry(self, field_path: str, pred_value: Any, gt_value: Any) -> dict[str, Any] | None:
        if _is_empty(pred_value) and _is_empty(gt_value):
            return None
        pred_text = None if _is_empty(pred_value) else str(pred_value)
        gt_text = None if _is_empty(gt_value) else str(gt_value)
        return {
            "field_path": field_path,
            "predicted": pred_value,
            "ground_truth": gt_value,
            "anls": _compute_anls(pred_text, gt_text),
            "token_f1": round(_token_f1(_normalize_text(pred_text), _normalize_text(gt_text)), 4),
            "cer": _compute_cer(pred_text, gt_text),
        }

    def _build_entity_metric(
        self,
        pred_items: list[dict[str, Any]],
        gt_items: list[dict[str, Any]],
        matched: list[tuple[int, int]],
        unmatched_pred: list[int],
        unmatched_gt: list[int],
    ) -> dict[str, Any]:
        return _with_strict_aliases(
            _metric_dict(
                len(matched),
                len(unmatched_pred),
                len(unmatched_gt),
                matched_count=len(matched),
                total_predicted=len(pred_items),
                total_ground_truth=len(gt_items),
            )
        )

    def _combine_sections(self, section_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
        return _merge_metric_counts([section_results[section] for section in SECTION_ORDER if section in section_results])

    def _macro_f1(self, metrics: Any) -> float:
        applicable = [float(metric.get("f1", 0.0)) for metric in metrics if metric.get("applicable")]
        if not applicable:
            return 0.0
        return round(sum(applicable) / len(applicable), 4)

    def _compute_exact_accuracy(self, per_field: dict[str, dict[str, Any]], field_paths: tuple[str, ...]) -> dict[str, Any]:
        evaluated = 0
        correct = 0
        incorrect_fields: list[str] = []
        for field_path in field_paths:
            metric = per_field.get(field_path)
            if not metric or not metric.get("applicable"):
                continue
            evaluated += 1
            if metric.get("tp", 0) == 1 and metric.get("fp", 0) == 0 and metric.get("fn", 0) == 0:
                correct += 1
            else:
                incorrect_fields.append(field_path)
        accuracy = correct / evaluated if evaluated else 0.0
        return {
            "accuracy": round(accuracy, 4),
            "correct": correct,
            "total": evaluated,
            "incorrect_fields": incorrect_fields,
        }

    def _aggregate_text_entries(self, entries: list[dict[str, Any]]) -> dict[str, Any]:
        if not entries:
            return {
                "average_anls": 0.0,
                "average_token_f1": 0.0,
                "average_cer": 0.0,
                "count": 0,
                "items": [],
                "by_field": {},
            }

        by_field: dict[str, list[dict[str, Any]]] = {}
        for entry in entries:
            by_field.setdefault(str(entry["field_path"]), []).append(entry)

        field_summary: dict[str, dict[str, Any]] = {}
        for field_path, field_entries in by_field.items():
            field_summary[field_path] = {
                "average_anls": round(sum(float(item["anls"]) for item in field_entries) / len(field_entries), 4),
                "average_token_f1": round(sum(float(item["token_f1"]) for item in field_entries) / len(field_entries), 4),
                "average_cer": round(sum(float(item["cer"]) for item in field_entries) / len(field_entries), 4),
                "count": len(field_entries),
            }

        return {
            "average_anls": round(sum(float(item["anls"]) for item in entries) / len(entries), 4),
            "average_token_f1": round(sum(float(item["token_f1"]) for item in entries) / len(entries), 4),
            "average_cer": round(sum(float(item["cer"]) for item in entries) / len(entries), 4),
            "count": len(entries),
            "items": entries,
            "by_field": field_summary,
        }

    def _compute_edit_burden(
        self,
        per_field: dict[str, dict[str, Any]],
        entity_metrics: dict[str, dict[str, Any]],
        text_entries: list[dict[str, Any]],
    ) -> float:
        burden = 0.0
        text_by_field = self._index_text_entries(text_entries)
        entity_field_paths = {"medicines", "other_charges", "medical_context.diagnoses"}

        for field_path, metric in per_field.items():
            if field_path in entity_field_paths or not metric.get("applicable"):
                continue
            if self._field_is_strict_match(metric):
                continue

            units = self._field_error_units(metric)
            if field_path in FINANCIAL_EXACT_FIELD_PATHS:
                if self._financial_field_within_verify_tolerance(metric):
                    burden += self.t.EDIT_COST_PARTIAL_MATCH
                else:
                    burden += self.t.EDIT_COST_FINANCIAL_MISMATCH
                continue

            if field_path in CRITICAL_FIELD_PATHS:
                if self._field_passes_relaxed_text_gate(field_path, text_by_field):
                    burden += self.t.EDIT_COST_PARTIAL_MATCH
                else:
                    burden += self.t.EDIT_COST_CRITICAL_FIELD
                continue

            if self._field_passes_relaxed_text_gate(field_path, text_by_field):
                burden += self.t.EDIT_COST_PARTIAL_MATCH
            else:
                burden += units * self.t.EDIT_COST_PARTIAL_MATCH

        for entity_name in ("medicines", "other_charges"):
            metric = entity_metrics.get(entity_name, {})
            burden += float(metric.get("fp", 0)) * self.t.EDIT_COST_EXTRA_ITEM
            burden += float(metric.get("fn", 0)) * self.t.EDIT_COST_MISSING_ITEM
        diagnosis_metric = entity_metrics.get("diagnoses", {})
        burden += float(diagnosis_metric.get("fp", 0) + diagnosis_metric.get("fn", 0)) * self.t.EDIT_COST_PARTIAL_MATCH
        return burden

    def _collect_verify_only_fail_reasons(
        self,
        section_results: dict[str, dict[str, Any]],
        per_field: dict[str, dict[str, Any]],
        entity_metrics: dict[str, dict[str, Any]],
        text_entries: list[dict[str, Any]],
    ) -> dict[str, list[str]]:
        blocking_reasons: list[str] = []
        advisory_reasons: list[str] = []
        text_by_field = self._index_text_entries(text_entries)
        advisory_exact_fields = {
            "summary.service_date",
            "summary.transaction_date",
            "summary.currency",
        }

        patient_name = per_field.get("patient_info.name", {})
        if patient_name.get("applicable") and not self._field_is_strict_match(patient_name):
            blocking_reasons.append("patient_info.name exact mismatch")

        for field_path in advisory_exact_fields:
            field_metrics = per_field.get(field_path, {})
            if field_metrics.get("applicable") and not self._field_is_strict_match(field_metrics):
                advisory_reasons.append(f"{field_path} exact mismatch")

        facility = per_field.get("summary.medical_facility_name", {})
        if (
            facility.get("applicable")
            and not self._field_is_strict_match(facility)
            and not self._field_passes_relaxed_text_gate("summary.medical_facility_name", text_by_field)
        ):
            advisory_reasons.append("summary.medical_facility_name below relaxed text threshold")

        for field_path in FINANCIAL_EXACT_FIELD_PATHS:
            field_metrics = per_field.get(field_path, {})
            if (
                field_metrics.get("applicable")
                and not self._field_is_strict_match(field_metrics)
                and not self._financial_field_within_verify_tolerance(field_metrics)
            ):
                blocking_reasons.append(f"{field_path} outside verify-only tolerance")

        for entity_name in ("medicines", "other_charges", "diagnoses"):
            metric = entity_metrics.get(entity_name, {})
            if metric.get("applicable") and metric.get("f1", 0.0) < self.t.VERIFY_ONLY_ENTITY_F1_THRESHOLD:
                advisory_reasons.append(f"{entity_name} KIE F1 below relaxed verify-only threshold")
        context = section_results.get("medical_context", {})
        if context.get("applicable") and context.get("f1", 0.0) < self.t.VERIFY_ONLY_CONTEXT_F1_THRESHOLD:
            advisory_reasons.append("medical_context KIE F1 below relaxed verify-only threshold")
        return {
            "blocking_fail_reasons": blocking_reasons,
            "advisory_fail_reasons": advisory_reasons,
        }

    def _compute_verify_only_gate(
        self,
        edit_burden_score: float,
        blocking_fail_reasons: list[str],
    ) -> bool:
        if blocking_fail_reasons:
            return False
        if edit_burden_score > self.t.VERIFY_ONLY_MAX_EDIT_BURDEN:
            return False
        return True

    def _field_is_strict_match(self, metric: dict[str, Any]) -> bool:
        return float(metric.get("tp", 0.0)) > 0.0 and float(metric.get("fp", 0.0)) == 0.0 and float(metric.get("fn", 0.0)) == 0.0

    def _field_error_units(self, metric: dict[str, Any]) -> float:
        status = metric.get("status")
        if status in {"missing", "extra", "mismatch"}:
            return 1.0
        return float(metric.get("fp", 0.0) + metric.get("fn", 0.0))

    def _index_text_entries(self, text_entries: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        by_field: dict[str, list[dict[str, Any]]] = {}
        for entry in text_entries:
            by_field.setdefault(str(entry.get("field_path", "")), []).append(entry)
        return by_field

    def _field_passes_relaxed_text_gate(self, field_path: str, text_by_field: dict[str, list[dict[str, Any]]]) -> bool:
        entries = text_by_field.get(field_path, [])
        if not entries:
            return False
        return any(
            float(entry.get("anls", 0.0)) >= self.t.VERIFY_ONLY_TEXT_ANLS_THRESHOLD
            or float(entry.get("token_f1", 0.0)) >= self.t.VERIFY_ONLY_TEXT_TOKEN_F1_THRESHOLD
            for entry in entries
        )

    def _financial_field_within_verify_tolerance(self, metric: dict[str, Any]) -> bool:
        pred_number = _to_number(metric.get("predicted"))
        gt_number = _to_number(metric.get("ground_truth"))
        if pred_number is None or gt_number is None:
            return False
        return abs(pred_number - gt_number) <= self._financial_verify_tolerance(gt_number)

    def _financial_verify_tolerance(self, gt_number: float) -> float:
        relative_tolerance = abs(gt_number) * self.t.VERIFY_ONLY_FINANCIAL_REL_TOLERANCE
        return min(
            self.t.VERIFY_ONLY_FINANCIAL_MAX_TOLERANCE,
            max(self.t.VERIFY_ONLY_FINANCIAL_ABS_TOLERANCE, relative_tolerance),
        )
