import re
from typing import Any

from app.services.standard_vocabulary import (
    get_document_type_display_name,
    get_category_display_name,
    map_category_to_standard,
    map_category_to_standard_strict,
    map_description_to_category,
    map_description_to_category_strict,
    normalize_facility_name,
    normalize_to_standard_document_type,
    normalize_to_standard_document_type_strict,
    STANDARD_DOCUMENT_TYPES,
    OTHER_CHARGE_CATEGORIES,
)
from eval.config import DEFAULT_THRESHOLDS, EvalThresholds
from eval.medicine_matcher import match_medicines

CASE_INSENSITIVE_FIELDS = frozenset({
    "patient_info.name",
    "patient_info.gender",
    "summary.medical_facility_name",
    "summary.currency",
})


def _normalize_text(text: str | None) -> list[str]:
    if not text:
        return []
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip().split()


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


def _compute_cer(pred: str | None, gt: str | None) -> float:
    """Compute Character Error Rate (CER). Lower is better.

    CER = (insertions + deletions + substitutions) / len(ground_truth_chars)
    """
    if not pred and not gt:
        return 0.0
    if not pred or not gt:
        return 1.0

    pred = str(pred)
    gt = str(gt)

    m, n = len(pred), len(gt)
    if m == 0:
        return 1.0 if n > 0 else 0.0

    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred[i - 1] == gt[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + 1)

    edits = dp[m][n]
    return round(edits / n, 4)


def _exact_match(pred: Any, gt: Any) -> bool:
    if pred is None and gt is None:
        return True
    if pred is None or gt is None:
        return False
    pred_str = str(pred).strip()
    gt_str = str(gt).strip()
    return pred_str == gt_str


def _case_insensitive_match(pred: Any, gt: Any) -> bool:
    if pred is None and gt is None:
        return True
    if pred is None or gt is None:
        return False
    return str(pred).strip().lower() == str(gt).strip().lower()


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


def _list_exact_match(pred: Any, gt: Any) -> bool:
    pred_list = pred if isinstance(pred, list) else []
    gt_list = gt if isinstance(gt, list) else []
    if not pred_list and not gt_list:
        return True
    return sorted(str(item).strip() for item in pred_list) == sorted(str(item).strip() for item in gt_list)


def _num_match(pred: Any, gt: Any, tolerance: float = 0.0) -> bool:
    p = float(pred) if pred is not None else None
    g = float(gt) if gt is not None else None
    if p is None and g is None:
        return True
    if p is None or g is None:
        return False
    if tolerance == 0.0:
        return abs(p - g) < 1e-9
    return abs(p - g) <= tolerance


def _metric_dict(tp: int, fp: int, fn: int, **extra: Any) -> dict[str, Any]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    result: dict[str, Any] = {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }
    result.update(extra)
    return result


def _round_metric_value(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 4)
    return value


def _round_metric_dict(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: _round_metric_value(value) for key, value in metrics.items()}


def _merge_section_tracks(soft: dict[str, Any], strict: dict[str, Any], **extra: Any) -> dict[str, Any]:
    result = {
        "precision": soft.get("precision", 0.0),
        "recall": soft.get("recall", 0.0),
        "f1": soft.get("f1", 0.0),
        "tp": soft.get("tp", 0),
        "fp": soft.get("fp", 0),
        "fn": soft.get("fn", 0),
        "soft_precision": soft.get("precision", 0.0),
        "soft_recall": soft.get("recall", 0.0),
        "soft_f1": soft.get("f1", 0.0),
        "soft_tp": soft.get("tp", 0),
        "soft_fp": soft.get("fp", 0),
        "soft_fn": soft.get("fn", 0),
        "strict_precision": strict.get("precision", 0.0),
        "strict_recall": strict.get("recall", 0.0),
        "strict_f1": strict.get("f1", 0.0),
        "strict_tp": strict.get("tp", 0),
        "strict_fp": strict.get("fp", 0),
        "strict_fn": strict.get("fn", 0),
        "soft": _round_metric_dict(soft),
        "strict": _round_metric_dict(strict),
    }
    result.update(extra)
    return result


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

        section_weights: dict[str, float] = {
            "summary": 1.0,
            "patient_info": 1.0,
            "medical_context": 1.0,
            "medicines": 2.0,
            "other_charges": 1.0,
            "financial_recap": 1.0,
        }

        summary_metrics = self._evaluate_summary(predicted.get("summary", {}), ground_truth.get("summary", {}))
        section_results["summary"] = summary_metrics["section"]
        per_field.update(summary_metrics["per_field"])

        patient_metrics = self._evaluate_patient_info(
            predicted.get("patient_info", {}),
            ground_truth.get("patient_info", {}),
        )
        section_results["patient_info"] = patient_metrics["section"]
        per_field.update(patient_metrics["per_field"])

        context_metrics = self._evaluate_medical_context(
            predicted.get("medical_context", {}),
            ground_truth.get("medical_context", {}),
        )
        section_results["medical_context"] = context_metrics["section"]
        per_field.update(context_metrics["per_field"])

        medicines_metrics = self._evaluate_medicines(
            predicted.get("medicines", []),
            ground_truth.get("medicines", []),
        )
        section_results["medicines"] = medicines_metrics["section"]
        per_field.update(medicines_metrics["per_field"])

        charges_metrics = self._evaluate_other_charges(
            predicted.get("other_charges", []),
            ground_truth.get("other_charges", []),
        )
        section_results["other_charges"] = charges_metrics["section"]
        per_field.update(charges_metrics["per_field"])

        financial_recap_metrics = self._evaluate_financial_recap(
            predicted.get("financial_recap", {}),
            ground_truth.get("financial_recap", {}),
        )
        section_results["financial_recap"] = financial_recap_metrics["section"]
        per_field.update(financial_recap_metrics["per_field"])

        soft_section_results = {section: metrics.get("soft", metrics) for section, metrics in section_results.items()}
        strict_section_results = {section: metrics.get("strict", metrics) for section, metrics in section_results.items()}

        soft_f1 = self._compute_overall_f1(soft_section_results, section_weights)
        strict_f1 = self._compute_overall_f1(strict_section_results, section_weights)

        total_soft_tp = sum(float(metrics.get("tp", 0)) for metrics in soft_section_results.values())
        total_soft_fp = sum(float(metrics.get("fp", 0)) for metrics in soft_section_results.values())
        total_soft_fn = sum(float(metrics.get("fn", 0)) for metrics in soft_section_results.values())
        overall_precision = total_soft_tp / (total_soft_tp + total_soft_fp) if (total_soft_tp + total_soft_fp) > 0 else 0.0
        overall_recall = total_soft_tp / (total_soft_tp + total_soft_fn) if (total_soft_tp + total_soft_fn) > 0 else 0.0

        edit_burden_score = self._compute_edit_burden(section_results)
        critical_fail_reasons = self._collect_critical_fail_reasons(section_results, per_field)
        pass_verify_only = self._compute_verify_only_gate(section_results, edit_burden_score, critical_fail_reasons)

        return {
            "overall_f1": round(soft_f1, 4),
            "soft_f1": round(soft_f1, 4),
            "strict_f1": round(strict_f1, 4),
            "overall_precision": round(overall_precision, 4),
            "overall_recall": round(overall_recall, 4),
            "edit_burden_score": round(edit_burden_score, 4),
            "pass_verify_only": pass_verify_only,
            "critical_fail_reasons": critical_fail_reasons,
            "per_section": section_results,
            "per_field": per_field,
            "medicine_details": medicines_metrics["section"].get("details", {}),
            "other_charges_details": charges_metrics["section"].get("details", {}),
        }

    def _evaluate_summary(self, pred: dict[str, Any], gt: dict[str, Any]) -> dict[str, Any]:
        pred_detected = pred.get("detected_documents", [])
        gt_detected = gt.get("detected_documents", [])
        detected_metrics = self._evaluate_detected_documents(pred_detected, gt_detected)

        pred_facility = pred.get("medical_facility_name")
        gt_facility = gt.get("medical_facility_name")
        facility_match = self._evaluate_facility_name(pred_facility, gt_facility)

        base_metrics = self._evaluate_scalar_fields(
            "summary",
            pred,
            gt,
            [
                ("service_date", "service_date"),
                ("transaction_date", "transaction_date"),
                ("total_amount", "total_amount"),
                ("currency", "currency"),
            ],
        )

        tp = base_metrics["section"]["tp"] + facility_match["tp"] + detected_metrics["tp"]
        fp = base_metrics["section"]["fp"] + facility_match["fp"] + detected_metrics["fp"]
        fn = base_metrics["section"]["fn"] + facility_match["fn"] + detected_metrics["fn"]

        per_field = {**base_metrics["per_field"]}
        per_field["summary.detected_documents"] = detected_metrics
        per_field["summary.medical_facility_name"] = facility_match

        soft_section = _metric_dict(tp, fp, fn)

        strict_base = self._evaluate_scalar_fields(
            "summary",
            pred,
            gt,
            [
                ("service_date", "service_date"),
                ("transaction_date", "transaction_date"),
                ("total_amount", "total_amount"),
                ("currency", "currency"),
            ],
        )
        strict_detected = self._evaluate_detected_documents_strict(pred_detected, gt_detected)
        strict_facility = self._evaluate_facility_name_strict(pred_facility, gt_facility)
        strict_tp = strict_base["section"]["tp"] + strict_detected["tp"] + strict_facility["tp"]
        strict_fp = strict_base["section"]["fp"] + strict_detected["fp"] + strict_facility["fp"]
        strict_fn = strict_base["section"]["fn"] + strict_detected["fn"] + strict_facility["fn"]

        per_field["summary.detected_documents"].update(
            strict_f1=strict_detected["f1"],
            strict_precision=strict_detected["precision"],
            strict_recall=strict_detected["recall"],
            strict_tp=strict_detected["tp"],
            strict_fp=strict_detected["fp"],
            strict_fn=strict_detected["fn"],
        )
        per_field["summary.medical_facility_name"].update(
            strict_f1=strict_facility["f1"],
            strict_precision=strict_facility["precision"],
            strict_recall=strict_facility["recall"],
            strict_tp=strict_facility["tp"],
            strict_fp=strict_facility["fp"],
            strict_fn=strict_facility["fn"],
        )
        for field_name in ("summary.service_date", "summary.transaction_date", "summary.total_amount", "summary.currency"):
            strict_field = strict_base["per_field"].get(field_name, {})
            if field_name in per_field:
                per_field[field_name].update(
                    strict_f1=strict_field.get("f1", 0.0),
                    strict_precision=strict_field.get("precision", 0.0),
                    strict_recall=strict_field.get("recall", 0.0),
                    strict_tp=strict_field.get("tp", 0),
                    strict_fp=strict_field.get("fp", 0),
                    strict_fn=strict_field.get("fn", 0),
                )

        strict_section = _metric_dict(strict_tp, strict_fp, strict_fn)
        return {
            "section": _merge_section_tracks(soft_section, strict_section),
            "per_field": per_field,
        }

    def _evaluate_detected_documents(self, pred: list, gt: list) -> dict[str, Any]:
        pred_types = [normalize_to_standard_document_type(d) for d in pred] if pred else []
        gt_types = [normalize_to_standard_document_type(d) for d in gt] if gt else []

        pred_set = set(pred_types)
        gt_set = set(gt_types)

        common = pred_set & gt_set
        tp = len(common)
        fp = len(pred_set - gt_set)
        fn = len(gt_set - pred_set)

        details = {
            "predicted_types": pred_types,
            "ground_truth_types": gt_types,
            "matched": list(common),
            "unmatched_predicted": list(pred_set - gt_set),
            "unmatched_ground_truth": list(gt_set - pred_set),
        }
        return _metric_dict(tp, fp, fn, **details)

    def _evaluate_detected_documents_strict(self, pred: list, gt: list) -> dict[str, Any]:
        pred_types = {doc_type for doc_type in (normalize_to_standard_document_type_strict(d) for d in pred or []) if doc_type}
        gt_types = {doc_type for doc_type in (normalize_to_standard_document_type_strict(d) for d in gt or []) if doc_type}
        common = pred_types & gt_types
        return _metric_dict(len(common), len(pred_types - gt_types), len(gt_types - pred_types))

    def _evaluate_facility_name(self, pred: str | None, gt: str | None) -> dict[str, Any]:
        pred_normalized = normalize_facility_name(pred) if pred else None
        gt_normalized = normalize_facility_name(gt) if gt else None

        if pred_normalized and gt_normalized:
            if pred_normalized.lower() == gt_normalized.lower():
                return _metric_dict(1, 0, 0)
            cer = _compute_cer(pred_normalized, gt_normalized)
            if cer < 0.2:
                return _metric_dict(1, 0, 0, cer=cer)
            return _metric_dict(0, 1, 1, cer=cer)
        if pred_normalized and not gt_normalized:
            return _metric_dict(0, 1, 0)
        if not pred_normalized and gt_normalized:
            return _metric_dict(0, 0, 1)
        return _metric_dict(1, 0, 0)

    def _evaluate_facility_name_strict(self, pred: str | None, gt: str | None) -> dict[str, Any]:
        pred_normalized = normalize_facility_name(pred) if pred else None
        gt_normalized = normalize_facility_name(gt) if gt else None
        if pred_normalized and gt_normalized:
            pred_tokens = set(_normalize_text(pred_normalized))
            gt_tokens = set(_normalize_text(gt_normalized))
            overlap = len(pred_tokens & gt_tokens)
            denom = max(len(gt_tokens), 1)
            token_ratio = overlap / denom
            if token_ratio >= self.t.STRICT_FACILITY_TOKEN_THRESHOLD:
                return _metric_dict(1, 0, 0, similarity=round(token_ratio, 4))
            return _metric_dict(0, 1, 1, similarity=round(token_ratio, 4))
        if pred_normalized and not gt_normalized:
            return _metric_dict(0, 1, 0)
        if not pred_normalized and gt_normalized:
            return _metric_dict(0, 0, 1)
        return _metric_dict(1, 0, 0)

    def _evaluate_patient_info(self, pred: dict[str, Any], gt: dict[str, Any]) -> dict[str, Any]:
        soft_metrics = self._evaluate_scalar_fields(
            "patient_info",
            pred,
            gt,
            [
                ("name", "name"),
                ("gender", "gender"),
                ("date_of_birth", "date_of_birth"),
                ("age", "age"),
                ("address", "address"),
            ],
        )
        for metrics in soft_metrics["per_field"].values():
            metrics.update(
                strict_f1=metrics.get("f1", 0.0),
                strict_precision=metrics.get("precision", 0.0),
                strict_recall=metrics.get("recall", 0.0),
                strict_tp=metrics.get("tp", 0),
                strict_fp=metrics.get("fp", 0),
                strict_fn=metrics.get("fn", 0),
            )
        return {
            "section": _merge_section_tracks(soft_metrics["section"], soft_metrics["section"]),
            "per_field": soft_metrics["per_field"],
        }

    def _evaluate_scalar_fields(
        self,
        section_name: str,
        pred: dict[str, Any],
        gt: dict[str, Any],
        fields: list[tuple[str, str]],
    ) -> dict[str, Any]:
        tp = fp = fn = 0
        per_field: dict[str, dict[str, Any]] = {}

        for gt_key, pred_key in fields:
            p_val = pred.get(pred_key)
            g_val = gt.get(gt_key)
            field_path = f"{section_name}.{pred_key}"
            
            if isinstance(g_val, (list, tuple)):
                is_match = _list_exact_match(p_val, g_val)
            elif isinstance(g_val, (int, float)):
                is_match = _num_match(p_val, g_val)
            elif field_path in CASE_INSENSITIVE_FIELDS:
                is_match = _case_insensitive_match(p_val, g_val)
            else:
                is_match = _exact_match(p_val, g_val)

            if is_match:
                field_tp, field_fp, field_fn = 1, 0, 0
            elif p_val is not None and g_val is None:
                field_tp, field_fp, field_fn = 0, 1, 0
            elif p_val is None and g_val is not None:
                field_tp, field_fp, field_fn = 0, 0, 1
            else:
                field_tp, field_fp, field_fn = 0, 1, 1

            tp += field_tp
            fp += field_fp
            fn += field_fn
            per_field[f"{section_name}.{pred_key}"] = _metric_dict(
                field_tp,
                field_fp,
                field_fn,
                predicted=p_val,
                ground_truth=g_val,
            )

        return {
            "section": _metric_dict(tp, fp, fn),
            "per_field": per_field,
        }

    def _evaluate_medical_context(self, pred: dict[str, Any], gt: dict[str, Any]) -> dict[str, Any]:
        anamnesis_pred = pred.get("anamnesis")
        anamnesis_gt = gt.get("anamnesis")
        anamnesis_f1 = _token_f1(_normalize_text(anamnesis_pred), _normalize_text(anamnesis_gt))
        anamnesis_cer = _compute_cer(anamnesis_pred, anamnesis_gt)
        anam_tp = 1 if anamnesis_f1 >= self.t.TEXT_TOKEN_F1_THRESHOLD else 0
        anam_fp = 0 if anam_tp else (1 if anamnesis_pred else 0)
        anam_fn = 0 if anam_tp else (1 if anamnesis_gt else 0)

        therapy_pred = pred.get("therapy_note")
        therapy_gt = gt.get("therapy_note")
        therapy_f1 = _token_f1(_normalize_text(therapy_pred), _normalize_text(therapy_gt))
        therapy_cer = _compute_cer(therapy_pred, therapy_gt)
        therapy_tp = 1 if therapy_f1 >= self.t.TEXT_TOKEN_F1_THRESHOLD else 0
        therapy_fp = 0 if therapy_tp else (1 if therapy_pred else 0)
        therapy_fn = 0 if therapy_tp else (1 if therapy_gt else 0)

        diag_metrics = self._evaluate_diagnoses(pred.get("diagnoses", []), gt.get("diagnoses", []))
        total_tp = diag_metrics["tp"] + anam_tp + therapy_tp
        total_fp = diag_metrics["fp"] + anam_fp + therapy_fp
        total_fn = diag_metrics["fn"] + anam_fn + therapy_fn

        soft_section = _metric_dict(
                total_tp,
                total_fp,
                total_fn,
                anamnesis_f1=round(anamnesis_f1, 4),
                anamnesis_cer=anamnesis_cer,
                therapy_note_f1=round(therapy_f1, 4),
                therapy_note_cer=therapy_cer,
                diagnoses_f1=round(diag_metrics["f1"], 4),
            )

        strict_text_threshold = max(self.t.TEXT_TOKEN_F1_THRESHOLD, 0.8)
        strict_anam_tp = 1 if anamnesis_f1 >= strict_text_threshold else 0
        strict_anam_fp = 0 if strict_anam_tp else (1 if anamnesis_pred else 0)
        strict_anam_fn = 0 if strict_anam_tp else (1 if anamnesis_gt else 0)
        strict_therapy_tp = 1 if therapy_f1 >= strict_text_threshold else 0
        strict_therapy_fp = 0 if strict_therapy_tp else (1 if therapy_pred else 0)
        strict_therapy_fn = 0 if strict_therapy_tp else (1 if therapy_gt else 0)
        strict_diag = self._evaluate_diagnoses_strict(pred.get("diagnoses", []), gt.get("diagnoses", []))
        strict_section = _metric_dict(
            strict_diag["tp"] + strict_anam_tp + strict_therapy_tp,
            strict_diag["fp"] + strict_anam_fp + strict_therapy_fp,
            strict_diag["fn"] + strict_anam_fn + strict_therapy_fn,
        )

        per_field = {
                "medical_context.anamnesis": _metric_dict(
                    anam_tp,
                    anam_fp,
                    anam_fn,
                    similarity=round(anamnesis_f1, 4),
                    cer=anamnesis_cer,
                    predicted=anamnesis_pred,
                    ground_truth=anamnesis_gt,
                ),
                "medical_context.therapy_note": _metric_dict(
                    therapy_tp,
                    therapy_fp,
                    therapy_fn,
                    similarity=round(therapy_f1, 4),
                    cer=therapy_cer,
                    predicted=therapy_pred,
                    ground_truth=therapy_gt,
                ),
                "medical_context.diagnoses": diag_metrics,
            }
        per_field["medical_context.anamnesis"].update(strict_f1=_metric_dict(strict_anam_tp, strict_anam_fp, strict_anam_fn)["f1"], strict_precision=_metric_dict(strict_anam_tp, strict_anam_fp, strict_anam_fn)["precision"], strict_recall=_metric_dict(strict_anam_tp, strict_anam_fp, strict_anam_fn)["recall"], strict_tp=strict_anam_tp, strict_fp=strict_anam_fp, strict_fn=strict_anam_fn)
        per_field["medical_context.therapy_note"].update(strict_f1=_metric_dict(strict_therapy_tp, strict_therapy_fp, strict_therapy_fn)["f1"], strict_precision=_metric_dict(strict_therapy_tp, strict_therapy_fp, strict_therapy_fn)["precision"], strict_recall=_metric_dict(strict_therapy_tp, strict_therapy_fp, strict_therapy_fn)["recall"], strict_tp=strict_therapy_tp, strict_fp=strict_therapy_fp, strict_fn=strict_therapy_fn)
        per_field["medical_context.diagnoses"].update(
            strict_f1=strict_diag["f1"],
            strict_precision=strict_diag["precision"],
            strict_recall=strict_diag["recall"],
            strict_tp=strict_diag["tp"],
            strict_fp=strict_diag["fp"],
            strict_fn=strict_diag["fn"],
            strict_matches=strict_diag.get("matches", []),
            strict_avg_similarity=strict_diag.get("avg_similarity", 0.0),
        )
        return {
            "section": _merge_section_tracks(soft_section, strict_section, avg_similarity=diag_metrics.get("avg_similarity", 0.0)),
            "per_field": per_field,
        }

    def _evaluate_diagnoses(self, pred: list[dict[str, Any]], gt: list[dict[str, Any]]) -> dict[str, Any]:
        if not gt:
            if not pred:
                return _metric_dict(0, 0, 0, unmatched_predicted=[], unmatched_ground_truth=[], matches=[], avg_similarity=0.0)
            return _metric_dict(0, len(pred), 0, unmatched_predicted=pred, unmatched_ground_truth=[], matches=[], avg_similarity=0.0)
        if not pred:
            return _metric_dict(0, 0, len(gt), unmatched_predicted=[], unmatched_ground_truth=gt, matches=[], avg_similarity=0.0)

        tp = fp = 0
        gt_matched = [False] * len(gt)
        unmatched_predicted: list[dict[str, Any]] = []
        matches: list[dict[str, Any]] = []

        for pred_diag in pred:
            pred_desc = pred_diag.get("description", "")
            pred_doc = pred_diag.get("doctor_name")
            best_idx = -1
            best_score = 0.0

            for index, gt_diag in enumerate(gt):
                if gt_matched[index]:
                    continue
                desc_score = _token_f1(
                    _normalize_text(pred_desc),
                    _normalize_text(gt_diag.get("description", "")),
                )
                doc_score = _fuzzy_name_match(pred_doc, gt_diag.get("doctor_name"))
                combined = (desc_score + doc_score) / 2.0
                if combined > best_score:
                    best_score = combined
                    best_idx = index

            if best_idx >= 0 and best_score >= self.t.DIAGNOSIS_SIM_THRESHOLD:
                tp += 1
                gt_matched[best_idx] = True
                matches.append(
                    {
                        "predicted": pred_diag,
                        "ground_truth": gt[best_idx],
                        "similarity": round(best_score, 4),
                    }
                )
            else:
                fp += 1
                unmatched_predicted.append(pred_diag)

        fn = sum(1 for matched in gt_matched if not matched)
        avg_similarity = sum(m["similarity"] for m in matches) / len(matches) if matches else 0.0
        return _metric_dict(
            tp,
            fp,
            fn,
            unmatched_predicted=unmatched_predicted,
            unmatched_ground_truth=[gt[index] for index, matched in enumerate(gt_matched) if not matched],
            matches=matches,
            avg_similarity=round(avg_similarity, 4),
        )

    def _evaluate_diagnoses_strict(self, pred: list[dict[str, Any]], gt: list[dict[str, Any]]) -> dict[str, Any]:
        if not gt:
            if not pred:
                return _metric_dict(0, 0, 0, unmatched_predicted=[], unmatched_ground_truth=[], matches=[], avg_similarity=0.0)
            return _metric_dict(0, len(pred), 0, unmatched_predicted=pred, unmatched_ground_truth=[], matches=[], avg_similarity=0.0)
        if not pred:
            return _metric_dict(0, 0, len(gt), unmatched_predicted=[], unmatched_ground_truth=gt, matches=[], avg_similarity=0.0)

        tp = fp = 0
        gt_matched = [False] * len(gt)
        unmatched_predicted: list[dict[str, Any]] = []
        matches: list[dict[str, Any]] = []

        for pred_diag in pred:
            pred_desc = pred_diag.get("description", "")
            pred_doc = pred_diag.get("doctor_name")
            best_idx = -1
            best_score = 0.0
            best_desc_score = 0.0
            best_doc_score = 0.0

            for index, gt_diag in enumerate(gt):
                if gt_matched[index]:
                    continue
                desc_score = _token_f1(_normalize_text(pred_desc), _normalize_text(gt_diag.get("description", "")))
                doc_score = _fuzzy_name_match(pred_doc, gt_diag.get("doctor_name"))
                combined = (desc_score + doc_score) / 2.0
                if combined > best_score:
                    best_score = combined
                    best_desc_score = desc_score
                    best_doc_score = doc_score
                    best_idx = index

            doctor_required = bool(gt[best_idx].get("doctor_name")) if best_idx >= 0 else False
            strict_pass = (
                best_idx >= 0
                and best_desc_score >= self.t.STRICT_DIAGNOSIS_DESC_THRESHOLD
                and (not doctor_required or best_doc_score >= self.t.STRICT_DIAGNOSIS_DOCTOR_THRESHOLD)
            )
            if strict_pass:
                tp += 1
                gt_matched[best_idx] = True
                matches.append({
                    "predicted": pred_diag,
                    "ground_truth": gt[best_idx],
                    "similarity": round(best_score, 4),
                    "description_similarity": round(best_desc_score, 4),
                    "doctor_similarity": round(best_doc_score, 4),
                })
            else:
                fp += 1
                unmatched_predicted.append(pred_diag)

        fn = sum(1 for matched in gt_matched if not matched)
        avg_similarity = sum(m["similarity"] for m in matches) / len(matches) if matches else 0.0
        return _metric_dict(tp, fp, fn, unmatched_predicted=unmatched_predicted, unmatched_ground_truth=[gt[index] for index, matched in enumerate(gt_matched) if not matched], matches=matches, avg_similarity=round(avg_similarity, 4))

    def _evaluate_medicines(self, pred: list[dict[str, Any]], gt: list[dict[str, Any]]) -> dict[str, Any]:
        if not pred and not gt:
            metric = _metric_dict(0, 0, 0, details={"total_predicted": 0, "total_ground_truth": 0, "matched_count": 0, "unmatched_predicted": [], "unmatched_ground_truth": [], "matches": []})
            return {"section": metric, "per_field": {"medicines": metric}}

        matched, unmatched_pred, unmatched_gt = match_medicines(pred, gt, self.t)

        tp_weight = sum(sim for _, _, sim in matched)
        fp_count = len(unmatched_pred)
        fn_count = len(unmatched_gt)

        details = {
            "total_predicted": len(pred),
            "total_ground_truth": len(gt),
            "matched_count": len(matched),
            "unmatched_predicted": [pred[index] for index in unmatched_pred],
            "unmatched_ground_truth": [gt[index] for index in unmatched_gt],
            "matches": [
                {"predicted": pred[pred_index], "ground_truth": gt[gt_index], "similarity": round(similarity, 4)}
                for pred_index, gt_index, similarity in matched
            ],
        }
        avg_similarity = sum(sim for _, _, sim in matched) / len(matched) if matched else 0.0

        precision = tp_weight / (tp_weight + fp_count) if (tp_weight + fp_count) > 0 else 0.0
        recall = tp_weight / (tp_weight + fn_count) if (tp_weight + fn_count) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        weighted_metric = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": round(tp_weight, 4),
            "fp": fp_count,
            "fn": fn_count,
            "avg_similarity": round(avg_similarity, 4),
            "details": details,
        }

        strict_matches: list[dict[str, Any]] = []
        strict_fp = 0
        strict_fn = 0
        strict_matched_pred: set[int] = set()
        strict_matched_gt: set[int] = set()
        for pred_index, gt_index, similarity in matched:
            pred_item = pred[pred_index]
            gt_item = gt[gt_index]
            qty_prox = 1.0 if pred_item.get("quantity") is None and gt_item.get("quantity") is None else 0.0
            if pred_item.get("quantity") is not None or gt_item.get("quantity") is not None:
                gt_qty = gt_item.get("quantity")
                pred_qty = pred_item.get("quantity")
                if pred_qty is not None and gt_qty is not None and float(gt_qty) != 0:
                    qty_prox = max(0.0, 1.0 - abs(float(pred_qty) - float(gt_qty)) / max(float(gt_qty), 1.0))
            pred_unit = (pred_item.get("unit") or "").strip().lower()
            gt_unit = (gt_item.get("unit") or "").strip().lower()
            unit_ok = not pred_unit or not gt_unit or pred_unit == gt_unit
            if similarity >= self.t.STRICT_MEDICINE_SIM_THRESHOLD and qty_prox >= self.t.STRICT_MEDICINE_QTY_THRESHOLD and unit_ok:
                strict_matches.append({"predicted": pred_item, "ground_truth": gt_item, "similarity": round(similarity, 4), "quantity_similarity": round(qty_prox, 4), "unit_match": unit_ok})
                strict_matched_pred.add(pred_index)
                strict_matched_gt.add(gt_index)
        strict_fp = len(pred) - len(strict_matched_pred)
        strict_fn = len(gt) - len(strict_matched_gt)
        strict_metric = _metric_dict(len(strict_matches), strict_fp, strict_fn)
        medicine_metric = dict(weighted_metric)
        medicine_metric.update(
            strict_f1=strict_metric["f1"],
            strict_precision=strict_metric["precision"],
            strict_recall=strict_metric["recall"],
            strict_tp=strict_metric["tp"],
            strict_fp=strict_metric["fp"],
            strict_fn=strict_metric["fn"],
            strict_matches=strict_matches,
        )
        return {"section": _merge_section_tracks(weighted_metric, strict_metric, avg_similarity=round(avg_similarity, 4), details=details, strict_matches=strict_matches), "per_field": {"medicines": medicine_metric}}

    def _evaluate_other_charges(self, pred: list[dict[str, Any]], gt: list[dict[str, Any]]) -> dict[str, Any]:
        if not gt:
            if not pred:
                metric = _metric_dict(0, 0, 0, details={"unmatched_predicted": [], "unmatched_ground_truth": []})
                return {"section": metric, "per_field": {"other_charges": metric}}
            metric = _metric_dict(0, len(pred), 0, details={"unmatched_predicted": pred, "unmatched_ground_truth": []})
            return {"section": metric, "per_field": {"other_charges": metric}}
        if not pred:
            metric = _metric_dict(0, 0, len(gt), details={"unmatched_predicted": [], "unmatched_ground_truth": gt})
            return {"section": metric, "per_field": {"other_charges": metric}}

        tp = fp = 0
        gt_matched = [False] * len(gt)
        unmatched_predicted: list[dict[str, Any]] = []
        matches: list[dict[str, Any]] = []

        for pred_charge in pred:
            pred_cat_raw = pred_charge.get("category") or ""
            pred_desc = pred_charge.get("description") or ""
            pred_cat_mapped = map_category_to_standard(pred_cat_raw) or map_description_to_category(pred_desc)
            best_idx = -1
            best_score = 0.0

            for index, gt_charge in enumerate(gt):
                if gt_matched[index]:
                    continue
                gt_cat_raw = gt_charge.get("category") or ""
                gt_desc = gt_charge.get("description") or ""
                gt_cat_mapped = map_category_to_standard(gt_cat_raw) or map_description_to_category(gt_desc)
                cat_match = pred_cat_mapped and gt_cat_mapped and pred_cat_mapped.upper() == gt_cat_mapped.upper()
                desc_score = _token_f1(_normalize_text(pred_desc), _normalize_text(gt_desc))
                combined = (1.0 if cat_match else 0.0) * 0.5 + desc_score * 0.5
                if combined > best_score:
                    best_score = combined
                    best_idx = index

            if best_idx >= 0 and best_score >= self.t.OTHER_CHARGE_SIM_THRESHOLD:
                tp += 1
                gt_matched[best_idx] = True
                matches.append(
                    {
                        "predicted": pred_charge,
                        "ground_truth": gt[best_idx],
                        "similarity": round(best_score, 4),
                    }
                )
            else:
                fp += 1
                unmatched_predicted.append(pred_charge)

        fn = sum(1 for matched in gt_matched if not matched)
        avg_similarity = sum(m["similarity"] for m in matches) / len(matches) if matches else 0.0
        metric = _metric_dict(
            tp,
            fp,
            fn,
            details={
                "matches": matches,
                "avg_similarity": round(avg_similarity, 4),
                "unmatched_predicted": unmatched_predicted,
                "unmatched_ground_truth": [gt[index] for index, matched in enumerate(gt_matched) if not matched],
            },
            avg_similarity=round(avg_similarity, 4),
        )
        strict_tp = strict_fp = 0
        strict_gt_matched = [False] * len(gt)
        strict_matches: list[dict[str, Any]] = []
        for pred_charge in pred:
            pred_cat_raw = pred_charge.get("category") or ""
            pred_desc = pred_charge.get("description") or ""
            pred_amount = pred_charge.get("amount")
            pred_cat_mapped = map_category_to_standard_strict(pred_cat_raw) or map_description_to_category_strict(pred_desc)
            best_idx = -1
            best_score = 0.0
            best_desc_score = 0.0
            for index, gt_charge in enumerate(gt):
                if strict_gt_matched[index]:
                    continue
                gt_cat_raw = gt_charge.get("category") or ""
                gt_desc = gt_charge.get("description") or ""
                gt_amount = gt_charge.get("amount")
                gt_cat_mapped = map_category_to_standard_strict(gt_cat_raw) or map_description_to_category_strict(gt_desc)
                desc_score = _token_f1(_normalize_text(pred_desc), _normalize_text(gt_desc))
                amount_match = _num_match(pred_amount, gt_amount)
                cat_match = pred_cat_mapped and gt_cat_mapped and pred_cat_mapped.upper() == gt_cat_mapped.upper()
                combined = (1.0 if cat_match else 0.0) * 0.5 + desc_score * 0.5
                strict_pass = bool(cat_match and amount_match and desc_score >= self.t.STRICT_OTHER_CHARGE_DESC_THRESHOLD)
                if strict_pass and combined > best_score:
                    best_score = combined
                    best_desc_score = desc_score
                    best_idx = index
            if best_idx >= 0:
                strict_tp += 1
                strict_gt_matched[best_idx] = True
                strict_matches.append({"predicted": pred_charge, "ground_truth": gt[best_idx], "similarity": round(best_score, 4), "description_similarity": round(best_desc_score, 4)})
            else:
                strict_fp += 1
        strict_fn = sum(1 for matched in strict_gt_matched if not matched)
        strict_metric = _metric_dict(strict_tp, strict_fp, strict_fn)
        other_metric = dict(metric)
        other_metric.update(
            strict_f1=strict_metric["f1"],
            strict_precision=strict_metric["precision"],
            strict_recall=strict_metric["recall"],
            strict_tp=strict_metric["tp"],
            strict_fp=strict_metric["fp"],
            strict_fn=strict_metric["fn"],
            strict_matches=strict_matches,
        )
        return {"section": _merge_section_tracks(metric, strict_metric, avg_similarity=round(avg_similarity, 4), details=metric.get("details", {}), strict_matches=strict_matches), "per_field": {"other_charges": other_metric}}

    def _evaluate_financial_recap(self, pred: dict[str, Any], gt: dict[str, Any]) -> dict[str, Any]:
        soft_metrics = self._evaluate_scalar_fields(
            "financial_recap",
            pred,
            gt,
            [
                ("total_medicines", "total_medicines"),
                ("total_other_charges", "total_other_charges"),
                ("grand_total", "grand_total"),
            ],
        )
        for metrics in soft_metrics["per_field"].values():
            metrics.update(
                strict_f1=metrics.get("f1", 0.0),
                strict_precision=metrics.get("precision", 0.0),
                strict_recall=metrics.get("recall", 0.0),
                strict_tp=metrics.get("tp", 0),
                strict_fp=metrics.get("fp", 0),
                strict_fn=metrics.get("fn", 0),
            )
        return {
            "section": _merge_section_tracks(soft_metrics["section"], soft_metrics["section"]),
            "per_field": soft_metrics["per_field"],
        }

    def _compute_edit_burden(self, section_results: dict[str, dict[str, Any]]) -> float:
        burden = 0.0
        summary_strict = section_results.get("summary", {}).get("strict", {})
        patient_strict = section_results.get("patient_info", {}).get("strict", {})
        financial_strict = section_results.get("financial_recap", {}).get("strict", {})
        medical_context_strict = section_results.get("medical_context", {}).get("strict", {})
        medicines = section_results.get("medicines", {})
        other_charges = section_results.get("other_charges", {})

        burden += float(summary_strict.get("fp", 0) + summary_strict.get("fn", 0)) * self.t.EDIT_COST_CRITICAL_FIELD
        burden += float(patient_strict.get("fp", 0) + patient_strict.get("fn", 0)) * self.t.EDIT_COST_CRITICAL_FIELD
        burden += float(financial_strict.get("fp", 0) + financial_strict.get("fn", 0)) * self.t.EDIT_COST_FINANCIAL_MISMATCH
        burden += float(medical_context_strict.get("fp", 0) + medical_context_strict.get("fn", 0)) * self.t.EDIT_COST_PARTIAL_MATCH
        burden += float(medicines.get("strict_fp", 0)) * self.t.EDIT_COST_EXTRA_ITEM
        burden += float(medicines.get("strict_fn", 0)) * self.t.EDIT_COST_MISSING_ITEM
        burden += max(0.0, float(medicines.get("soft_tp", 0)) - float(medicines.get("strict_tp", 0))) * self.t.EDIT_COST_PARTIAL_MATCH
        burden += float(other_charges.get("strict_fp", 0)) * self.t.EDIT_COST_EXTRA_ITEM
        burden += float(other_charges.get("strict_fn", 0)) * self.t.EDIT_COST_MISSING_ITEM
        burden += max(0.0, float(other_charges.get("soft_tp", 0)) - float(other_charges.get("strict_tp", 0))) * self.t.EDIT_COST_CATEGORY_FIX
        return burden

    def _collect_critical_fail_reasons(self, section_results: dict[str, dict[str, Any]], per_field: dict[str, dict[str, Any]]) -> list[str]:
        reasons: list[str] = []
        for field_path in (
            "summary.service_date",
            "summary.transaction_date",
            "summary.total_amount",
            "summary.currency",
            "summary.medical_facility_name",
            "patient_info.name",
            "financial_recap.total_medicines",
            "financial_recap.total_other_charges",
            "financial_recap.grand_total",
        ):
            field_metrics = per_field.get(field_path, {})
            if field_metrics.get("strict_f1", field_metrics.get("f1", 0.0)) < 1.0:
                reasons.append(f"{field_path} strict mismatch")
        if section_results.get("medicines", {}).get("strict_f1", 0.0) < 0.95:
            reasons.append("medicines strict_f1 below verify-only threshold")
        if section_results.get("other_charges", {}).get("strict_f1", 0.0) < 0.95:
            reasons.append("other_charges strict_f1 below verify-only threshold")
        if section_results.get("medical_context", {}).get("strict_f1", 0.0) < 0.90:
            reasons.append("medical_context strict_f1 below verify-only threshold")
        return reasons

    def _compute_verify_only_gate(self, section_results: dict[str, dict[str, Any]], edit_burden_score: float, critical_fail_reasons: list[str]) -> bool:
        if critical_fail_reasons:
            return False
        if edit_burden_score > self.t.VERIFY_ONLY_MAX_EDIT_BURDEN:
            return False
        return (
            section_results.get("medicines", {}).get("strict_f1", 0.0) >= 0.95
            and section_results.get("other_charges", {}).get("strict_f1", 0.0) >= 0.95
            and section_results.get("medical_context", {}).get("strict_f1", 0.0) >= 0.90
            and section_results.get("summary", {}).get("strict_f1", 0.0) >= 1.0
            and section_results.get("patient_info", {}).get("strict_f1", 0.0) >= 1.0
            and section_results.get("financial_recap", {}).get("strict_f1", 0.0) >= 1.0
        )

    def _compute_overall_f1(
        self,
        section_results: dict[str, dict[str, Any]],
        section_weights: dict[str, float],
    ) -> float:
        total_weight = 0.0
        weighted_f1 = 0.0
        for section, metrics in section_results.items():
            weight = section_weights.get(section, 1.0)
            weighted_f1 += weight * metrics["f1"]
            total_weight += weight
        return weighted_f1 / total_weight if total_weight > 0 else 0.0
