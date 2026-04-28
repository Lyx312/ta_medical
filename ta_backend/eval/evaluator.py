import re
from typing import Any

from app.services.standard_vocabulary import (
    get_document_type_display_name,
    get_category_display_name,
    map_category_to_standard,
    map_description_to_category,
    normalize_facility_name,
    normalize_to_standard_document_type,
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

        overall_f1 = self._compute_overall_f1(section_results, section_weights)
        total_tp = sum(metrics.get("tp", 0) for metrics in section_results.values())
        total_fp = sum(metrics.get("fp", 0) for metrics in section_results.values())
        total_fn = sum(metrics.get("fn", 0) for metrics in section_results.values())
        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        per_section = {
            section: {
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "tp": metrics["tp"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
            }
            for section, metrics in section_results.items()
        }

        return {
            "overall_f1": round(overall_f1, 4),
            "overall_precision": round(overall_precision, 4),
            "overall_recall": round(overall_recall, 4),
            "per_section": per_section,
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

        return {
            "section": _metric_dict(tp, fp, fn),
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

    def _evaluate_patient_info(self, pred: dict[str, Any], gt: dict[str, Any]) -> dict[str, Any]:
        return self._evaluate_scalar_fields(
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

        return {
            "section": _metric_dict(
                total_tp,
                total_fp,
                total_fn,
                anamnesis_f1=round(anamnesis_f1, 4),
                anamnesis_cer=anamnesis_cer,
                therapy_note_f1=round(therapy_f1, 4),
                therapy_note_cer=therapy_cer,
                diagnoses_f1=round(diag_metrics["f1"], 4),
            ),
            "per_field": {
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
            },
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
        return {"section": weighted_metric, "per_field": {"medicines": weighted_metric}}

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
        return {"section": metric, "per_field": {"other_charges": metric}}

    def _evaluate_financial_recap(self, pred: dict[str, Any], gt: dict[str, Any]) -> dict[str, Any]:
        return self._evaluate_scalar_fields(
            "financial_recap",
            pred,
            gt,
            [
                ("total_medicines", "total_medicines"),
                ("total_other_charges", "total_other_charges"),
                ("grand_total", "grand_total"),
            ],
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
