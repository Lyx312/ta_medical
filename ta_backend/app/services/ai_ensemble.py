from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
import math
import re
from typing import Any, Awaitable, Callable, cast

from app.schemas.ai import (
    AIMedicalScanResponse,
    AIScanDiagnosis,
    AIScanFinancialRecap,
    AIScanMedicalContext,
    AIScanMedicine,
    AIScanOtherCharge,
    AIScanPatientInfo,
    AIScanSummary,
)

FUSION_VERSION = "ensemble-fusion-v1"
CALIBRATION_VERSION = "heuristic-v1"
AI_EXTRACTION_MAX_PRODUCERS = 3
AI_EXTRACTION_RUNS_PER_MODEL = 2
AI_EXTRACTION_MIN_SUCCESSFUL_RUNS = 2
AI_EXTRACTION_TEMPERATURE = 0.2
CRITICAL_FIELD_THRESHOLD = 0.80
NON_CRITICAL_FIELD_THRESHOLD = 0.70
SEMANTIC_FIELD_THRESHOLD = 0.65
CRITICAL_FIELD_PATHS = {
    "summary.medical_facility_name",
    "summary.service_date",
    "patient_info.name",
    "financial_recap.grand_total",
}
SEMANTIC_FIELD_PATHS = {"medical_context.anamnesis", "medical_context.therapy_note"}
NUMERIC_FIELD_PATHS = {
    "summary.total_amount",
    "patient_info.age",
    "financial_recap.total_medicines",
    "financial_recap.total_other_charges",
    "financial_recap.grand_total",
    "medicines.quantity",
    "medicines.unit_price",
    "medicines.total_price",
    "other_charges.amount",
}
DATE_FIELD_SUFFIXES = {"service_date", "transaction_date", "date_of_birth"}
GENERIC_MEDICINE_KEYWORDS = ("obat", "medicine", "medication", "drug", "pharmacy", "apotek")
CONSULTATION_KEYWORDS = ("consult", "doctor", "dokter", "clinic", "klinik", "visit", "treatment", "therapy", "consultation")


def contains_keyword(text: str, keywords: tuple[str, ...]) -> bool:
    lowered_text = text.casefold()
    return any(keyword in lowered_text for keyword in keywords)


def classify_other_charge_kind(category: Any, description: Any) -> str:
    category_text = normalize_whitespace(str(category or "")).casefold()
    description_text = normalize_whitespace(str(description or "")).casefold()
    combined_text = f"{category_text} {description_text}".strip()

    if contains_keyword(combined_text, GENERIC_MEDICINE_KEYWORDS):
        return "medicine"
    if contains_keyword(combined_text, CONSULTATION_KEYWORDS):
        return "consultation"
    return "other"


def score_candidate_group(producer_ranks: dict[str, int], producer_ids: set[str], run_count: int) -> float:
    producer_weight = sum(max(0.6, 1.0 - (producer_ranks[producer_id] * 0.1)) for producer_id in producer_ids)
    run_bonus = max(0, run_count - len(producer_ids)) * 0.05
    return producer_weight + run_bonus


def choose_target_list_count(counts: list[int]) -> int:
    if not counts:
        return 0

    frequencies: dict[int, int] = {}
    for count in counts:
        frequencies[count] = frequencies.get(count, 0) + 1

    return sorted(frequencies.items(), key=lambda item: (-item[1], -item[0]))[0][0]


@dataclass(frozen=True)
class ProducerInferenceTrace:
    producer_id: str
    provider: str
    model: str
    producer_rank: int
    run_index: int
    temperature: float
    success: bool
    payload: AIMedicalScanResponse | None = None
    error: str | None = None


@dataclass(frozen=True)
class SuccessfulProducerInferenceTrace:
    producer_id: str
    provider: str
    model: str
    producer_rank: int
    run_index: int
    temperature: float
    success: bool
    payload: AIMedicalScanResponse
    error: str | None = None


def to_successful_trace(trace: ProducerInferenceTrace) -> SuccessfulProducerInferenceTrace:
    if trace.payload is None:
        raise ValueError("Cannot convert a producer trace without payload into a successful trace.")
    return SuccessfulProducerInferenceTrace(
        producer_id=trace.producer_id,
        provider=trace.provider,
        model=trace.model,
        producer_rank=trace.producer_rank,
        run_index=trace.run_index,
        temperature=trace.temperature,
        success=trace.success,
        payload=trace.payload,
        error=trace.error,
    )


@dataclass(frozen=True)
class MedicalExtractionOutcome:
    result: AIMedicalScanResponse
    overall_confidence: float | None
    field_confidence: dict[str, Any]
    review_required: bool
    review_reasons: list[str]
    calibration_version: str
    ai_execution_mode: str
    primary_model_provider: str | None
    primary_model_name: str | None
    ai_runs: list[ProducerInferenceTrace]
    extraction_meta_payload: dict[str, Any]


JudgeFieldFn = Callable[[str, list[Any], list[dict[str, Any]]], Awaitable[tuple[str | None, str | None]]]


def clamp_confidence(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 4)


def normalize_whitespace(value: str) -> str:
    return " ".join(value.strip().split())


def normalize_comparison_text(value: str) -> str:
    return re.sub(r"[\W_]+", "", normalize_whitespace(value).casefold())


def slugify_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-") or "item"


def coerce_numeric_value(value: Any) -> float | int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            return None
        return int(value) if isinstance(value, float) and value.is_integer() else value
    if not isinstance(value, str):
        return None

    cleaned = value.strip().replace(" ", "").replace("IDR", "").replace("idr", "")
    if "," in cleaned and "." in cleaned:
        cleaned = cleaned.replace(".", "").replace(",", ".") if cleaned.rfind(",") > cleaned.rfind(".") else cleaned.replace(",", "")
    elif cleaned.count(",") > 1:
        cleaned = cleaned.replace(",", "")
    elif cleaned.count(".") > 1:
        cleaned = cleaned.replace(".", "")
    else:
        cleaned = cleaned.replace(",", "")
    cleaned = re.sub(r"[^0-9.\-]", "", cleaned)
    if not cleaned or cleaned in {"-", ".", "-."}:
        return None

    parsed = float(cleaned)
    if not math.isfinite(parsed):
        return None
    return int(parsed) if parsed.is_integer() else round(parsed, 2)


def coerce_date_value(value: Any) -> str | None:
    if value is None or value == "" or not isinstance(value, str):
        return None
    candidate = value.strip()
    if not candidate:
        return None

    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d", "%d.%m.%Y", "%d %m %Y"):
        try:
            return datetime.strptime(candidate, fmt).date().isoformat()
        except ValueError:
            continue

    return candidate


def normalize_candidate_key(field_path: str, value: Any) -> str:
    if value is None:
        return "__null__"
    leaf_name = field_path.rsplit(".", 1)[-1]
    if leaf_name in DATE_FIELD_SUFFIXES:
        return coerce_date_value(value) or "__null__"
    if field_path in NUMERIC_FIELD_PATHS:
        numeric_value = coerce_numeric_value(value)
        return "__null__" if numeric_value is None else str(numeric_value)
    if field_path.endswith("currency") or field_path.endswith("gender"):
        return normalize_whitespace(str(value)).upper()
    if field_path in SEMANTIC_FIELD_PATHS:
        return normalize_whitespace(str(value)).casefold()
    return normalize_comparison_text(str(value))


def format_output_value(field_path: str, value: Any) -> Any:
    if value is None:
        return None
    leaf_name = field_path.rsplit(".", 1)[-1]
    if leaf_name in DATE_FIELD_SUFFIXES:
        return coerce_date_value(value)
    if field_path in NUMERIC_FIELD_PATHS:
        return coerce_numeric_value(value)
    if field_path.endswith("currency") or field_path.endswith("gender"):
        return normalize_whitespace(str(value)).upper()
    return normalize_whitespace(value) if isinstance(value, str) else value


def serialize_supported_runs(traces: list[ProducerInferenceTrace]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for trace in traces:
        entry = grouped.setdefault(
            trace.producer_id,
            {
                "producer_id": trace.producer_id,
                "provider": trace.provider,
                "model": trace.model,
                "producer_rank": trace.producer_rank,
                "run_indices": [],
            },
        )
        entry["run_indices"].append(trace.run_index)
    return sorted(grouped.values(), key=lambda item: (item["producer_rank"], item["producer_id"]))


def serialize_field_candidates(candidates: list[dict[str, Any]], *, selected_key: str) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_traces = candidate.get("traces", [])
        traces = cast(list[ProducerInferenceTrace], candidate_traces)
        producer_ids = candidate.get("producers", set())
        producer_count = len(producer_ids) if isinstance(producer_ids, set) else int(candidate.get("producer_count", 0))
        serialized.append(
            {
                "normalized_key": candidate.get("normalized_key"),
                "value": candidate.get("raw_value"),
                "selected": candidate.get("normalized_key") == selected_key,
                "support_count": candidate.get("run_count", 0),
                "producer_count": producer_count,
                "support": serialize_supported_runs(traces),
            }
        )
    return serialized


class MedicalExtractionFusionEngine:
    def __init__(self, *, judge_field: JudgeFieldFn) -> None:
        self._judge_field = judge_field

    async def fuse(
        self,
        *,
        successful_traces: list[SuccessfulProducerInferenceTrace],
        all_traces: list[ProducerInferenceTrace],
        images: list[Any],
        execution_mode: str,
        primary_model_provider: str | None,
        primary_model_name: str | None,
    ) -> MedicalExtractionOutcome:
        field_confidence: dict[str, Any] = {}
        judge_invocations: list[dict[str, Any]] = []

        summary = await self._fuse_summary(successful_traces, images, field_confidence, judge_invocations)
        patient_info = await self._fuse_section_object(
            section_name="patient_info",
            section_source=[trace.payload.patient_info for trace in successful_traces],
            fields=("name", "date_of_birth", "age", "gender", "address"),
            section_model=AIScanPatientInfo,
            traces=successful_traces,
            images=images,
            field_confidence=field_confidence,
            judge_invocations=judge_invocations,
        )
        medical_context = await self._fuse_medical_context(successful_traces, images, field_confidence, judge_invocations)
        financial_recap = await self._fuse_section_object(
            section_name="financial_recap",
            section_source=[trace.payload.financial_recap for trace in successful_traces],
            fields=("total_medicines", "total_other_charges", "grand_total"),
            section_model=AIScanFinancialRecap,
            traces=successful_traces,
            images=images,
            field_confidence=field_confidence,
            judge_invocations=judge_invocations,
        )
        medicines = await self._fuse_list_items(
            successful_traces,
            images,
            "medicines",
            field_confidence,
            judge_invocations,
        )
        medicine_amount_keys = {
            normalize_candidate_key("medicines.total_price", medicine.total_price)
            for medicine in medicines
            if medicine.total_price is not None
        }
        other_charges = await self._fuse_list_items(
            successful_traces,
            images,
            "other_charges",
            field_confidence,
            judge_invocations,
            exclude_amount_keys=medicine_amount_keys,
        )

        result = AIMedicalScanResponse(
            summary=summary,
            patient_info=patient_info,
            medical_context=medical_context,
            medicines=medicines,
            other_charges=other_charges,
            financial_recap=financial_recap,
        )
        constraint_checks = self._apply_financial_constraints(result, field_confidence)
        review_reasons = self._build_review_reasons(field_confidence, constraint_checks)
        review_required = bool(review_reasons)
        overall_confidence = self._compute_overall_confidence(field_confidence)

        extraction_meta_payload = {
            "fusion_version": FUSION_VERSION,
            "calibration_version": CALIBRATION_VERSION,
            "field_confidence": field_confidence,
            "constraint_checks": constraint_checks,
            "judge_invocations": judge_invocations,
            "review_required": review_required,
            "review_reasons": review_reasons,
            "overall_confidence": overall_confidence,
        }
        return MedicalExtractionOutcome(
            result=result,
            overall_confidence=overall_confidence,
            field_confidence=field_confidence,
            review_required=review_required,
            review_reasons=review_reasons,
            calibration_version=CALIBRATION_VERSION,
            ai_execution_mode=execution_mode,
            primary_model_provider=primary_model_provider,
            primary_model_name=primary_model_name,
            ai_runs=all_traces,
            extraction_meta_payload=extraction_meta_payload,
        )

    async def _fuse_summary(self, traces: list[SuccessfulProducerInferenceTrace], images: list[Any], field_confidence: dict[str, Any], judge_invocations: list[dict[str, Any]]) -> AIScanSummary:
        detected_documents = self._fuse_detected_documents(traces)
        field_confidence["summary.detected_documents"] = {"value": detected_documents, "confidence": 1.0 if detected_documents else 0.0, "support": [], "alternates": [], "judge_used": False, "rule_hits": [], "review_flags": [], "calibration_version": CALIBRATION_VERSION}
        medical_facility_name = await self._fuse_scalar("summary.medical_facility_name", [(trace, trace.payload.summary.medical_facility_name) for trace in traces], images, field_confidence, judge_invocations)
        service_date = await self._fuse_scalar("summary.service_date", [(trace, trace.payload.summary.service_date or trace.payload.summary.transaction_date) for trace in traces], images, field_confidence, judge_invocations)
        currency = await self._fuse_scalar("summary.currency", [(trace, trace.payload.summary.currency) for trace in traces], images, field_confidence, judge_invocations)
        total_amount = await self._fuse_scalar("summary.total_amount", [(trace, trace.payload.summary.total_amount) for trace in traces], images, field_confidence, judge_invocations)
        field_confidence["summary.transaction_date"] = {**field_confidence["summary.service_date"], "rule_hits": [*field_confidence["summary.service_date"].get("rule_hits", []), "mirrors summary.service_date for legacy compatibility"]}
        return AIScanSummary(detected_documents=detected_documents, medical_facility_name=medical_facility_name, service_date=service_date, transaction_date=service_date, currency=currency, total_amount=total_amount)

    async def _fuse_section_object(self, *, section_name: str, section_source: list[Any], fields: tuple[str, ...], section_model: Any, traces: list[SuccessfulProducerInferenceTrace], images: list[Any], field_confidence: dict[str, Any], judge_invocations: list[dict[str, Any]]) -> Any:
        fused: dict[str, Any] = {}
        for field_name in fields:
            fused[field_name] = await self._fuse_scalar(
                f"{section_name}.{field_name}",
                [(trace, getattr(source, field_name)) for trace, source in zip(traces, section_source)],
                images,
                field_confidence,
                judge_invocations,
            )
        return section_model(**fused)

    async def _fuse_medical_context(self, traces: list[SuccessfulProducerInferenceTrace], images: list[Any], field_confidence: dict[str, Any], judge_invocations: list[dict[str, Any]]) -> AIScanMedicalContext:
        anamnesis = await self._fuse_scalar("medical_context.anamnesis", [(trace, trace.payload.medical_context.anamnesis) for trace in traces], images, field_confidence, judge_invocations)
        therapy_note = await self._fuse_scalar("medical_context.therapy_note", [(trace, trace.payload.medical_context.therapy_note) for trace in traces], images, field_confidence, judge_invocations)
        diagnoses = await self._fuse_list_items(traces, images, "diagnoses", field_confidence, judge_invocations)
        return AIScanMedicalContext(anamnesis=anamnesis, diagnoses=diagnoses, therapy_note=therapy_note)

    async def _fuse_list_items(self, traces: list[SuccessfulProducerInferenceTrace], images: list[Any], list_name: str, field_confidence: dict[str, Any], judge_invocations: list[dict[str, Any]], *, exclude_amount_keys: set[str] | None = None) -> list[Any]:
        item_groups: dict[str, dict[str, Any]] = {}
        source_getter = {
            "diagnoses": lambda payload: payload.medical_context.diagnoses,
            "medicines": lambda payload: payload.medicines,
            "other_charges": lambda payload: payload.other_charges,
        }[list_name]
        item_model = {"diagnoses": AIScanDiagnosis, "medicines": AIScanMedicine, "other_charges": AIScanOtherCharge}[list_name]
        item_fields = {
            "diagnoses": ("description", "doctor_name"),
            "medicines": ("name", "quantity", "unit", "dosage_instruction", "unit_price", "total_price"),
            "other_charges": ("category", "description", "amount"),
        }[list_name]
        item_counts = [len(source_getter(trace.payload)) for trace in traces]
        target_count = choose_target_list_count(item_counts)

        for trace in traces:
            for item in source_getter(trace.payload):
                item_key = self._build_list_item_key(list_name, item)
                if item_key == "__null__":
                    continue
                similar_key = self._find_similar_group_key(list_name, item_key, list(item_groups))
                resolved_key = similar_key or item_key
                group = item_groups.setdefault(
                    resolved_key,
                    {
                        "key": resolved_key,
                        "items": [],
                        "producers": set(),
                        "producer_ranks": {},
                    },
                )
                group["items"].append((trace, item))
                group["producers"].add(trace.producer_id)
                group["producer_ranks"][trace.producer_id] = trace.producer_rank

        ranked_groups: list[dict[str, Any]] = []
        for group in item_groups.values():
            run_count = len(group["items"])
            score = score_candidate_group(group["producer_ranks"], group["producers"], run_count)
            ranked_groups.append(
                {
                    **group,
                    "run_count": run_count,
                    "score": score,
                }
            )

        ranked_groups.sort(
            key=lambda group: (
                -group["score"],
                -len(group["producers"]),
                -group["run_count"],
                group["key"],
            )
        )

        if list_name == "other_charges" and exclude_amount_keys:
            ranked_groups = [
                group
                for group in ranked_groups
                if not self._should_exclude_other_charge_group(group["key"], exclude_amount_keys)
            ]

        if target_count <= 0 or not ranked_groups:
            return []

        stable_groups = [
            group
            for group in ranked_groups
            if group["run_count"] > 1 or len(group["producers"]) > 1
        ]
        selection_pool = stable_groups if stable_groups else ranked_groups
        selected_groups = selection_pool[:target_count]

        fused_items: list[Any] = []
        for group in selected_groups:
            item_key = group["key"]
            grouped_items = group["items"]
            fused_fields: dict[str, Any] = {}
            path_prefix = f"{list_name}[{slugify_key(item_key)}]"
            for field_name in item_fields:
                fused_fields[field_name] = await self._fuse_scalar(
                    f"{list_name}.{field_name}",
                    [(trace, getattr(item, field_name)) for trace, item in grouped_items],
                    images,
                    field_confidence,
                    judge_invocations,
                    output_path=f"{path_prefix}.{field_name}",
                )
            fused_items.append(item_model(**fused_fields))
        return fused_items

    async def _fuse_scalar(self, field_path: str, values: list[tuple[SuccessfulProducerInferenceTrace, Any]], images: list[Any], field_confidence: dict[str, Any], judge_invocations: list[dict[str, Any]], *, output_path: str | None = None) -> Any:
        total_runs = len(values)
        total_producers = len({trace.producer_id for trace, _ in values})
        groups: dict[str, dict[str, Any]] = {}
        for trace, value in values:
            normalized_key = normalize_candidate_key(field_path, value)
            group = groups.setdefault(normalized_key, {"normalized_key": normalized_key, "raw_value": format_output_value(field_path, value), "traces": [], "producers": set(), "producer_ranks": {}})
            if group["raw_value"] is None and value is not None:
                group["raw_value"] = format_output_value(field_path, value)
            group["traces"].append(trace)
            group["producers"].add(trace.producer_id)
            group["producer_ranks"][trace.producer_id] = trace.producer_rank

        ranked = []
        for group in groups.values():
            ranked.append(
                {
                    **group,
                    "score": score_candidate_group(group["producer_ranks"], group["producers"], len(group["traces"])),
                    "run_count": len(group["traces"]),
                }
            )
        ranked.sort(key=lambda item: (-item["score"], -len(item["producers"]), -item["run_count"], item["raw_value"] is None, str(item["raw_value"])))
        winner = ranked[0] if ranked else {"raw_value": None, "traces": [], "producers": set(), "score": 0.0, "run_count": 0, "normalized_key": "__null__"}
        runner_up = ranked[1] if len(ranked) > 1 else None

        judge_used = False
        judge_reason: str | None = None
        close_runner_up = False
        if runner_up is not None and winner["score"] > 0:
            close_runner_up = ((winner["score"] - runner_up["score"]) / winner["score"]) < 0.18

        if len(ranked) > 1 and close_runner_up:
            selected_key, judge_reason = await self._judge_field(field_path, images, ranked[:3])
            if selected_key is not None:
                matching = next((candidate for candidate in ranked if candidate["normalized_key"] == selected_key), None)
                if matching is not None:
                    winner = matching
                    judge_used = True
                    judge_invocations.append({"field_path": output_path or field_path, "selected_candidate": winner["raw_value"], "reason": judge_reason})

        producer_support_ratio = len(winner["producers"]) / max(total_producers, 1)
        run_support_ratio = winner["run_count"] / max(total_runs, 1)
        margin_ratio = 1.0 if runner_up is None or winner["score"] <= 0 else max((winner["score"] - runner_up["score"]) / winner["score"], 0.0)
        raw_confidence = clamp_confidence((0.45 * producer_support_ratio) + (0.35 * run_support_ratio) + (0.20 * margin_ratio))
        if judge_used and winner["raw_value"] is not None:
            raw_confidence = max(raw_confidence, 0.72 if field_path in CRITICAL_FIELD_PATHS else 0.68)
        confidence = self._calibrate_confidence(field_path, raw_confidence)
        review_flags = ["low_confidence"] if confidence < self._threshold_for(field_path) else []
        if judge_used:
            review_flags.append("judge_resolved")

        alternate_candidates = [
            candidate for candidate in ranked if candidate["normalized_key"] != winner["normalized_key"]
        ]

        field_confidence[output_path or field_path] = {
            "value": winner["raw_value"],
            "confidence": confidence,
            "raw_confidence": raw_confidence,
            "support": serialize_supported_runs(winner["traces"]),
            "alternates": [
                {
                    "value": candidate["raw_value"],
                    "support_count": candidate["run_count"],
                    "producer_count": len(candidate["producers"]),
                }
                for candidate in alternate_candidates[:3]
            ],
            "candidates": serialize_field_candidates(ranked, selected_key=winner["normalized_key"]),
            "judge_used": judge_used,
            "judge_reason": judge_reason,
            "rule_hits": [],
            "review_flags": review_flags,
            "calibration_version": CALIBRATION_VERSION,
            "support_count": winner["run_count"],
            "producer_agreement": clamp_confidence(producer_support_ratio),
        }
        return winner["raw_value"]

    def _apply_financial_constraints(self, result: AIMedicalScanResponse, field_confidence: dict[str, Any]) -> dict[str, Any]:
        medicines_sum = self._sum_defined_numbers([medicine.total_price for medicine in result.medicines])
        other_charges_sum = self._sum_defined_numbers([charge.amount for charge in result.other_charges])
        recap_total = self._sum_defined_numbers([medicines_sum, other_charges_sum])
        return {
            "line_item.total_medicines": self._apply_numeric_constraint(result, "financial_recap.total_medicines", medicines_sum, field_confidence),
            "line_item.total_other_charges": self._apply_numeric_constraint(result, "financial_recap.total_other_charges", other_charges_sum, field_confidence),
            "financial_recap.grand_total": self._apply_numeric_constraint(result, "financial_recap.grand_total", recap_total, field_confidence),
            "summary.total_amount": self._apply_numeric_constraint(result, "summary.total_amount", result.financial_recap.grand_total, field_confidence),
        }

    def _apply_numeric_constraint(self, result: AIMedicalScanResponse, field_path: str, derived_value: float | int | None, field_confidence: dict[str, Any]) -> dict[str, Any]:
        current_value = self._get_field_value(result, field_path)
        passed = derived_value is not None and current_value is not None and abs(float(current_value) - float(derived_value)) < 0.01
        used_derived_value = derived_value is not None and current_value is None
        if used_derived_value:
            self._set_field_value(result, field_path, derived_value)
            meta = field_confidence.get(field_path)
            if isinstance(meta, dict):
                meta["value"] = derived_value
                meta["confidence"] = max(float(meta.get("confidence", 0.0)), 0.83)
                meta["rule_hits"] = [*meta.get("rule_hits", []), "constraint backfilled missing total"]
                if "constraint_adjusted" not in meta["review_flags"]:
                    meta["review_flags"].append("constraint_adjusted")
        return {"passed": passed, "derived_value": derived_value, "original_value": current_value, "final_value": self._get_field_value(result, field_path), "used_derived_value": used_derived_value}

    def _sum_defined_numbers(self, values: list[float | int | None]) -> float | int | None:
        parsed = [coerce_numeric_value(value) for value in values if value is not None]
        if not parsed:
            return None
        total = 0.0
        for value in parsed:
            if value is not None:
                total += float(value)
        return int(total) if total.is_integer() else round(total, 2)

    def _build_review_reasons(self, field_confidence: dict[str, Any], constraint_checks: dict[str, Any]) -> list[str]:
        reasons = [f"{field_path} confidence is below the review threshold." for field_path, meta in field_confidence.items() if isinstance(meta, dict) and float(meta.get("confidence", 0.0)) < self._threshold_for(field_path)]
        reasons.extend(f"{name} did not match derived financial totals." for name, check in constraint_checks.items() if not bool(check.get("passed", False)))
        return list(dict.fromkeys(reasons))

    def _compute_overall_confidence(self, field_confidence: dict[str, Any]) -> float | None:
        weighted = [(float(meta["confidence"]), 2.0 if field_path in CRITICAL_FIELD_PATHS else 1.0) for field_path, meta in field_confidence.items() if isinstance(meta, dict) and "confidence" in meta]
        if not weighted:
            return None
        return clamp_confidence(sum(score * weight for score, weight in weighted) / sum(weight for _, weight in weighted))

    def _calibrate_confidence(self, field_path: str, raw_confidence: float) -> float:
        adjusted = raw_confidence + (0.03 if field_path in CRITICAL_FIELD_PATHS else 0.0) - (0.04 if field_path in SEMANTIC_FIELD_PATHS else 0.0)
        return clamp_confidence(adjusted)

    def _threshold_for(self, field_path: str) -> float:
        if field_path in CRITICAL_FIELD_PATHS:
            return CRITICAL_FIELD_THRESHOLD
        if field_path in SEMANTIC_FIELD_PATHS:
            return SEMANTIC_FIELD_THRESHOLD
        return NON_CRITICAL_FIELD_THRESHOLD

    def _get_field_value(self, result: AIMedicalScanResponse, field_path: str) -> Any:
        section_name, field_name = field_path.split(".", 1)
        return getattr(getattr(result, section_name), field_name)

    def _set_field_value(self, result: AIMedicalScanResponse, field_path: str, value: Any) -> None:
        section_name, field_name = field_path.split(".", 1)
        setattr(getattr(result, section_name), field_name, value)

    def _fuse_detected_documents(self, traces: list[SuccessfulProducerInferenceTrace]) -> list[str]:
        target_count = choose_target_list_count([len(trace.payload.summary.detected_documents) for trace in traces])
        if target_count <= 0:
            return []

        groups: dict[str, dict[str, Any]] = {}
        for trace in traces:
            for document_label in trace.payload.summary.detected_documents:
                normalized_key = normalize_candidate_key("summary.detected_documents", document_label)
                group = groups.setdefault(
                    normalized_key,
                    {
                        "raw_value": normalize_whitespace(document_label),
                        "producers": set(),
                        "producer_ranks": {},
                        "run_count": 0,
                    },
                )
                group["producers"].add(trace.producer_id)
                group["producer_ranks"][trace.producer_id] = trace.producer_rank
                group["run_count"] += 1

        ranked = sorted(
            groups.values(),
            key=lambda group: (
                -score_candidate_group(group["producer_ranks"], group["producers"], group["run_count"]),
                -len(group["producers"]),
                -group["run_count"],
                group["raw_value"],
            ),
        )
        return [group["raw_value"] for group in ranked[:target_count]]

    def _build_list_item_key(self, list_name: str, item: Any) -> str:
        if list_name == "diagnoses":
            return normalize_candidate_key("diagnoses.description", getattr(item, "description", None))

        if list_name == "medicines":
            name = getattr(item, "name", None)
            return normalize_candidate_key("medicines.name", name)

        amount = coerce_numeric_value(getattr(item, "amount", None))
        if amount is None:
            return normalize_candidate_key("other_charges.description", getattr(item, "description", None))

        charge_kind = classify_other_charge_kind(getattr(item, "category", None), getattr(item, "description", None))
        return f"{charge_kind}:{amount}"

    def _find_similar_group_key(self, list_name: str, candidate_key: str, existing_keys: list[str]) -> str | None:
        if list_name != "diagnoses":
            return None

        for existing_key in existing_keys:
            if SequenceMatcher(None, candidate_key, existing_key).ratio() >= 0.9:
                return existing_key
        return None

    def _should_exclude_other_charge_group(self, group_key: str, exclude_amount_keys: set[str]) -> bool:
        if not group_key.startswith("medicine:"):
            return False

        _, amount_text = group_key.split(":", 1)
        return amount_text in exclude_amount_keys
