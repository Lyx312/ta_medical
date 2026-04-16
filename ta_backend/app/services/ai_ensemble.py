from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from functools import lru_cache
from itertools import combinations
import math
import re
from typing import Any, Awaitable, Callable, Collection, cast

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
from app.services.ai_ensemble_config import (
    AI_EXTRACTION_MAX_PRODUCERS,
    AI_EXTRACTION_MIN_SUCCESSFUL_RUNS,
    AI_EXTRACTION_RUNS_PER_MODEL,
    AI_EXTRACTION_TEMPERATURE,
    CALIBRATION_VERSION,
    CONSULTATION_KEYWORDS,
    CRITICAL_FIELD_PATHS,
    CRITICAL_FIELD_THRESHOLD,
    DATE_FIELD_SUFFIXES,
    FINANCIAL_FIELD_PATHS,
    FUSION_VERSION,
    GENERIC_MEDICINE_KEYWORDS,
    GENDER_AMBIGUITY_MARGIN_THRESHOLD,
    GENDER_NULL_SUPPORT_RATIO_THRESHOLD,
    HANDWRITTEN_DIVERGENCE_THRESHOLD,
    MEDICINE_PRIMARY_COUNT_BUFFER,
    MEDICINE_PRIMARY_COUNT_SOFT_CAP,
    MEDICINE_SPLIT_COMPONENT_MIN,
    NON_CRITICAL_FIELD_THRESHOLD,
    NUMERIC_FIELD_PATHS,
    OTHER_CHARGE_DESCRIPTOR_STOPWORDS,
    OTHER_CHARGE_BUCKET_ORDER,
    PACKAGE_LEVEL_MEDICINE_KEYWORDS,
    RECONCILIATION_HEAVY_DROP_COUNT_THRESHOLD,
    RECONCILIATION_LARGE_OVERRIDE_RATIO_THRESHOLD,
    RECONCILIATION_VERSION,
    RECONCILIATION_WEAK_SOURCE_SUPPORT_THRESHOLD,
    SEMANTIC_FIELD_PATHS,
    SEMANTIC_FIELD_SUFFIXES,
    SEMANTIC_FIELD_THRESHOLD,
    SERVICE_KEYWORDS,
    STRUCTURED_TERM_LEXICON,
)


COMPOUND_MEDICINE_SPLIT_PATTERN = re.compile(r",|\+|/|&|\bdan\b", re.IGNORECASE)
ALPHANUMERIC_TOKEN_SPLIT_PATTERN = re.compile(r"[^a-z0-9]+")
LIST_ITEM_FIELD_PATH_PATTERN = re.compile(r"^(diagnoses|medicines|other_charges)\[([^\]]+)\]\.([a-z_]+)$")
FINANCIAL_TOTAL_FIELD_SEQUENCE = (
    "summary.total_amount",
    "financial_recap.total_medicines",
    "financial_recap.total_other_charges",
    "financial_recap.grand_total",
)


def contains_keyword(text: str, keywords: Collection[str]) -> bool:
    lowered_text = text.casefold()
    return any(keyword in lowered_text for keyword in keywords)


def sequence_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, left, right).ratio()


def is_critical_field_path(field_path: str) -> bool:
    return field_path in CRITICAL_FIELD_PATHS


def is_semantic_field_path(field_path: str) -> bool:
    return field_path in SEMANTIC_FIELD_PATHS or field_path.endswith(SEMANTIC_FIELD_SUFFIXES)


def is_financial_field_path(field_path: str) -> bool:
    return field_path in FINANCIAL_FIELD_PATHS


def is_generic_medicine_key(value: str) -> bool:
    lowered_value = value.casefold()
    return lowered_value in {
        "obat",
        "obatan",
        "obatobatan",
        "medicine",
        "medication",
        "drug",
        "pharmacy",
        "apotek",
    }


def is_package_level_medicine_name(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    normalized_value = normalize_whitespace(value)
    if not normalized_value:
        return False
    lowered_value = normalized_value.casefold()
    normalized_key = normalize_comparison_text(normalized_value)
    return (
        contains_keyword(lowered_value, PACKAGE_LEVEL_MEDICINE_KEYWORDS)
        or is_generic_medicine_key(normalized_key)
    )


def extract_compound_medicine_components(value: Any) -> list[str]:
    if not isinstance(value, str):
        return []
    normalized_value = normalize_whitespace(value)
    if not normalized_value:
        return []
    parts = COMPOUND_MEDICINE_SPLIT_PATTERN.split(normalized_value)
    components: list[str] = []
    seen: set[str] = set()
    for part in parts:
        normalized_part = normalize_candidate_key("medicines.name", part)
        if normalized_part in {"__null__", ""} or is_generic_medicine_key(normalized_part):
            continue
        if len(normalized_part) < 4 or normalized_part in seen:
            continue
        seen.add(normalized_part)
        components.append(normalized_part)
    return components


def normalize_other_charge_bucket(raw_kind: str) -> str:
    return raw_kind if raw_kind in {"medicine", "consultation", "service"} else "other"


def classify_other_charge_kind(category: Any, description: Any) -> str:
    category_text = normalize_whitespace(str(category or "")).casefold()
    description_text = normalize_whitespace(str(description or "")).casefold()
    combined_text = f"{category_text} {description_text}".strip()

    has_medicine_keyword = contains_keyword(combined_text, GENERIC_MEDICINE_KEYWORDS)
    has_consultation_keyword = contains_keyword(combined_text, CONSULTATION_KEYWORDS)
    has_service_keyword = contains_keyword(combined_text, SERVICE_KEYWORDS)

    if has_medicine_keyword and (has_consultation_keyword or has_service_keyword):
        return "mixed"
    if has_medicine_keyword:
        return "medicine"
    if has_service_keyword:
        return "service"
    if has_consultation_keyword:
        return "consultation"
    return "other"


def extract_other_charge_descriptor_signature(category: Any, description: Any) -> str:
    raw_parts = [normalize_whitespace(str(description or "")).casefold(), normalize_whitespace(str(category or "")).casefold()]
    tokens: list[str] = []
    seen_tokens: set[str] = set()
    for part in raw_parts:
        for token in ALPHANUMERIC_TOKEN_SPLIT_PATTERN.split(part):
            normalized_token = normalize_comparison_text(token)
            if (
                not normalized_token
                or normalized_token in OTHER_CHARGE_DESCRIPTOR_STOPWORDS
                or is_generic_medicine_key(normalized_token)
            ):
                continue
            if normalized_token in seen_tokens:
                continue
            seen_tokens.add(normalized_token)
            tokens.append(normalized_token)

    if tokens:
        return "".join(tokens)

    description_key = normalize_candidate_key("other_charges.description", description)
    if description_key != "__null__":
        return description_key

    category_key = normalize_candidate_key("other_charges.category", category)
    return category_key if category_key != "__null__" else "other"


def parse_other_charge_group_key(group_key: str) -> tuple[str, str | None, str]:
    raw_kind, amount_text, descriptor_key = (group_key.split(":", 2) + ["", ""])[:3]
    return raw_kind or "other", None if amount_text in {"", "__null__"} else amount_text, descriptor_key


def other_charge_kinds_share_family(left_kind: str, right_kind: str) -> bool:
    if left_kind == right_kind:
        return True
    return {left_kind, right_kind}.issubset({"consultation", "service"})


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


@dataclass
class FusionWorkingState:
    field_confidence: dict[str, Any] = field(default_factory=dict)
    judge_invocations: list[dict[str, Any]] = field(default_factory=list)
    list_group_meta: dict[str, Any] = field(default_factory=dict)
    inventory_conflicts: list[str] = field(default_factory=list)
    normalized_terms: list[dict[str, Any]] = field(default_factory=list)
    nullified_fields: list[dict[str, Any]] = field(default_factory=list)
    marked_choice_inferences: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class SelectedFusionOutput:
    summary: AIScanSummary
    patient_info: AIScanPatientInfo
    medical_context: AIScanMedicalContext
    medicines: list[AIScanMedicine]
    other_charges: list[AIScanOtherCharge]
    financial_recap: AIScanFinancialRecap


@dataclass(frozen=True)
class ReconciliationArtifacts:
    derived_financials: dict[str, Any]
    corrected_fields: list[dict[str, Any]]
    unresolved_conflicts: list[str]


JudgeFieldFn = Callable[[str, list[Any], list[dict[str, Any]]], Awaitable[tuple[str | None, str | None]]]


def clamp_confidence(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 4)


@lru_cache(maxsize=2048)
def normalize_whitespace(value: str) -> str:
    return " ".join(value.strip().split())


@lru_cache(maxsize=4096)
def normalize_comparison_text(value: str) -> str:
    return re.sub(r"[\W_]+", "", normalize_whitespace(value).casefold())


def canonical_structured_field_key(field_path: str) -> str | None:
    if field_path == "diagnoses.description" or (
        field_path.startswith("diagnoses[") and field_path.endswith("].description")
    ):
        return "diagnoses.description"
    if field_path == "medicines.name" or (
        field_path.startswith("medicines[") and field_path.endswith("].name")
    ):
        return "medicines.name"
    return None


def infer_document_type_from_label(label: str) -> str:
    normalized = normalize_comparison_text(label)
    if any(keyword in normalized for keyword in ("resep", "prescription")):
        return "prescription"
    if any(keyword in normalized for keyword in ("formulir", "medicalform", "rawatjalan", "doctornote", "clinicalnote")):
        return "medical_form"
    if any(keyword in normalized for keyword in ("lab", "laboratorium")):
        return "lab_result"
    if any(keyword in normalized for keyword in ("radiology", "radiologi", "xray")):
        return "radiology_result"
    if any(keyword in normalized for keyword in ("receipt", "kwitansi", "invoice", "struk", "apotek")):
        return "receipt"
    return "unknown"


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


def normalize_gender_value(value: Any) -> str | None:
    if value is None:
        return None
    normalized_value = normalize_whitespace(str(value)).casefold()
    if not normalized_value:
        return None
    if normalized_value in {"male", "m", "pria", "laki laki", "lakilaki", "lelaki"}:
        return "MALE"
    if normalized_value in {"female", "f", "wanita", "perempuan"}:
        return "FEMALE"
    return normalize_whitespace(str(value)).upper()


def normalize_candidate_key(field_path: str, value: Any) -> str:
    if value is None:
        return "__null__"
    leaf_name = field_path.rsplit(".", 1)[-1]
    if leaf_name in DATE_FIELD_SUFFIXES:
        return coerce_date_value(value) or "__null__"
    if field_path in NUMERIC_FIELD_PATHS:
        numeric_value = coerce_numeric_value(value)
        return "__null__" if numeric_value is None else str(numeric_value)
    if field_path.endswith("currency"):
        return normalize_whitespace(str(value)).upper()
    if field_path.endswith("gender"):
        return normalize_gender_value(value) or "__null__"
    if is_semantic_field_path(field_path):
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
    if field_path.endswith("currency"):
        return normalize_whitespace(str(value)).upper()
    if field_path.endswith("gender"):
        return normalize_gender_value(value)
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
        validation_hints: dict[str, Any] | None,
        execution_mode: str,
        primary_model_provider: str | None,
        primary_model_name: str | None,
    ) -> MedicalExtractionOutcome:
        state = FusionWorkingState()
        normalized_hints = validation_hints or {}

        selected_output = await self._select_packet_output(
            successful_traces,
            images,
            state,
            validation_hints=normalized_hints,
        )

        result = AIMedicalScanResponse(
            summary=selected_output.summary,
            patient_info=selected_output.patient_info,
            medical_context=selected_output.medical_context,
            medicines=selected_output.medicines,
            other_charges=selected_output.other_charges,
            financial_recap=selected_output.financial_recap,
        )
        reconciliation = await self._reconcile_selection_stage(
            result,
            images=images,
            state=state,
        )
        self._apply_safe_output_policies(result, state)
        constraint_checks = self._apply_financial_constraints(result, state.field_confidence)
        review_reasons = self._build_review_reasons(
            state.field_confidence,
            constraint_checks,
            [
                *reconciliation.unresolved_conflicts,
                *state.inventory_conflicts,
            ],
        )
        review_required = bool(review_reasons)
        overall_confidence = self._compute_overall_confidence(state.field_confidence)

        extraction_meta_payload = self._build_extraction_meta_payload(
            state=state,
            validation_hints=normalized_hints,
            constraint_checks=constraint_checks,
            reconciliation=reconciliation,
            review_required=review_required,
            review_reasons=review_reasons,
            overall_confidence=overall_confidence,
        )
        return MedicalExtractionOutcome(
            result=result,
            overall_confidence=overall_confidence,
            field_confidence=state.field_confidence,
            review_required=review_required,
            review_reasons=review_reasons,
            calibration_version=CALIBRATION_VERSION,
            ai_execution_mode=execution_mode,
            primary_model_provider=primary_model_provider,
            primary_model_name=primary_model_name,
            ai_runs=all_traces,
            extraction_meta_payload=extraction_meta_payload,
        )

    async def _select_packet_output(
        self,
        traces: list[SuccessfulProducerInferenceTrace],
        images: list[Any],
        state: FusionWorkingState,
        *,
        validation_hints: dict[str, Any],
    ) -> SelectedFusionOutput:
        summary = await self._fuse_summary(
            traces,
            images,
            state.field_confidence,
            state.judge_invocations,
            validation_hints=validation_hints,
            inventory_conflicts=state.inventory_conflicts,
        )
        patient_info = await self._fuse_section_object(
            section_name="patient_info",
            section_source=[trace.payload.patient_info for trace in traces],
            fields=("name", "date_of_birth", "age", "gender", "address"),
            section_model=AIScanPatientInfo,
            traces=traces,
            images=images,
            field_confidence=state.field_confidence,
            judge_invocations=state.judge_invocations,
        )
        medical_context = await self._fuse_medical_context(
            traces,
            images,
            state.field_confidence,
            state.judge_invocations,
            list_group_meta=state.list_group_meta,
        )
        financial_recap = await self._fuse_section_object(
            section_name="financial_recap",
            section_source=[trace.payload.financial_recap for trace in traces],
            fields=("total_medicines", "total_other_charges", "grand_total"),
            section_model=AIScanFinancialRecap,
            traces=traces,
            images=images,
            field_confidence=state.field_confidence,
            judge_invocations=state.judge_invocations,
        )
        medicines = await self._fuse_list_items(
            traces,
            images,
            "medicines",
            state.field_confidence,
            state.judge_invocations,
            validation_hints=validation_hints,
            list_group_meta=state.list_group_meta,
        )
        medicine_amount_keys = {
            normalize_candidate_key("medicines.total_price", medicine.total_price)
            for medicine in medicines
            if medicine.total_price is not None
        }
        other_charges = await self._fuse_list_items(
            traces,
            images,
            "other_charges",
            state.field_confidence,
            state.judge_invocations,
            exclude_amount_keys=medicine_amount_keys,
            validation_hints=validation_hints,
            list_group_meta=state.list_group_meta,
        )
        return SelectedFusionOutput(
            summary=summary,
            patient_info=patient_info,
            medical_context=medical_context,
            medicines=medicines,
            other_charges=other_charges,
            financial_recap=financial_recap,
        )

    async def _reconcile_selection_stage(
        self,
        result: AIMedicalScanResponse,
        *,
        images: list[Any],
        state: FusionWorkingState,
    ) -> ReconciliationArtifacts:
        reconciliation_meta = await self._reconcile_financials(
            result,
            images=images,
            field_confidence=state.field_confidence,
            judge_invocations=state.judge_invocations,
            list_group_meta=state.list_group_meta,
        )
        return ReconciliationArtifacts(
            derived_financials=cast(dict[str, Any], reconciliation_meta.get("derived_financials", {})),
            corrected_fields=cast(list[dict[str, Any]], reconciliation_meta.get("corrected_fields", [])),
            unresolved_conflicts=cast(list[str], reconciliation_meta.get("unresolved_conflicts", [])),
        )

    def _apply_safe_output_policies(
        self,
        result: AIMedicalScanResponse,
        state: FusionWorkingState,
    ) -> None:
        state.normalized_terms.extend(
            self._normalize_structured_terms(result, state.field_confidence, state.list_group_meta)
        )
        state.nullified_fields.extend(
            self._nullify_ambiguous_fields(result, state.field_confidence, state.list_group_meta)
        )
        state.marked_choice_inferences.extend(self._collect_marked_choice_inferences(result))

    def _collect_marked_choice_inferences(self, result: AIMedicalScanResponse) -> list[dict[str, Any]]:
        if result.patient_info.gender not in {"MALE", "FEMALE"}:
            return []
        return [
            {
                "field_path": "patient_info.gender",
                "value": result.patient_info.gender,
                "reason": "preserved explicit gender selection from a marked exclusive-choice form field",
            }
        ]

    def _build_extraction_meta_payload(
        self,
        *,
        state: FusionWorkingState,
        validation_hints: dict[str, Any],
        constraint_checks: dict[str, Any],
        reconciliation: ReconciliationArtifacts,
        review_required: bool,
        review_reasons: list[str],
        overall_confidence: float | None,
    ) -> dict[str, Any]:
        return {
            "fusion_version": FUSION_VERSION,
            "calibration_version": CALIBRATION_VERSION,
            "reconciliation_version": RECONCILIATION_VERSION,
            "field_confidence": state.field_confidence,
            "list_group_meta": state.list_group_meta,
            "constraint_checks": constraint_checks,
            "judge_invocations": state.judge_invocations,
            "packet_inventory_hints": validation_hints,
            "inventory_conflicts": state.inventory_conflicts,
            "derived_financials": reconciliation.derived_financials,
            "corrected_fields": reconciliation.corrected_fields,
            "normalized_terms": state.normalized_terms,
            "nullified_fields": state.nullified_fields,
            "marked_choice_inferences": state.marked_choice_inferences,
            "unresolved_conflicts": reconciliation.unresolved_conflicts,
            "review_required": review_required,
            "review_reasons": review_reasons,
            "overall_confidence": overall_confidence,
        }

    def _iter_summary_scalar_values(
        self,
        traces: list[SuccessfulProducerInferenceTrace],
    ) -> list[tuple[str, list[tuple[SuccessfulProducerInferenceTrace, Any]]]]:
        summary_field_getters: tuple[tuple[str, Callable[[Any], Any]], ...] = (
            ("medical_facility_name", lambda payload: payload.summary.medical_facility_name),
            ("service_date", lambda payload: payload.summary.service_date or payload.summary.transaction_date),
            ("transaction_date", lambda payload: payload.summary.transaction_date),
            ("currency", lambda payload: payload.summary.currency),
            ("total_amount", lambda payload: payload.summary.total_amount),
        )
        return [
            (
                field_name,
                [(trace, getter(trace.payload)) for trace in traces],
            )
            for field_name, getter in summary_field_getters
        ]

    def _list_items_for_trace(self, trace: SuccessfulProducerInferenceTrace, list_name: str) -> list[Any]:
        if list_name == "diagnoses":
            return trace.payload.medical_context.diagnoses
        if list_name == "medicines":
            return trace.payload.medicines
        if list_name == "other_charges":
            return trace.payload.other_charges
        raise ValueError(f"Unsupported list name: {list_name}")

    def _list_item_schema(self, list_name: str) -> tuple[Any, tuple[str, ...]]:
        if list_name == "diagnoses":
            return AIScanDiagnosis, ("description", "doctor_name")
        if list_name == "medicines":
            return AIScanMedicine, ("name", "quantity", "unit", "dosage_instruction", "unit_price", "total_price")
        if list_name == "other_charges":
            return AIScanOtherCharge, ("category", "description", "amount")
        raise ValueError(f"Unsupported list name: {list_name}")

    def _store_list_group_meta(
        self,
        *,
        list_name: str,
        list_group_meta: dict[str, Any] | None,
        ranked_groups: list[dict[str, Any]],
        selected_groups: list[dict[str, Any]],
        target_count: int,
        bucket_targets: dict[str, int] | None,
        excluded_keys: dict[str, str],
    ) -> None:
        if list_group_meta is None:
            return
        list_group_meta[list_name] = self._serialize_list_group_meta(
            list_name,
            ranked_groups,
            selected_groups,
            target_count=target_count,
            bucket_targets=bucket_targets,
            excluded_keys=excluded_keys,
        )

    @staticmethod
    def _build_constraint_review_reasons(constraint_checks: dict[str, Any]) -> list[str]:
        reasons: list[str] = []
        for name, check in constraint_checks.items():
            if bool(check.get("used_derived_value", False)):
                reasons.append(
                    f"{name} was backfilled from derived financial totals because the extracted value was missing."
                )
            elif not bool(check.get("passed", False)):
                reasons.append(f"{name} did not match derived financial totals.")
        return reasons

    async def _fuse_summary(
        self,
        traces: list[SuccessfulProducerInferenceTrace],
        images: list[Any],
        field_confidence: dict[str, Any],
        judge_invocations: list[dict[str, Any]],
        *,
        validation_hints: dict[str, Any],
        inventory_conflicts: list[str],
    ) -> AIScanSummary:
        detected_documents, detected_document_meta = await self._fuse_detected_documents(
            traces,
            images,
            judge_invocations,
            validation_hints=validation_hints,
            inventory_conflicts=inventory_conflicts,
        )
        field_confidence["summary.detected_documents"] = detected_document_meta
        fused_scalars: dict[str, Any] = {}
        for field_name, values in self._iter_summary_scalar_values(traces):
            fused_scalars[field_name] = await self._fuse_scalar(
                f"summary.{field_name}",
                values,
                images,
                field_confidence,
                judge_invocations,
            )
        return AIScanSummary(detected_documents=detected_documents, **fused_scalars)

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

    async def _fuse_medical_context(
        self,
        traces: list[SuccessfulProducerInferenceTrace],
        images: list[Any],
        field_confidence: dict[str, Any],
        judge_invocations: list[dict[str, Any]],
        *,
        list_group_meta: dict[str, Any],
    ) -> AIScanMedicalContext:
        scalar_values = {
            "anamnesis": [(trace, trace.payload.medical_context.anamnesis) for trace in traces],
            "therapy_note": [(trace, trace.payload.medical_context.therapy_note) for trace in traces],
        }
        fused_scalars: dict[str, Any] = {}
        for field_name, values in scalar_values.items():
            fused_scalars[field_name] = await self._fuse_scalar(
                f"medical_context.{field_name}",
                values,
                images,
                field_confidence,
                judge_invocations,
            )
        diagnoses = await self._fuse_list_items(traces, images, "diagnoses", field_confidence, judge_invocations, list_group_meta=list_group_meta)
        return AIScanMedicalContext(
            anamnesis=fused_scalars["anamnesis"],
            diagnoses=diagnoses,
            therapy_note=fused_scalars["therapy_note"],
        )

    def _collect_list_groups(
        self,
        traces: list[SuccessfulProducerInferenceTrace],
        list_name: str,
    ) -> tuple[list[int], list[dict[str, Any]]]:
        item_groups: dict[str, dict[str, Any]] = {}
        primary_rank = min((trace.producer_rank for trace in traces), default=0)
        item_counts = [len(self._list_items_for_trace(trace, list_name)) for trace in traces]

        for trace in traces:
            for item in self._list_items_for_trace(trace, list_name):
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
            raw_kind = self._extract_other_charge_raw_kind(group["key"]) if list_name == "other_charges" else None
            bucket = normalize_other_charge_bucket(raw_kind or "other")
            score = score_candidate_group(group["producer_ranks"], group["producers"], run_count)
            if raw_kind == "mixed":
                score = max(score - 0.2, 0.0)
            representative_item = cast(tuple[SuccessfulProducerInferenceTrace, Any], group["items"][0])[1]
            medicine_names = [
                normalize_whitespace(getattr(item, "name", ""))
                for _, item in cast(list[tuple[SuccessfulProducerInferenceTrace, Any]], group["items"])
                if getattr(item, "name", None)
            ]
            compound_components = sorted(
                {
                    component
                    for medicine_name in medicine_names
                    for component in extract_compound_medicine_components(medicine_name)
                }
            )
            ranked_groups.append(
                {
                    **group,
                    "run_count": run_count,
                    "score": score,
                    "raw_kind": raw_kind,
                    "bucket": bucket,
                    "has_primary_support": any(
                        trace.producer_rank == primary_rank
                        for trace, _ in cast(list[tuple[SuccessfulProducerInferenceTrace, Any]], group["items"])
                    ),
                    "representative_name": getattr(representative_item, "name", None),
                    "is_package_level_medicine": list_name == "medicines"
                    and any(is_package_level_medicine_name(name) for name in medicine_names),
                    "compound_components": compound_components,
                    "is_grouped_compound": list_name == "medicines"
                    and len(compound_components) >= MEDICINE_SPLIT_COMPONENT_MIN,
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
        return item_counts, ranked_groups

    def _select_list_groups(
        self,
        list_name: str,
        *,
        traces: list[SuccessfulProducerInferenceTrace],
        ranked_groups: list[dict[str, Any]],
        target_count: int,
        bucket_targets: dict[str, int] | None,
        exclude_amount_keys: set[str] | None,
    ) -> tuple[list[dict[str, Any]], dict[str, str]]:
        excluded_keys: dict[str, str] = {}
        candidate_groups = ranked_groups
        if list_name == "other_charges" and exclude_amount_keys:
            filtered_groups: list[dict[str, Any]] = []
            for group in candidate_groups:
                if self._should_exclude_other_charge_group(group["key"], exclude_amount_keys):
                    excluded_keys[group["key"]] = "duplicates selected medicine pricing"
                    continue
                filtered_groups.append(group)
            candidate_groups = filtered_groups

        if target_count <= 0 or not candidate_groups:
            return [], excluded_keys

        if list_name == "other_charges":
            return (
                self._select_other_charge_groups(
                    candidate_groups,
                    target_count=target_count,
                    bucket_targets=bucket_targets or {},
                ),
                excluded_keys,
            )
        if list_name == "medicines":
            selected_groups, medicine_exclusions = self._select_medicine_groups(
                candidate_groups,
                target_count=target_count,
            )
            excluded_keys.update(medicine_exclusions)
            return selected_groups, excluded_keys
        return self._select_groups_by_stability(candidate_groups, target_count), excluded_keys

    async def _enrich_selected_list_groups(
        self,
        list_name: str,
        selected_groups: list[dict[str, Any]],
        *,
        images: list[Any],
        field_confidence: dict[str, Any],
        judge_invocations: list[dict[str, Any]],
        list_group_meta: dict[str, Any] | None,
    ) -> list[Any]:
        item_model, item_fields = self._list_item_schema(list_name)
        selected_group_meta: list[dict[str, Any]] | None = None
        if list_group_meta is not None:
            selected_group_meta = cast(list[dict[str, Any]], list_group_meta[list_name]["selected_groups"])
        fused_items: list[Any] = []
        for output_index, group in enumerate(selected_groups):
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
            if selected_group_meta is not None:
                selected_group_meta[output_index]["path_prefix"] = path_prefix
        return fused_items

    async def _fuse_list_items(
        self,
        traces: list[SuccessfulProducerInferenceTrace],
        images: list[Any],
        list_name: str,
        field_confidence: dict[str, Any],
        judge_invocations: list[dict[str, Any]],
        *,
        exclude_amount_keys: set[str] | None = None,
        validation_hints: dict[str, Any] | None = None,
        list_group_meta: dict[str, Any] | None = None,
    ) -> list[Any]:
        item_counts, ranked_groups = self._collect_list_groups(traces, list_name)
        target_count = self._determine_list_target_count(
            list_name,
            traces=traces,
            raw_counts=item_counts,
            validation_hints=validation_hints or {},
        )
        bucket_targets = self._determine_other_charge_bucket_targets(traces) if list_name == "other_charges" else None

        selected_groups, excluded_keys = self._select_list_groups(
            list_name,
            traces=traces,
            ranked_groups=ranked_groups,
            target_count=target_count,
            bucket_targets=bucket_targets,
            exclude_amount_keys=exclude_amount_keys,
        )

        self._store_list_group_meta(
            list_name=list_name,
            list_group_meta=list_group_meta,
            ranked_groups=ranked_groups,
            selected_groups=selected_groups,
            target_count=target_count,
            bucket_targets=bucket_targets,
            excluded_keys=excluded_keys,
        )

        if target_count <= 0 or not ranked_groups or not selected_groups:
            return []

        return await self._enrich_selected_list_groups(
            list_name,
            selected_groups,
            images=images,
            field_confidence=field_confidence,
            judge_invocations=judge_invocations,
            list_group_meta=list_group_meta,
        )

    def _build_scalar_candidates(
        self,
        field_path: str,
        values: list[tuple[SuccessfulProducerInferenceTrace, Any]],
    ) -> list[dict[str, Any]]:
        groups: dict[str, dict[str, Any]] = {}
        for trace, value in values:
            normalized_key = normalize_candidate_key(field_path, value)
            group = groups.setdefault(
                normalized_key,
                {
                    "normalized_key": normalized_key,
                    "raw_value": format_output_value(field_path, value),
                    "traces": [],
                    "producers": set(),
                    "producer_ranks": {},
                },
            )
            if group["raw_value"] is None and value is not None:
                group["raw_value"] = format_output_value(field_path, value)
            group["traces"].append(trace)
            group["producers"].add(trace.producer_id)
            group["producer_ranks"][trace.producer_id] = trace.producer_rank

        ranked = [
            {
                **group,
                "score": score_candidate_group(group["producer_ranks"], group["producers"], len(group["traces"])),
                "run_count": len(group["traces"]),
                "producer_count": len(group["producers"]),
            }
            for group in groups.values()
        ]
        ranked.sort(
            key=lambda item: (
                -item["score"],
                -item["producer_count"],
                -item["run_count"],
                item["raw_value"] is None,
                str(item["raw_value"]),
            )
        )
        return ranked

    def _compute_scalar_confidence(
        self,
        field_path: str,
        *,
        winner: dict[str, Any],
        runner_up: dict[str, Any] | None,
        total_runs: int,
        total_producers: int,
        judge_used: bool,
    ) -> tuple[float, float]:
        producer_support_ratio = len(winner["producers"]) / max(total_producers, 1)
        run_support_ratio = winner["run_count"] / max(total_runs, 1)
        margin_ratio = 1.0 if runner_up is None or winner["score"] <= 0 else max((winner["score"] - runner_up["score"]) / winner["score"], 0.0)
        raw_confidence = clamp_confidence((0.45 * producer_support_ratio) + (0.20 * run_support_ratio) + (0.35 * margin_ratio))
        if judge_used and winner["raw_value"] is not None:
            raw_confidence = max(raw_confidence, 0.72 if is_critical_field_path(field_path) else 0.68)
        return raw_confidence, clamp_confidence(producer_support_ratio)

    async def _fuse_scalar(self, field_path: str, values: list[tuple[SuccessfulProducerInferenceTrace, Any]], images: list[Any], field_confidence: dict[str, Any], judge_invocations: list[dict[str, Any]], *, output_path: str | None = None) -> Any:
        total_runs = len(values)
        total_producers = len({trace.producer_id for trace, _ in values})
        output_field_path = output_path or field_path
        ranked = self._build_scalar_candidates(field_path, values)
        winner = ranked[0] if ranked else {"raw_value": None, "traces": [], "producers": set(), "score": 0.0, "run_count": 0, "normalized_key": "__null__"}
        runner_up = ranked[1] if len(ranked) > 1 else None

        judge_used = False
        judge_reason: str | None = None
        rule_hits: list[str] = []
        close_runner_up = False
        if output_field_path == "patient_info.gender" and any(
            isinstance(value, str) and normalize_whitespace(value).upper() not in {"MALE", "FEMALE"}
            and normalize_gender_value(value) in {"MALE", "FEMALE"}
            for _, value in values
        ):
            rule_hits.append("normalized local gender label to canonical enum")
        if runner_up is not None and winner["score"] > 0:
            close_runner_up = ((winner["score"] - runner_up["score"]) / winner["score"]) < 0.18

        winner, gender_rule_hit = self._apply_conservative_gender_guard(ranked, winner, output_field_path)
        if gender_rule_hit is not None:
            rule_hits.append(gender_rule_hit)

        judge_policy = self._judge_policy_for_field(
            field_path,
            output_field_path=output_field_path,
            ranked=ranked,
            winner=winner,
            runner_up=runner_up,
            close_runner_up=close_runner_up,
        )
        if judge_policy is not None:
            selected_key, judge_reason = await self._judge_field(field_path, images, ranked[:3])
            if selected_key is not None:
                matching = next((candidate for candidate in ranked if candidate["normalized_key"] == selected_key), None)
                if matching is not None:
                    winner = matching
                    judge_used = True
                    rule_hits.append(judge_policy)
                    judge_invocations.append({"field_path": output_field_path, "selected_candidate": winner["raw_value"], "reason": judge_reason})

        raw_confidence, producer_agreement = self._compute_scalar_confidence(
            field_path,
            winner=winner,
            runner_up=runner_up,
            total_runs=total_runs,
            total_producers=total_producers,
            judge_used=judge_used,
        )
        confidence = self._calibrate_confidence(field_path, raw_confidence)
        review_flags = ["low_confidence"] if confidence < self._threshold_for(field_path) else []
        if judge_used:
            review_flags.append("judge_resolved")

        alternate_candidates = [
            candidate for candidate in ranked if candidate["normalized_key"] != winner["normalized_key"]
        ]

        field_confidence[output_field_path] = {
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
            "rule_hits": rule_hits,
            "review_flags": review_flags,
            "calibration_version": CALIBRATION_VERSION,
            "support_count": winner["run_count"],
            "producer_agreement": producer_agreement,
        }
        return winner["raw_value"]

    def _set_meta_selected_value(
        self,
        meta: dict[str, Any],
        field_path: str,
        value: Any,
        *,
        min_confidence: float | None = None,
        rule_hit: str | None = None,
        review_flag: str | None = None,
    ) -> None:
        formatted_value = format_output_value(field_path, value)
        meta["value"] = formatted_value
        if min_confidence is not None:
            meta["confidence"] = max(float(meta.get("confidence", 0.0)), min_confidence)
        if rule_hit is not None:
            meta["rule_hits"] = [*meta.get("rule_hits", []), rule_hit]
        if review_flag is not None and review_flag not in meta.get("review_flags", []):
            meta.setdefault("review_flags", []).append(review_flag)

        candidates = cast(list[dict[str, Any]], meta.get("candidates", []))
        if not candidates:
            return

        selected_key = normalize_candidate_key(field_path, value)
        selected_candidate = None
        for candidate in candidates:
            candidate["selected"] = candidate.get("normalized_key") == selected_key
            if candidate["selected"]:
                candidate["value"] = formatted_value
                selected_candidate = candidate
        if selected_candidate is None:
            candidates.insert(
                0,
                {
                    "normalized_key": selected_key,
                    "value": formatted_value,
                    "selected": True,
                    "support_count": 0,
                    "producer_count": 0,
                    "support": [],
                },
            )
        meta["alternates"] = [
            {
                "value": candidate.get("value"),
                "support_count": candidate.get("support_count", 0),
                "producer_count": candidate.get("producer_count", 0),
            }
            for candidate in candidates
            if not candidate.get("selected")
        ][:3]

    def _apply_financial_constraints(self, result: AIMedicalScanResponse, field_confidence: dict[str, Any]) -> dict[str, Any]:
        recap_medicines_total = coerce_numeric_value(result.financial_recap.total_medicines)
        medicine_line_item_sum = self._sum_defined_numbers([medicine.total_price for medicine in result.medicines])
        medicines_sum = medicine_line_item_sum if medicine_line_item_sum is not None else recap_medicines_total
        other_charges_sum = self._sum_defined_numbers([charge.amount for charge in result.other_charges])
        recap_total = self._sum_defined_numbers([
            recap_medicines_total if recap_medicines_total is not None else medicine_line_item_sum,
            other_charges_sum,
        ])
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
                self._set_meta_selected_value(
                    meta,
                    field_path,
                    derived_value,
                    min_confidence=0.83,
                    rule_hit="constraint backfilled missing total",
                    review_flag="constraint_adjusted",
                )
        return {"passed": passed, "derived_value": derived_value, "original_value": current_value, "final_value": self._get_field_value(result, field_path), "used_derived_value": used_derived_value}

    def _sum_defined_numbers(self, values: list[float | int | None] | list[float | int]) -> float | int | None:
        parsed = [coerce_numeric_value(value) for value in values if value is not None]
        if not parsed:
            return None
        total = 0.0
        for value in parsed:
            if value is not None:
                total += float(value)
        return int(total) if total.is_integer() else round(total, 2)

    def _build_review_reasons(self, field_confidence: dict[str, Any], constraint_checks: dict[str, Any], unresolved_conflicts: list[str]) -> list[str]:
        reasons = [f"{field_path} confidence is below the review threshold." for field_path, meta in field_confidence.items() if isinstance(meta, dict) and float(meta.get("confidence", 0.0)) < self._threshold_for(field_path)]
        reasons.extend(self._build_constraint_review_reasons(constraint_checks))
        reasons.extend(unresolved_conflicts)
        return list(dict.fromkeys(reasons))

    def _compute_overall_confidence(self, field_confidence: dict[str, Any]) -> float | None:
        weighted = [(float(meta["confidence"]), 2.0 if is_critical_field_path(field_path) else 1.0) for field_path, meta in field_confidence.items() if isinstance(meta, dict) and "confidence" in meta]
        if not weighted:
            return None
        return clamp_confidence(sum(score * weight for score, weight in weighted) / sum(weight for _, weight in weighted))

    def _calibrate_confidence(self, field_path: str, raw_confidence: float) -> float:
        adjusted = raw_confidence + (0.03 if is_critical_field_path(field_path) else 0.0) - (0.04 if is_semantic_field_path(field_path) else 0.0)
        return clamp_confidence(adjusted)

    def _threshold_for(self, field_path: str) -> float:
        if is_critical_field_path(field_path):
            return CRITICAL_FIELD_THRESHOLD
        if is_semantic_field_path(field_path):
            return SEMANTIC_FIELD_THRESHOLD
        return NON_CRITICAL_FIELD_THRESHOLD

    def _get_field_value(self, result: AIMedicalScanResponse, field_path: str) -> Any:
        section_name, field_name = field_path.split(".", 1)
        return getattr(getattr(result, section_name), field_name)

    def _set_field_value(self, result: AIMedicalScanResponse, field_path: str, value: Any) -> None:
        section_name, field_name = field_path.split(".", 1)
        setattr(getattr(result, section_name), field_name, value)

    def _resolve_list_item_reference(
        self,
        result: AIMedicalScanResponse,
        list_group_meta: dict[str, Any],
        field_path: str,
    ) -> tuple[Any, str] | None:
        match = LIST_ITEM_FIELD_PATH_PATTERN.match(field_path)
        if match is None:
            return None
        list_name, slug, field_name = match.groups()
        selected_groups = cast(list[dict[str, Any]], cast(dict[str, Any], list_group_meta.get(list_name, {})).get("selected_groups", []))
        path_prefix = f"{list_name}[{slug}]"
        selected_group = next((group for group in selected_groups if group.get("path_prefix") == path_prefix), None)
        if selected_group is None:
            return None

        if list_name == "diagnoses":
            items = result.medical_context.diagnoses
        else:
            items = getattr(result, list_name)

        output_index = int(selected_group.get("output_index", -1))
        if output_index < 0 or output_index >= len(items):
            return None
        return items[output_index], field_name

    def _get_output_field_value(
        self,
        result: AIMedicalScanResponse,
        list_group_meta: dict[str, Any],
        field_path: str,
    ) -> Any:
        if "[" not in field_path:
            return self._get_field_value(result, field_path)
        resolved = self._resolve_list_item_reference(result, list_group_meta, field_path)
        if resolved is None:
            return None
        item, field_name = resolved
        return getattr(item, field_name)

    def _set_output_field_value(
        self,
        result: AIMedicalScanResponse,
        list_group_meta: dict[str, Any],
        field_path: str,
        value: Any,
    ) -> bool:
        if "[" not in field_path:
            self._set_field_value(result, field_path, value)
            return True
        resolved = self._resolve_list_item_reference(result, list_group_meta, field_path)
        if resolved is None:
            return False
        item, field_name = resolved
        setattr(item, field_name, value)
        return True

    def _sync_list_group_representative_field(
        self,
        list_group_meta: dict[str, Any],
        field_path: str,
        value: Any,
    ) -> None:
        match = LIST_ITEM_FIELD_PATH_PATTERN.match(field_path)
        if match is None:
            return
        list_name, slug, field_name = match.groups()
        selected_groups = cast(list[dict[str, Any]], cast(dict[str, Any], list_group_meta.get(list_name, {})).get("selected_groups", []))
        path_prefix = f"{list_name}[{slug}]"
        for group in selected_groups:
            if group.get("path_prefix") != path_prefix:
                continue
            representative_fields = cast(dict[str, Any], group.setdefault("representative_fields", {}))
            representative_fields[field_name] = value
            break

    def _note_field_rule_hit(
        self,
        field_confidence: dict[str, Any],
        field_path: str,
        rule_hit: str,
        *,
        review_flag: str | None = None,
    ) -> None:
        meta = field_confidence.get(field_path)
        if not isinstance(meta, dict):
            return
        existing_rule_hits = cast(list[str], meta.setdefault("rule_hits", []))
        if rule_hit not in existing_rule_hits:
            existing_rule_hits.append(rule_hit)
        if review_flag is not None:
            existing_review_flags = cast(list[str], meta.setdefault("review_flags", []))
            if review_flag not in existing_review_flags:
                existing_review_flags.append(review_flag)

    def _normalize_structured_terms(
        self,
        result: AIMedicalScanResponse,
        field_confidence: dict[str, Any],
        list_group_meta: dict[str, Any],
    ) -> list[dict[str, Any]]:
        normalized_terms: list[dict[str, Any]] = []
        for field_path, meta in field_confidence.items():
            if not isinstance(meta, dict):
                continue
            structured_key = canonical_structured_field_key(field_path)
            if structured_key is None:
                continue
            raw_value = self._get_output_field_value(result, list_group_meta, field_path)
            if not isinstance(raw_value, str) or not raw_value.strip():
                continue
            normalized_value = normalize_comparison_text(raw_value)
            best_match: str | None = None
            best_score = 0.0
            for canonical_value in STRUCTURED_TERM_LEXICON.get(structured_key, ()):
                score = sequence_similarity(normalized_value, normalize_comparison_text(canonical_value))
                if score > best_score:
                    best_score = score
                    best_match = canonical_value
            if best_match is None or best_score < 0.85 or normalize_comparison_text(best_match) == normalized_value:
                continue
            if not self._set_output_field_value(result, list_group_meta, field_path, best_match):
                continue
            self._set_meta_selected_value(
                meta,
                field_path,
                best_match,
                min_confidence=max(float(meta.get("confidence", 0.0)), 0.72),
                rule_hit="canonicalized against structured medical lexicon",
            )
            self._sync_list_group_representative_field(list_group_meta, field_path, best_match)
            normalized_terms.append(
                {
                    "field_path": field_path,
                    "raw_value": raw_value,
                    "canonical_value": best_match,
                    "similarity": clamp_confidence(best_score),
                }
            )
        return normalized_terms

    def _nullify_ambiguous_fields(
        self,
        result: AIMedicalScanResponse,
        field_confidence: dict[str, Any],
        list_group_meta: dict[str, Any],
    ) -> list[dict[str, Any]]:
        nullified_fields: list[dict[str, Any]] = []
        for field_path, meta in field_confidence.items():
            if not isinstance(meta, dict) or not is_semantic_field_path(field_path):
                continue
            candidates = [
                candidate
                for candidate in cast(list[dict[str, Any]], meta.get("candidates", []))
                if candidate.get("value") not in {None, ""}
            ]
            if meta.get("judge_used") or len(candidates) <= 1:
                continue
            if float(meta.get("confidence", 0.0)) >= self._threshold_for(field_path):
                continue
            if not self._meta_candidates_materially_divergent(candidates):
                continue
            if int(meta.get("support_count", 0)) > 1 and float(meta.get("producer_agreement", 0.0)) >= 0.67:
                continue
            original_value = self._get_output_field_value(result, list_group_meta, field_path)
            if original_value in {None, ""}:
                continue
            if not self._set_output_field_value(result, list_group_meta, field_path, None):
                continue
            self._set_meta_selected_value(
                meta,
                field_path,
                None,
                rule_hit="nullified ambiguous handwritten field after unresolved adjudication",
                review_flag="nullified_ambiguous",
            )
            self._sync_list_group_representative_field(list_group_meta, field_path, None)
            nullified_fields.append(
                {
                    "field_path": field_path,
                    "original_value": original_value,
                    "reason": "ambiguous handwritten field remained unresolved after adjudication",
                }
            )
        return nullified_fields

    def _apply_conservative_gender_guard(
        self,
        ranked: list[dict[str, Any]],
        winner: dict[str, Any],
        output_field_path: str,
    ) -> tuple[dict[str, Any], str | None]:
        if output_field_path != "patient_info.gender" or winner.get("raw_value") is None:
            return winner, None

        null_candidate = next((candidate for candidate in ranked if candidate.get("raw_value") is None), None)
        non_null_candidates = [candidate for candidate in ranked if candidate.get("raw_value") is not None]
        if null_candidate is None or len(non_null_candidates) < 2:
            return winner, None

        explicit_winner = non_null_candidates[0]
        explicit_runner_up = non_null_candidates[1]
        explicit_margin_ratio = 0.0
        if float(explicit_winner.get("score", 0.0)) > 0:
            explicit_margin_ratio = max(
                (float(explicit_winner.get("score", 0.0)) - float(explicit_runner_up.get("score", 0.0))) / float(explicit_winner.get("score", 0.0)),
                0.0,
            )

        total_gender_runs = sum(int(candidate.get("run_count", 0)) for candidate in ranked)
        null_support_ratio = int(null_candidate.get("run_count", 0)) / max(total_gender_runs, 1)

        if explicit_margin_ratio >= GENDER_AMBIGUITY_MARGIN_THRESHOLD or null_support_ratio < GENDER_NULL_SUPPORT_RATIO_THRESHOLD:
            return winner, None

        return null_candidate, "prefer null over inferred gender when model support is ambiguous"

    def _judge_policy_for_field(
        self,
        field_path: str,
        *,
        output_field_path: str,
        ranked: list[dict[str, Any]],
        winner: dict[str, Any],
        runner_up: dict[str, Any] | None,
        close_runner_up: bool,
    ) -> str | None:
        if len(ranked) <= 1:
            return None
        if runner_up is None:
            return None
        if field_path == "summary.service_date" and winner["normalized_key"] != runner_up["normalized_key"]:
            return "judge used for conflicting service date candidates"
        if is_financial_field_path(field_path) and winner["normalized_key"] != runner_up["normalized_key"]:
            return "judge used for conflicting financial candidates"
        if output_field_path.startswith("medicines[") and output_field_path.endswith(".name") and self._materially_divergent_candidates(ranked):
            return "judge used for divergent medicine-name candidates"
        if field_path.endswith(".dosage_instruction") and self._is_material_dosage_conflict(ranked, winner, runner_up):
            return "judge used for material dosage conflict"
        if output_field_path in {"medical_context.anamnesis", "medical_context.therapy_note"} and self._materially_divergent_candidates(ranked):
            return "judge used for divergent handwritten semantic field"
        if output_field_path.startswith("diagnoses[") and output_field_path.endswith(".description") and self._materially_divergent_candidates(ranked):
            return "judge used for divergent diagnosis candidates"
        if close_runner_up and is_critical_field_path(field_path):
            return "judge used for close critical-field conflict"
        return None

    def _materially_divergent_candidates(self, ranked: list[dict[str, Any]]) -> bool:
        comparable_candidates = [candidate for candidate in ranked if candidate.get("raw_value") not in {None, ""}]
        if len(comparable_candidates) < 2:
            return False

        first_key = str(comparable_candidates[0]["normalized_key"])
        second_key = str(comparable_candidates[1]["normalized_key"])
        return sequence_similarity(first_key, second_key) < HANDWRITTEN_DIVERGENCE_THRESHOLD

    def _meta_candidates_materially_divergent(self, candidates: list[dict[str, Any]]) -> bool:
        non_empty_candidates = [candidate for candidate in candidates if candidate.get("value") not in {None, ""}]
        if len(non_empty_candidates) < 2:
            return False
        first_key = str(non_empty_candidates[0].get("normalized_key", ""))
        second_key = str(non_empty_candidates[1].get("normalized_key", ""))
        return sequence_similarity(first_key, second_key) < HANDWRITTEN_DIVERGENCE_THRESHOLD

    def _is_material_dosage_conflict(
        self,
        ranked: list[dict[str, Any]],
        winner: dict[str, Any],
        runner_up: dict[str, Any] | None,
    ) -> bool:
        if runner_up is None:
            return False
        if winner.get("raw_value") in {None, ""} or runner_up.get("raw_value") in {None, ""}:
            return False
        comparable_candidates = [candidate for candidate in ranked if candidate.get("raw_value") not in {None, ""}]
        if len(comparable_candidates) < 2:
            return False
        return winner.get("normalized_key") != runner_up.get("normalized_key") and (
            winner.get("run_count", 0) <= 1 or runner_up.get("run_count", 0) <= 1
            or winner.get("producer_count", 0) <= 1
            or runner_up.get("producer_count", 0) <= 1
        )

    async def _maybe_rejudge_field_from_candidates(
        self,
        result: AIMedicalScanResponse,
        *,
        field_path: str,
        images: list[Any],
        field_confidence: dict[str, Any],
        judge_invocations: list[dict[str, Any]],
    ) -> bool:
        meta = field_confidence.get(field_path)
        if not isinstance(meta, dict):
            return False
        if meta.get("judge_used") or len(meta.get("candidates", [])) <= 1:
            return False

        candidate_inputs = [
            {
                "normalized_key": candidate.get("normalized_key"),
                "raw_value": candidate.get("value"),
                "run_count": candidate.get("support_count", 0),
                "producers": set(range(int(candidate.get("producer_count", 0)))),
            }
            for candidate in cast(list[dict[str, Any]], meta.get("candidates", []))[:3]
        ]
        if len(candidate_inputs) <= 1:
            return False

        selected_key, judge_reason = await self._judge_field(field_path, images, candidate_inputs)
        if selected_key is None:
            return False

        selected_candidate = next((candidate for candidate in candidate_inputs if candidate["normalized_key"] == selected_key), None)
        if selected_candidate is None or selected_candidate["raw_value"] == meta.get("value"):
            return False

        self._set_field_value(result, field_path, selected_candidate["raw_value"])
        meta["value"] = selected_candidate["raw_value"]
        meta["judge_used"] = True
        meta["judge_reason"] = judge_reason
        meta["confidence"] = max(float(meta.get("confidence", 0.0)), 0.72 if is_critical_field_path(field_path) else 0.68)
        meta["rule_hits"] = [*meta.get("rule_hits", []), "judge applied after reconciliation conflict"]
        if "judge_resolved" not in meta.get("review_flags", []):
            meta.setdefault("review_flags", []).append("judge_resolved")
        judge_invocations.append(
            {
                "field_path": field_path,
                "selected_candidate": selected_candidate["raw_value"],
                "reason": judge_reason,
            }
        )
        return True

    def _prune_noisy_medicines_for_reconciliation(
        self,
        result: AIMedicalScanResponse,
        *,
        field_confidence: dict[str, Any],
        list_group_meta: dict[str, Any],
    ) -> list[str]:
        medicine_meta = cast(dict[str, Any], list_group_meta.get("medicines", {}))
        selected_groups = cast(list[dict[str, Any]], medicine_meta.get("selected_groups", []))
        removed_indices: list[int] = []
        notes: list[str] = []

        for index, medicine in enumerate(result.medicines):
            if is_package_level_medicine_name(medicine.name):
                removed_indices.append(index)

        if not removed_indices:
            return notes

        for index in sorted(set(removed_indices), reverse=True):
            if index < len(result.medicines):
                result.medicines.pop(index)
            if index < len(selected_groups):
                group_meta = selected_groups.pop(index)
                path_prefix = group_meta.get("path_prefix")
                if isinstance(path_prefix, str) and path_prefix:
                    for field_key in [key for key in field_confidence if key.startswith(f"{path_prefix}.")]:
                        field_confidence.pop(field_key, None)

        for output_index, selected_group in enumerate(selected_groups):
            selected_group["output_index"] = output_index

        notes.append("Suppressed generic/package-level medicine artifacts before ledger reconciliation.")
        return notes

    def _build_medicine_ledger_candidates(
        self,
        result: AIMedicalScanResponse,
        *,
        selected_other_charge_groups: list[dict[str, Any]],
        list_group_meta: dict[str, Any],
    ) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        seen_amounts: set[float | int] = set()

        def add_candidate(candidate: dict[str, Any]) -> None:
            amount = coerce_numeric_value(candidate.get("amount"))
            if amount is None or amount in seen_amounts:
                return
            seen_amounts.add(amount)
            candidates.append({**candidate, "amount": amount})

        for group in selected_other_charge_groups:
            if group.get("bucket") != "medicine":
                continue
            amount = coerce_numeric_value(group.get("amount"))
            if amount is None:
                continue
            add_candidate(
                {
                    "amount": amount,
                    "source_type": "other_charge",
                    "source_key": group.get("key"),
                    "path_prefix": group.get("path_prefix"),
                    "score": float(group.get("score", 0.0)) + (0.15 * float(group.get("support_count", 0))),
                    "support_count": int(group.get("support_count", 0)),
                    "producer_count": int(group.get("producer_count", 0)),
                }
            )

        selected_medicine_groups = cast(list[dict[str, Any]], cast(dict[str, Any], list_group_meta.get("medicines", {})).get("selected_groups", []))
        priced_medicine_totals: list[float | int] = []
        priced_medicine_paths: list[str] = []
        for group in selected_medicine_groups:
            output_index = int(group.get("output_index", -1))
            if output_index < 0 or output_index >= len(result.medicines):
                continue
            total_price = coerce_numeric_value(result.medicines[output_index].total_price)
            path_prefix = group.get("path_prefix")
            if total_price is None or not isinstance(path_prefix, str):
                continue
            priced_medicine_totals.append(total_price)
            priced_medicine_paths.append(f"{path_prefix}.total_price")
            add_candidate(
                {
                    "amount": total_price,
                    "source_type": "medicine_row",
                    "source_key": group.get("key"),
                    "path_prefix": path_prefix,
                    "score": float(group.get("score", 0.0))
                    + (0.2 if bool(group.get("is_grouped_compound")) or bool(group.get("is_package_level_medicine")) else 0.0),
                    "support_count": int(group.get("support_count", 0)),
                    "producer_count": int(group.get("producer_count", 0)),
                }
            )

        visible_medicine_total = self._sum_defined_numbers(priced_medicine_totals)
        if visible_medicine_total is not None:
            add_candidate(
                {
                    "amount": visible_medicine_total,
                    "source_type": "visible_medicine_sum",
                    "source_key": "visible_medicine_sum",
                    "path_prefixes": priced_medicine_paths,
                    "score": 0.2 + (0.05 * len(priced_medicine_paths)),
                    "support_count": len(priced_medicine_paths),
                    "producer_count": 0,
                }
            )

        return candidates

    def _annotate_ledger_selection(
        self,
        *,
        result: AIMedicalScanResponse,
        field_confidence: dict[str, Any],
        list_group_meta: dict[str, Any],
        canonical_source: dict[str, Any] | None,
        dropped_group_indices: list[int],
        selected_other_charge_groups: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        price_counted_in_ledger: list[dict[str, Any]] = []
        financial_outlier_suppressions: list[dict[str, Any]] = []
        selected_canonical_path = canonical_source.get("path_prefix") if isinstance(canonical_source, dict) else None

        selected_medicine_groups = cast(list[dict[str, Any]], cast(dict[str, Any], list_group_meta.get("medicines", {})).get("selected_groups", []))
        for group in selected_medicine_groups:
            path_prefix = group.get("path_prefix")
            if not isinstance(path_prefix, str):
                continue
            amount_path = f"{path_prefix}.total_price"
            output_index = int(group.get("output_index", -1))
            amount = None
            if 0 <= output_index < len(result.medicines):
                amount = coerce_numeric_value(result.medicines[output_index].total_price)
            if amount is None:
                continue
            counted = bool(
                canonical_source
                and canonical_source.get("source_type") in {"medicine_row", "visible_medicine_sum"}
                and (
                    canonical_source.get("path_prefix") == path_prefix
                    or amount_path in cast(list[str], canonical_source.get("path_prefixes", []))
                )
            )
            if counted:
                self._note_field_rule_hit(field_confidence, amount_path, "counted in canonical medicine ledger")
            else:
                self._note_field_rule_hit(
                    field_confidence,
                    amount_path,
                    "excluded from canonical ledger totals because a packet-level medicine source was preferred",
                )
            price_counted_in_ledger.append(
                {
                    "field_path": amount_path,
                    "amount": amount,
                    "counted": counted,
                }
            )

        if isinstance(selected_canonical_path, str):
            self._note_field_rule_hit(
                field_confidence,
                f"{selected_canonical_path}.amount",
                "counted as canonical medicine ledger source",
            )
            price_counted_in_ledger.append(
                {
                    "field_path": f"{selected_canonical_path}.amount",
                    "amount": coerce_numeric_value(canonical_source.get("amount") if isinstance(canonical_source, dict) else None),
                    "counted": True,
                }
            )

        for dropped_index in dropped_group_indices:
            if dropped_index >= len(selected_other_charge_groups):
                continue
            dropped_group = selected_other_charge_groups[dropped_index]
            path_prefix = dropped_group.get("path_prefix")
            if not isinstance(path_prefix, str):
                continue
            if selected_canonical_path == path_prefix:
                continue
            rule_hit = "suppressed as financial outlier during ledger reconciliation"
            self._note_field_rule_hit(
                field_confidence,
                f"{path_prefix}.amount",
                rule_hit,
                review_flag="constraint_adjusted",
            )
            financial_outlier_suppressions.append(
                {
                    "field_path": f"{path_prefix}.amount",
                    "key": dropped_group.get("key"),
                    "amount": dropped_group.get("amount"),
                    "reason": rule_hit,
                }
            )

        return price_counted_in_ledger, financial_outlier_suppressions

    async def _reconcile_financials(
        self,
        result: AIMedicalScanResponse,
        *,
        images: list[Any],
        field_confidence: dict[str, Any],
        judge_invocations: list[dict[str, Any]],
        list_group_meta: dict[str, Any],
    ) -> dict[str, Any]:
        corrected_fields: list[dict[str, Any]] = []
        unresolved_conflicts: list[str] = []
        reconciliation_notes: list[str] = []
        reconciliation_notes.extend(
            self._prune_noisy_medicines_for_reconciliation(
                result,
                field_confidence=field_confidence,
                list_group_meta=list_group_meta,
            )
        )
        selected_group_meta = cast(dict[str, Any], list_group_meta.get("other_charges", {}))
        selected_groups = cast(list[dict[str, Any]], selected_group_meta.get("selected_groups", []))
        supported_targets = self._extract_supported_numeric_targets(field_confidence)
        medicine_ledger_candidates = self._build_medicine_ledger_candidates(
            result,
            selected_other_charge_groups=selected_groups,
            list_group_meta=list_group_meta,
        )
        medicine_total_from_items = self._sum_defined_numbers([medicine.total_price for medicine in result.medicines])
        pre_reconciliation = {
            "medicine_total_from_items": medicine_total_from_items,
            "selected_other_charges_total": self._sum_defined_numbers([charge.amount for charge in result.other_charges]),
            "summary_total_amount": result.summary.total_amount,
            "financial_recap": result.financial_recap.model_dump(),
        }

        best_subset = self._choose_best_ledger_subset(
            selected_groups,
            medicine_candidates=medicine_ledger_candidates,
            supported_targets=supported_targets,
        )

        if (
            best_subset is not None
            and best_subset.get("coherent", False)
            and self._should_defer_coherent_financial_override(
                result=result,
                best_subset=best_subset,
                supported_targets=supported_targets,
            )
        ):
            best_subset = {
                **best_subset,
                "coherent": False,
                "deferred_by_consensus_guard": True,
            }
            reconciliation_notes.append(
                "Deferred coherent reconciliation override because it materially diverged from stronger consensus totals."
            )
            unresolved_conflicts.append(
                "Financial reconciliation candidate was coherent but materially weaker than the existing consensus totals."
            )

        if best_subset is not None and best_subset.get("coherent", False):
            dropped_indices = cast(list[int], best_subset["dropped_group_indices"])
            for dropped_index in dropped_indices:
                if dropped_index >= len(selected_groups):
                    continue
                path_prefix = selected_groups[dropped_index].get("path_prefix")
                if not isinstance(path_prefix, str) or not path_prefix:
                    continue
                for field_key in [key for key in field_confidence if key.startswith(f"{path_prefix}.")]:
                    field_confidence.pop(field_key, None)
            result.other_charges = [result.other_charges[index] for index in cast(list[int], best_subset["output_indices"])]
            if best_subset.get("dropped_medicine_charge_count", 0):
                reconciliation_notes.append("Pharmacy spending was deduplicated between medicines and other charges.")
            if best_subset.get("dropped_outlier_keys"):
                reconciliation_notes.append("Low-support financial outliers were removed to keep packet totals coherent.")
            price_counted_in_ledger, financial_outlier_suppressions = self._annotate_ledger_selection(
                result=result,
                field_confidence=field_confidence,
                list_group_meta=list_group_meta,
                canonical_source=cast(dict[str, Any] | None, best_subset.get("canonical_medicine_source")),
                dropped_group_indices=dropped_indices,
                selected_other_charge_groups=selected_groups,
            )
            reconciled_field_values = (
                ("financial_recap.total_medicines", best_subset.get("medicine_total")),
                ("financial_recap.total_other_charges", best_subset.get("other_total")),
                ("financial_recap.grand_total", best_subset.get("grand_total")),
                ("summary.total_amount", best_subset.get("grand_total")),
            )
            for field_path, reconciled_value in reconciled_field_values:
                self._apply_reconciled_numeric_field(
                    result,
                    field_path,
                    cast(float | int | None, reconciled_value),
                    field_confidence,
                    corrected_fields,
                    "reconciled from canonical packet ledger",
                )
        else:
            price_counted_in_ledger = []
            financial_outlier_suppressions = []
            for field_path in FINANCIAL_TOTAL_FIELD_SEQUENCE[1:] + FINANCIAL_TOTAL_FIELD_SEQUENCE[:1]:
                await self._maybe_rejudge_field_from_candidates(
                    result,
                    field_path=field_path,
                    images=images,
                    field_confidence=field_confidence,
                    judge_invocations=judge_invocations,
                )
            unresolved_conflicts.append("Financial totals could not be reconciled into a coherent packet ledger.")

        selected_group_meta["reconciliation"] = {
            "supported_targets": supported_targets,
            "selected_subset": best_subset,
            "notes": reconciliation_notes,
        }

        return {
            "derived_financials": {
                "pre_reconciliation": pre_reconciliation,
                "supported_targets": supported_targets,
                "canonical_medicine_ledger_source": best_subset.get("canonical_medicine_source") if best_subset else None,
                "medicine_ledger_candidates": medicine_ledger_candidates,
                "price_counted_in_ledger": price_counted_in_ledger,
                "financial_outlier_suppressions": financial_outlier_suppressions,
                "selected_subset": best_subset,
                "notes": reconciliation_notes,
                "post_reconciliation": {
                    "summary_total_amount": result.summary.total_amount,
                    "financial_recap": result.financial_recap.model_dump(),
                    "other_charges_total": self._sum_defined_numbers([charge.amount for charge in result.other_charges]),
                },
            },
            "corrected_fields": corrected_fields,
            "unresolved_conflicts": list(dict.fromkeys(unresolved_conflicts)),
        }

    def _extract_supported_numeric_targets(self, field_confidence: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
        targets: dict[str, list[dict[str, Any]]] = {}
        for field_path in FINANCIAL_TOTAL_FIELD_SEQUENCE:
            meta = field_confidence.get(field_path)
            if not isinstance(meta, dict):
                continue
            field_targets: list[dict[str, Any]] = []
            for candidate in cast(list[dict[str, Any]], meta.get("candidates", [])):
                numeric_value = coerce_numeric_value(candidate.get("value"))
                if numeric_value is None:
                    continue
                field_targets.append(
                    {
                        "value": numeric_value,
                        "support_count": int(candidate.get("support_count", 0)),
                        "producer_count": int(candidate.get("producer_count", 0)),
                    }
                )
            if field_targets:
                targets[field_path] = field_targets
        return targets

    def _score_numeric_match(self, value: float | int | None, candidates: list[dict[str, Any]]) -> tuple[bool, float]:
        numeric_value = coerce_numeric_value(value)
        if numeric_value is None:
            return False, 0.0
        for candidate in candidates:
            candidate_value = coerce_numeric_value(candidate.get("value"))
            if candidate_value is not None and abs(float(candidate_value) - float(numeric_value)) < 0.01:
                return True, float(candidate.get("support_count", 0)) + (0.2 * float(candidate.get("producer_count", 0)))
        return False, 0.0

    def _should_defer_coherent_financial_override(
        self,
        *,
        result: AIMedicalScanResponse,
        best_subset: dict[str, Any],
        supported_targets: dict[str, list[dict[str, Any]]],
    ) -> bool:
        current_grand_total = coerce_numeric_value(result.financial_recap.grand_total)
        candidate_grand_total = coerce_numeric_value(best_subset.get("grand_total"))
        if current_grand_total is None or candidate_grand_total is None:
            return False

        relative_delta = abs(float(candidate_grand_total) - float(current_grand_total)) / max(abs(float(current_grand_total)), 1.0)
        if relative_delta < RECONCILIATION_LARGE_OVERRIDE_RATIO_THRESHOLD:
            return False

        target_candidates = [
            *supported_targets.get("summary.total_amount", []),
            *supported_targets.get("financial_recap.grand_total", []),
        ]
        _, current_support_score = self._score_numeric_match(current_grand_total, target_candidates)
        _, candidate_support_score = self._score_numeric_match(candidate_grand_total, target_candidates)
        if current_support_score <= 0.0:
            return False

        canonical_source = cast(dict[str, Any], best_subset.get("canonical_medicine_source") or {})
        canonical_source_support = float(canonical_source.get("support_count", 0)) + (0.2 * float(canonical_source.get("producer_count", 0)))
        dropped_count = len(cast(list[int], best_subset.get("dropped_group_indices", [])))

        return (
            candidate_support_score <= current_support_score
            and canonical_source_support <= RECONCILIATION_WEAK_SOURCE_SUPPORT_THRESHOLD
            and dropped_count >= RECONCILIATION_HEAVY_DROP_COUNT_THRESHOLD
        )

    def _choose_best_ledger_subset(
        self,
        selected_groups: list[dict[str, Any]],
        *,
        medicine_candidates: list[dict[str, Any]],
        supported_targets: dict[str, list[dict[str, Any]]],
    ) -> dict[str, Any] | None:
        if not selected_groups and not medicine_candidates:
            return None

        best_choice: dict[str, Any] | None = None
        non_medicine_indices = [index for index, group in enumerate(selected_groups) if group.get("bucket") != "medicine"]
        require_grand_total_match = bool(supported_targets.get("summary.total_amount") or supported_targets.get("financial_recap.grand_total"))
        candidate_sources = medicine_candidates or [
            {
                "amount": None,
                "source_type": "none",
                "source_key": "none",
                "score": 0.0,
                "support_count": 0,
                "producer_count": 0,
            }
        ]

        for medicine_candidate in candidate_sources:
            subset_medicine_total = coerce_numeric_value(medicine_candidate.get("amount"))
            for subset_size in range(len(non_medicine_indices) + 1):
                for subset_indices_tuple in combinations(non_medicine_indices, subset_size):
                    subset_indices = list(subset_indices_tuple)

                    subset_groups = [selected_groups[index] for index in subset_indices]
                    subset_other_total = self._sum_defined_numbers(
                        [cast(float | int | None, group.get("amount")) for group in subset_groups]
                    )
                    subset_grand_total = self._sum_defined_numbers([subset_medicine_total, subset_other_total])
                    output_indices = list(subset_indices)

                    total_match, total_match_score = self._score_numeric_match(
                        subset_grand_total,
                        [
                            *supported_targets.get("summary.total_amount", []),
                            *supported_targets.get("financial_recap.grand_total", []),
                        ],
                    )
                    medicine_match, medicine_match_score = self._score_numeric_match(
                        subset_medicine_total,
                        supported_targets.get("financial_recap.total_medicines", []),
                    )
                    other_match, other_match_score = self._score_numeric_match(
                        subset_other_total,
                        supported_targets.get("financial_recap.total_other_charges", []),
                    )

                    coherent = total_match or not require_grand_total_match
                    support_score = float(medicine_candidate.get("score", 0.0)) + sum(float(group.get("score", 0.0)) for group in subset_groups)
                    dropped_indices = [index for index in range(len(selected_groups)) if index not in subset_indices]
                    canonical_source_support = float(medicine_candidate.get("support_count", 0)) + (0.2 * float(medicine_candidate.get("producer_count", 0)))
                    weak_source_penalty = 0.4 if canonical_source_support < RECONCILIATION_WEAK_SOURCE_SUPPORT_THRESHOLD else 0.0
                    dropped_count_penalty = 0.5 * len(dropped_indices)
                    candidate_choice = {
                        "coherent": coherent,
                        "selected_group_indices": subset_indices,
                        "output_indices": output_indices,
                        "medicine_total": subset_medicine_total,
                        "other_total": subset_other_total,
                        "grand_total": subset_grand_total,
                        "canonical_medicine_source": medicine_candidate,
                        "dropped_group_indices": dropped_indices,
                        "dropped_outlier_keys": [
                            selected_groups[index]["key"]
                            for index in dropped_indices
                            if selected_groups[index].get("bucket") != "medicine"
                        ],
                        "dropped_medicine_charge_count": len(
                            [index for index in dropped_indices if selected_groups[index].get("bucket") == "medicine"]
                        ),
                        "score_tuple": (
                            1 if coherent else 0,
                            1 if total_match else 0,
                            total_match_score,
                            1 if medicine_match else 0,
                            medicine_match_score,
                            1 if other_match else 0,
                            other_match_score,
                            canonical_source_support,
                            support_score - weak_source_penalty,
                            -dropped_count_penalty,
                        ),
                    }
                    if best_choice is None or candidate_choice["score_tuple"] > best_choice["score_tuple"]:
                        best_choice = candidate_choice

        return best_choice

    def _apply_reconciled_numeric_field(
        self,
        result: AIMedicalScanResponse,
        field_path: str,
        new_value: float | int | None,
        field_confidence: dict[str, Any],
        corrected_fields: list[dict[str, Any]],
        reason: str,
    ) -> None:
        if new_value is None:
            return

        original_value = self._get_field_value(result, field_path)
        unchanged_within_tolerance = original_value is not None and abs(float(original_value) - float(new_value)) < 0.01

        if not unchanged_within_tolerance:
            self._set_field_value(result, field_path, new_value)

        meta = field_confidence.get(field_path)
        if isinstance(meta, dict):
            self._set_meta_selected_value(
                meta,
                field_path,
                new_value,
                min_confidence=0.84,
                rule_hit=reason,
                review_flag="constraint_adjusted",
            )

        if unchanged_within_tolerance:
            return

        corrected_fields.append(
            {
                "field_path": field_path,
                "original_value": original_value,
                "corrected_value": new_value,
                "reason": reason,
            }
        )

    def _determine_other_charge_bucket_targets(self, traces: list[SuccessfulProducerInferenceTrace]) -> dict[str, int]:
        bucket_counts: dict[str, list[int]] = {bucket: [] for bucket in OTHER_CHARGE_BUCKET_ORDER}
        for trace in traces:
            counts = {bucket: 0 for bucket in OTHER_CHARGE_BUCKET_ORDER}
            for charge in trace.payload.other_charges:
                raw_kind = classify_other_charge_kind(charge.category, charge.description)
                counts[normalize_other_charge_bucket(raw_kind)] += 1
            for bucket in OTHER_CHARGE_BUCKET_ORDER:
                bucket_counts[bucket].append(counts[bucket])
        return {bucket: choose_target_list_count(counts) for bucket, counts in bucket_counts.items()}

    def _determine_list_target_count(
        self,
        list_name: str,
        *,
        traces: list[SuccessfulProducerInferenceTrace],
        raw_counts: list[int],
        validation_hints: dict[str, Any],
    ) -> int:
        base_count = choose_target_list_count(raw_counts)
        if list_name != "medicines":
            return base_count

        primary_rank = min((trace.producer_rank for trace in traces), default=0)
        primary_count = max(
            (len(trace.payload.medicines) for trace in traces if trace.producer_rank == primary_rank),
            default=0,
        )
        hinted_prescriptions = sum(
            1
            for document in cast(list[dict[str, Any]], validation_hints.get("ordered_documents", []))
            if document.get("document_type") == "prescription"
        )
        prescription_evidence_count = max(
            (
                len(
                    [
                        medicine
                        for medicine in trace.payload.medicines
                        if getattr(medicine, "name", None) and not is_package_level_medicine_name(getattr(medicine, "name", None))
                    ]
                )
                for trace in traces
                if trace.producer_rank == primary_rank
            ),
            default=0,
        )
        primary_soft_cap = max(
            MEDICINE_PRIMARY_COUNT_SOFT_CAP if hinted_prescriptions else 0,
            base_count + MEDICINE_PRIMARY_COUNT_BUFFER,
        )
        capped_primary_count = min(primary_count, primary_soft_cap) if primary_count > 0 else 0
        capped_prescription_count = min(prescription_evidence_count, primary_soft_cap) if hinted_prescriptions else 0
        evidence_floor = 1 if hinted_prescriptions and max(capped_primary_count, capped_prescription_count) > 0 else 0
        return max(base_count, capped_primary_count, capped_prescription_count, evidence_floor)

    def _select_groups_by_stability(self, ranked_groups: list[dict[str, Any]], target_count: int) -> list[dict[str, Any]]:
        if target_count <= 0:
            return []
        stable_groups = [
            group
            for group in ranked_groups
            if group["run_count"] > 1 or len(group["producers"]) > 1
        ]
        if not stable_groups:
            return ranked_groups[:target_count]

        selected_groups = stable_groups[:target_count]
        selected_keys = {group["key"] for group in selected_groups}
        for group in ranked_groups:
            if len(selected_groups) >= target_count:
                break
            if group["key"] in selected_keys:
                continue
            selected_groups.append(group)
            selected_keys.add(group["key"])
        return selected_groups

    def _should_prefer_split_medicine_group(
        self,
        group: dict[str, Any],
        candidate_groups: list[dict[str, Any]],
    ) -> bool:
        components = cast(list[str], group.get("compound_components", []))
        if len(components) < MEDICINE_SPLIT_COMPONENT_MIN:
            return False

        matched_components = 0
        for component in components:
            if any(
                other_group["key"] != group["key"]
                and not bool(other_group.get("is_grouped_compound"))
                and not bool(other_group.get("is_package_level_medicine"))
                and sequence_similarity(component, cast(str, other_group["key"])) >= 0.84
                for other_group in candidate_groups
            ):
                matched_components += 1

        return matched_components >= MEDICINE_SPLIT_COMPONENT_MIN

    def _select_medicine_groups(
        self,
        ranked_groups: list[dict[str, Any]],
        *,
        target_count: int,
    ) -> tuple[list[dict[str, Any]], dict[str, str]]:
        excluded_keys: dict[str, str] = {}
        candidate_groups: list[dict[str, Any]] = []
        for group in ranked_groups:
            if bool(group.get("is_package_level_medicine")):
                excluded_keys[group["key"]] = "suppressed generic/package-level medicine artifact"
                continue
            candidate_groups.append(group)

        filtered_groups: list[dict[str, Any]] = []
        for group in candidate_groups:
            if bool(group.get("is_grouped_compound")) and self._should_prefer_split_medicine_group(group, candidate_groups):
                excluded_keys[group["key"]] = "preferred split ingredient extraction over grouped compound"
                continue
            filtered_groups.append(group)
        candidate_groups = filtered_groups

        if target_count <= 0 or not candidate_groups:
            return [], excluded_keys

        stable_primary_groups = [
            group for group in candidate_groups if bool(group.get("has_primary_support")) and (group["run_count"] > 1 or len(group["producers"]) > 1)
        ]
        stable_secondary_groups = [
            group for group in candidate_groups if not bool(group.get("has_primary_support")) and (group["run_count"] > 1 or len(group["producers"]) > 1)
        ]
        primary_fallback_groups = [
            group for group in candidate_groups if bool(group.get("has_primary_support")) and group not in stable_primary_groups
        ]
        secondary_fallback_groups = [
            group
            for group in candidate_groups
            if group not in stable_primary_groups and group not in stable_secondary_groups and group not in primary_fallback_groups
        ]

        selected_groups: list[dict[str, Any]] = []
        selected_keys: set[str] = set()
        for bucket in (stable_primary_groups, stable_secondary_groups, primary_fallback_groups, secondary_fallback_groups):
            for group in bucket:
                if len(selected_groups) >= target_count or group["key"] in selected_keys:
                    continue
                selected_groups.append(group)
                selected_keys.add(group["key"])
            if len(selected_groups) >= target_count:
                break

        return selected_groups, excluded_keys

    def _select_other_charge_groups(
        self,
        ranked_groups: list[dict[str, Any]],
        *,
        target_count: int,
        bucket_targets: dict[str, int],
    ) -> list[dict[str, Any]]:
        reconciliation_buffer = 3
        selected_groups: list[dict[str, Any]] = []
        selected_keys: set[str] = set()
        for bucket in OTHER_CHARGE_BUCKET_ORDER:
            bucket_count = int(bucket_targets.get(bucket, 0))
            if bucket_count <= 0:
                continue
            bucket_groups = [group for group in ranked_groups if group.get("bucket") == bucket]
            for group in self._select_groups_by_stability(bucket_groups, bucket_count):
                if group["key"] in selected_keys:
                    continue
                selected_keys.add(group["key"])
                selected_groups.append(group)

        if not selected_groups:
            return self._select_groups_by_stability(ranked_groups, target_count + reconciliation_buffer)

        remaining_groups = [group for group in ranked_groups if group["key"] not in selected_keys]
        while len(selected_groups) < target_count + reconciliation_buffer and remaining_groups:
            next_group = remaining_groups.pop(0)
            selected_keys.add(next_group["key"])
            selected_groups.append(next_group)
        return selected_groups

    def _serialize_list_group_meta(
        self,
        list_name: str,
        ranked_groups: list[dict[str, Any]],
        selected_groups: list[dict[str, Any]],
        *,
        target_count: int,
        bucket_targets: dict[str, int] | None,
        excluded_keys: dict[str, str],
    ) -> dict[str, Any]:
        selected_keys = [group["key"] for group in selected_groups]
        selected_key_set = set(selected_keys)
        return {
            "target_count": target_count,
            "bucket_targets": bucket_targets,
            "groups": [
                {
                    "key": group["key"],
                    "bucket": group.get("bucket"),
                    "raw_kind": group.get("raw_kind"),
                    "support_count": group["run_count"],
                    "producer_count": len(group["producers"]),
                    "score": round(float(group["score"]), 4),
                    "selected": group["key"] in selected_key_set,
                    "excluded_reason": excluded_keys.get(group["key"]),
                    "has_primary_support": bool(group.get("has_primary_support")),
                    "is_package_level_medicine": bool(group.get("is_package_level_medicine")),
                    "is_grouped_compound": bool(group.get("is_grouped_compound")),
                    "representative_value": self._serialize_group_representative(list_name, group),
                }
                for group in ranked_groups
            ],
            "selected_groups": [
                {
                    "output_index": index,
                    "key": group["key"],
                    "bucket": group.get("bucket"),
                    "raw_kind": group.get("raw_kind"),
                    "amount": self._serialize_group_representative(list_name, group).get("amount"),
                    "score": round(float(group["score"]), 4),
                    "support_count": group["run_count"],
                    "producer_count": len(group["producers"]),
                    "has_primary_support": bool(group.get("has_primary_support")),
                    "is_package_level_medicine": bool(group.get("is_package_level_medicine")),
                    "is_grouped_compound": bool(group.get("is_grouped_compound")),
                    "representative_value": self._serialize_group_representative(list_name, group),
                }
                for index, group in enumerate(selected_groups)
            ],
        }

    def _serialize_group_representative(self, list_name: str, group: dict[str, Any]) -> dict[str, Any]:
        if not group.get("items"):
            return {}
        _, representative_item = cast(tuple[SuccessfulProducerInferenceTrace, Any], group["items"][0])
        if list_name == "other_charges":
            return {
                "category": getattr(representative_item, "category", None),
                "description": getattr(representative_item, "description", None),
                "amount": coerce_numeric_value(getattr(representative_item, "amount", None)),
            }
        if list_name == "medicines":
            return {
                "name": getattr(representative_item, "name", None),
                "total_price": coerce_numeric_value(getattr(representative_item, "total_price", None)),
            }
        return {
            "description": getattr(representative_item, "description", None),
        }

    def _order_detected_document_groups(
        self,
        groups: list[dict[str, Any]],
        validation_hints: dict[str, Any],
    ) -> list[dict[str, Any]]:
        hint_documents = cast(list[dict[str, Any]], validation_hints.get("ordered_documents", []))
        remaining_positions: dict[str, list[int]] = {}
        for position, document in enumerate(hint_documents):
            document_type = document.get("document_type")
            if not isinstance(document_type, str):
                continue
            remaining_positions.setdefault(document_type, []).append(position)

        annotated_groups: list[dict[str, Any]] = []
        for group in sorted(groups, key=lambda item: (int(item.get("primary_position", 10_000)), -float(item.get("score", 0.0)))):
            inferred_type = cast(str, group.get("inferred_type", "unknown"))
            hint_position = None
            if remaining_positions.get(inferred_type):
                hint_position = remaining_positions[inferred_type].pop(0)
            annotated_groups.append({**group, "hint_position": hint_position})

        return sorted(
            annotated_groups,
            key=lambda group: (
                10_000 if group.get("hint_position") is None else int(group["hint_position"]),
                int(group.get("primary_position", 10_000)),
                -float(group.get("score", 0.0)),
                cast(str, group["raw_value"]),
            ),
        )

    def _extract_other_charge_raw_kind(self, group_key: str) -> str:
        if ":" not in group_key:
            return "other"
        raw_kind, _ = group_key.split(":", 1)
        return raw_kind or "other"

    def _select_primary_document_sequence(
        self,
        traces: list[SuccessfulProducerInferenceTrace],
    ) -> list[dict[str, Any]]:
        primary_rank = min((trace.producer_rank for trace in traces), default=0)
        primary_traces = [trace for trace in traces if trace.producer_rank == primary_rank]
        if not primary_traces:
            return []
        selected_trace = max(
            primary_traces,
            key=lambda trace: (len(trace.payload.summary.detected_documents), -trace.run_index),
        )
        selected_sequence: list[dict[str, Any]] = []
        seen_keys: set[str] = set()
        for position, label in enumerate(selected_trace.payload.summary.detected_documents):
            normalized_key = normalize_candidate_key("summary.detected_documents", label)
            if normalized_key in seen_keys:
                continue
            seen_keys.add(normalized_key)
            selected_sequence.append(
                {
                    "raw_value": normalize_whitespace(label),
                    "normalized_key": normalized_key,
                    "inferred_type": infer_document_type_from_label(label),
                    "primary_position": position,
                    "trace": selected_trace,
                }
            )
        return selected_sequence

    def _build_detected_document_support(
        self,
        traces: list[SuccessfulProducerInferenceTrace],
    ) -> dict[str, dict[str, Any]]:
        support: dict[str, dict[str, Any]] = {}
        for trace in traces:
            for position, document_label in enumerate(trace.payload.summary.detected_documents):
                normalized_key = normalize_candidate_key("summary.detected_documents", document_label)
                meta = support.setdefault(
                    normalized_key,
                    {
                        "raw_value": normalize_whitespace(document_label),
                        "normalized_key": normalized_key,
                        "inferred_type": infer_document_type_from_label(document_label),
                        "producers": set(),
                        "producer_ranks": {},
                        "run_count": 0,
                        "positions": [],
                    },
                )
                meta["producers"].add(trace.producer_id)
                meta["producer_ranks"][trace.producer_id] = trace.producer_rank
                meta["run_count"] += 1
                cast(list[tuple[int, int]], meta["positions"]).append((trace.producer_rank, position))
        for meta in support.values():
            meta["score"] = score_candidate_group(meta["producer_ranks"], meta["producers"], meta["run_count"])
        return support

    def _compute_detected_document_meta(
        self,
        *,
        selected_docs: list[str],
        support_by_key: dict[str, dict[str, Any]],
        rule_hits: list[str],
        review_flags: list[str],
        judge_used: bool,
        judge_reason: str | None,
        total_producers: int,
    ) -> dict[str, Any]:
        selected_support = [
            support_by_key[normalize_candidate_key("summary.detected_documents", label)]
            for label in selected_docs
            if normalize_candidate_key("summary.detected_documents", label) in support_by_key
        ]
        if selected_support:
            avg_run_support = sum(int(item["run_count"]) for item in selected_support) / len(selected_support)
            avg_producer_support = sum(len(cast(set[str], item["producers"])) for item in selected_support) / len(selected_support)
            run_ratio = avg_run_support / max(len(selected_support), 1)
            producer_ratio = avg_producer_support / max(total_producers, 1)
            confidence = clamp_confidence((0.65 * producer_ratio) + (0.20 * run_ratio) + (0.15 * 1.0))
            support_count = sum(int(item["run_count"]) for item in selected_support)
            producer_agreement = clamp_confidence(producer_ratio)
        else:
            confidence = 0.0
            support_count = 0
            producer_agreement = 0.0
        return {
            "value": selected_docs,
            "confidence": confidence,
            "support": [],
            "alternates": [],
            "candidates": [],
            "judge_used": judge_used,
            "judge_reason": judge_reason,
            "rule_hits": rule_hits,
            "review_flags": review_flags,
            "calibration_version": CALIBRATION_VERSION,
            "support_count": support_count,
            "producer_agreement": producer_agreement,
        }

    async def _fuse_detected_documents(
        self,
        traces: list[SuccessfulProducerInferenceTrace],
        images: list[Any],
        judge_invocations: list[dict[str, Any]],
        *,
        validation_hints: dict[str, Any],
        inventory_conflicts: list[str],
    ) -> tuple[list[str], dict[str, Any]]:
        support_by_key = self._build_detected_document_support(traces)
        primary_sequence = self._select_primary_document_sequence(traces)
        total_producers = len({trace.producer_id for trace in traces})
        if not primary_sequence:
            return [], self._compute_detected_document_meta(
                selected_docs=[],
                support_by_key=support_by_key,
                rule_hits=[],
                review_flags=[],
                judge_used=False,
                judge_reason=None,
                total_producers=total_producers,
            )

        base_docs = [cast(str, candidate["raw_value"]) for candidate in primary_sequence]
        hint_documents = cast(list[dict[str, Any]], validation_hints.get("ordered_documents", []))
        if not hint_documents:
            return base_docs, self._compute_detected_document_meta(
                selected_docs=base_docs,
                support_by_key=support_by_key,
                rule_hits=["preserved primary packet inventory order"],
                review_flags=[],
                judge_used=False,
                judge_reason=None,
                total_producers=total_producers,
            )

        desired_type_counts: dict[str, int] = {}
        for hint_document in hint_documents:
            desired_type = hint_document.get("document_type")
            if not isinstance(desired_type, str):
                continue
            desired_type_counts[desired_type] = desired_type_counts.get(desired_type, 0) + 1

        selected_groups = [dict(candidate) for candidate in primary_sequence]
        selected_keys = {cast(str, candidate["normalized_key"]) for candidate in selected_groups}
        selected_type_counts: dict[str, int] = {}
        for candidate in selected_groups:
            inferred_type = cast(str, candidate.get("inferred_type", "unknown"))
            selected_type_counts[inferred_type] = selected_type_counts.get(inferred_type, 0) + 1

        supported_candidates = sorted(
            support_by_key.values(),
            key=lambda item: (
                -float(item.get("score", 0.0)),
                -len(cast(set[str], item["producers"])),
                min(cast(list[tuple[int, int]], item.get("positions", [(10_000, 10_000)]))),
                cast(str, item["raw_value"]),
            ),
        )
        slot_conflicts: list[str] = []
        advisory_expansion_used = False
        for desired_type, desired_count in desired_type_counts.items():
            while selected_type_counts.get(desired_type, 0) < desired_count:
                supported_match = next(
                    (
                        candidate
                        for candidate in supported_candidates
                        if candidate.get("inferred_type") == desired_type
                        and cast(str, candidate["normalized_key"]) not in selected_keys
                        and (
                            len(cast(set[str], candidate["producers"])) > 1
                            or int(candidate.get("run_count", 0)) > 1
                        )
                    ),
                    None,
                )
                if supported_match is None:
                    break
                advisory_expansion_used = True
                selected_groups.append(
                    {
                        **supported_match,
                        "primary_position": min(
                            (position for _, position in cast(list[tuple[int, int]], supported_match.get("positions", []))),
                            default=10_000,
                        ),
                    }
                )
                selected_keys.add(cast(str, supported_match["normalized_key"]))
                selected_type_counts[desired_type] = selected_type_counts.get(desired_type, 0) + 1

        final_docs = [cast(str, group["raw_value"]) for group in self._order_detected_document_groups(selected_groups, validation_hints)]
        if len(final_docs) < len(hint_documents):
            slot_conflicts.append(
                f"Validation detected {len(hint_documents)} documents, but extraction preserved {len(final_docs)} distinct labels after duplicate collapse."
            )

        judge_used = False
        judge_reason: str | None = None
        if advisory_expansion_used and slot_conflicts and final_docs != base_docs[: len(final_docs)]:
            selected_key, judge_reason = await self._judge_field(
                "summary.detected_documents",
                images,
                [
                    {
                        "normalized_key": "validation_advisory",
                        "raw_value": final_docs,
                        "run_count": len(final_docs),
                        "producers": set(range(max(total_producers, 1))),
                    },
                    {
                        "normalized_key": "primary_order",
                        "raw_value": base_docs[: len(final_docs)],
                        "run_count": len(base_docs[: len(final_docs)]),
                        "producers": {0},
                    },
                ],
            )
            if selected_key == "primary_order":
                final_docs = base_docs[: len(final_docs)]
            judge_used = selected_key in {"validation_advisory", "primary_order"}
            if judge_used:
                judge_invocations.append(
                    {
                        "field_path": "summary.detected_documents",
                        "selected_candidate": final_docs,
                        "reason": judge_reason,
                    }
                )

        inventory_conflicts.extend(slot_conflicts)
        return final_docs, self._compute_detected_document_meta(
            selected_docs=final_docs,
            support_by_key=support_by_key,
            rule_hits=["preserved packet inventory with advisory validation ordering"],
            review_flags=["inventory_advisory_reordered"] if final_docs != base_docs[: len(final_docs)] else [],
            judge_used=judge_used,
            judge_reason=judge_reason,
            total_producers=total_producers,
        )

    def _build_list_item_key(self, list_name: str, item: Any) -> str:
        if list_name == "diagnoses":
            return normalize_candidate_key("diagnoses.description", getattr(item, "description", None))

        if list_name == "medicines":
            name = getattr(item, "name", None)
            return normalize_candidate_key("medicines.name", name)

        amount = coerce_numeric_value(getattr(item, "amount", None))
        category = getattr(item, "category", None)
        description = getattr(item, "description", None)
        charge_kind = classify_other_charge_kind(category, description)
        descriptor_key = extract_other_charge_descriptor_signature(category, description)
        amount_key = "__null__" if amount is None else str(amount)
        return f"{charge_kind}:{amount_key}:{descriptor_key}"

    def _find_similar_group_key(self, list_name: str, candidate_key: str, existing_keys: list[str]) -> str | None:
        if list_name == "other_charges":
            candidate_kind, candidate_amount, candidate_descriptor = parse_other_charge_group_key(candidate_key)
            for existing_key in existing_keys:
                existing_kind, existing_amount, existing_descriptor = parse_other_charge_group_key(existing_key)
                if candidate_amount != existing_amount:
                    continue
                if not other_charge_kinds_share_family(candidate_kind, existing_kind):
                    continue
                if not candidate_descriptor or not existing_descriptor:
                    continue
                if (
                    candidate_descriptor == existing_descriptor
                    or candidate_descriptor in existing_descriptor
                    or existing_descriptor in candidate_descriptor
                    or sequence_similarity(candidate_descriptor, existing_descriptor) >= 0.88
                ):
                    return existing_key
            return None

        threshold = 0.9 if list_name == "diagnoses" else 0.86 if list_name == "medicines" else None
        if threshold is None:
            return None

        if list_name == "medicines" and is_generic_medicine_key(candidate_key):
            return None

        for existing_key in existing_keys:
            if list_name == "medicines" and is_generic_medicine_key(existing_key):
                continue
            if sequence_similarity(candidate_key, existing_key) >= threshold:
                return existing_key
        return None

    def _should_exclude_other_charge_group(self, group_key: str, exclude_amount_keys: set[str]) -> bool:
        if normalize_other_charge_bucket(self._extract_other_charge_raw_kind(group_key)) != "medicine":
            return False

        _, amount_text, _ = parse_other_charge_group_key(group_key)
        return amount_text in exclude_amount_keys if amount_text is not None else False
