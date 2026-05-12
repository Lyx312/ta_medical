from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy.orm import Session

from app.core.ai_pipeline_logging import ai_pipeline_exception_fields, ai_pipeline_log
from app.core.config import ROOT_DIR, get_settings
from app.repositories.claim_medicine_knowledge_repository import ClaimMedicineKnowledgeRepository
from app.schemas.ai import AIMedicalScanResponse, AIScanMedicine
from app.services.ai_service import BaseDocumentAIProcessor, DocumentImageInput
from app.services.claim_medicine_relevance import normalize_medicine_name_for_matching

IMAGE_SCORE_DOMINANCE_MARGIN = 0.12
MAX_BATCH_ITEMS = 6
MAX_GROUP_SUGGESTIONS = 5
MAX_DB_HINTS = 20
AUTO_CORRECT_INTERPRETATION_CONFIDENCE = 0.92
AUTO_CORRECT_SIMILARITY_FLOOR = 0.62
AUTO_CORRECT_RELEVANCE_CONFIDENCE_FLOOR = 0.55
MEDICINE_IMAGE_INTERPRETATION_PROMPT_FILE_PATH = (
    ROOT_DIR / "app" / "prompts" / "ai" / "medical_claim_medicine_image_interpretation.yaml"
)


@dataclass(frozen=True)
class _MedicinePriorityRecord:
    original_index: int
    medicine: AIScanMedicine
    image_score: float
    support_count: int
    producer_count: int
    relevance_bucket: int
    relevance_label: str
    relevance_confidence: float
    relevance_source: str
    interpreted_name: str | None = None
    interpretation_confidence: float = 0.0
    interpretation_reason: str | None = None


class BatchMedicineInterpretationJudgement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    original_index: int = Field(ge=0)
    interpreted_name: str | None = None
    interpretation_confidence: float = Field(ge=0, le=1)
    interpretation_reason: str = Field(min_length=1)
    relevance: str = Field(pattern="^(relevant|uncertain|not_relevant)$")
    relevance_confidence: float = Field(ge=0, le=1)
    matched_diagnosis: bool
    matched_anamnesis: bool


class BatchMedicineInterpretationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    judgements: list[BatchMedicineInterpretationJudgement] = Field(default_factory=list)


class ScanMedicineImageInterpretationJudge(BaseDocumentAIProcessor):
    def __init__(self, *, prompt_file_path: Path | None = None) -> None:
        super().__init__(
            settings=get_settings(),
            prompt_file_path=prompt_file_path or MEDICINE_IMAGE_INTERPRETATION_PROMPT_FILE_PATH,
        )

    async def judge_batch(
        self,
        *,
        images: list[DocumentImageInput],
        diagnosis_text: str,
        anamnesis_text: str,
        ambiguous_medicines: list[dict[str, Any]],
        ranked_group_suggestions: list[dict[str, Any]],
        db_medicine_hints: list[dict[str, Any]],
        ai_call_context: dict[str, Any] | None = None,
    ) -> BatchMedicineInterpretationResponse:
        if not ambiguous_medicines:
            return BatchMedicineInterpretationResponse(judgements=[])

        prompt_bundle = self._load_prompt_bundle()
        model_spec = self.get_primary_enabled_model_spec()
        model = self._create_chat_model(model_spec)
        ai_pipeline_log(
            event="medicine_prompt_started",
            component="scan_medicine_prioritization",
            scan_log_id=ai_call_context.get("scan_log_id") if isinstance(ai_call_context, dict) else None,
            user_id=ai_call_context.get("user_id") if isinstance(ai_call_context, dict) else None,
            provider=model_spec.provider,
            model=model_spec.model,
            call_kind="medicine_prioritization",
            image_count=len(images),
            ambiguous_medicine_count=len(ambiguous_medicines),
            group_suggestion_count=len(ranked_group_suggestions),
            db_hint_count=len(db_medicine_hints),
        )
        try:
            template_input = {
                "diagnosis_text_json": json.dumps(diagnosis_text, ensure_ascii=False),
                "anamnesis_text_json": json.dumps(anamnesis_text, ensure_ascii=False),
                "ambiguous_medicines_json": json.dumps(ambiguous_medicines, ensure_ascii=False, indent=2),
                "ranked_group_suggestions_json": json.dumps(ranked_group_suggestions, ensure_ascii=False, indent=2),
                "db_medicine_hints_json": json.dumps(db_medicine_hints, ensure_ascii=False, indent=2),
                "output_contract_json": json.dumps(prompt_bundle.output_contract, indent=2),
            }

            system_prompt = PromptTemplate.from_template(prompt_bundle.system_template).format(**template_input)
            user_prompt = PromptTemplate.from_template(prompt_bundle.user_template).format(**template_input)

            response, _tracking_payload = await self._ainvoke_with_tracking(
                model,
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=self._build_image_messages(user_prompt, images)),
                ],
                provider=model_spec.provider,
                model_name=model_spec.model,
                call_kind="medicine_prioritization",
                success_context={
                    **(ai_call_context or {}),
                    "temperature": model_spec.temperature,
                    "request_context_payload": {
                        "image_count": len(images),
                        "ambiguous_medicine_count": len(ambiguous_medicines),
                        "group_suggestion_count": len(ranked_group_suggestions),
                        "db_hint_count": len(db_medicine_hints),
                    },
                },
            )
            payload = self._parse_ai_response_to_json(response.content)
            result = BatchMedicineInterpretationResponse.model_validate(payload)
            ai_pipeline_log(
                event="medicine_prompt_completed",
                component="scan_medicine_prioritization",
                scan_log_id=ai_call_context.get("scan_log_id") if isinstance(ai_call_context, dict) else None,
                user_id=ai_call_context.get("user_id") if isinstance(ai_call_context, dict) else None,
                provider=model_spec.provider,
                model=model_spec.model,
                call_kind="medicine_prioritization",
                judgement_count=len(result.judgements),
                success=True,
            )
            return result
        except Exception as exc:
            ai_pipeline_log(
                event="medicine_prompt_failed",
                component="scan_medicine_prioritization",
                scan_log_id=ai_call_context.get("scan_log_id") if isinstance(ai_call_context, dict) else None,
                user_id=ai_call_context.get("user_id") if isinstance(ai_call_context, dict) else None,
                provider=model_spec.provider,
                model=model_spec.model,
                call_kind="medicine_prioritization",
                success=False,
                **ai_pipeline_exception_fields(exc),
            )
            raise RuntimeError("Batch medicine interpretation payload is invalid.") from exc


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _build_diagnosis_text(scan_result: AIMedicalScanResponse) -> str:
    diagnoses = [
        (diagnosis.description or "").strip()
        for diagnosis in scan_result.medical_context.diagnoses
        if (diagnosis.description or "").strip()
    ]
    return " ; ".join(diagnoses)


def _build_anamnesis_text(scan_result: AIMedicalScanResponse) -> str:
    return (scan_result.medical_context.anamnesis or "").strip()


def _relevance_bucket(label: str) -> int:
    if label == "relevant":
        return 2
    if label == "uncertain":
        return 1
    return 0


def _relevance_tiebreak_key(record: _MedicinePriorityRecord) -> tuple[int, float, int, int, int]:
    return (
        record.relevance_bucket,
        record.relevance_confidence,
        record.support_count,
        record.producer_count,
        -record.original_index,
    )


def _should_swap_by_relevance(left: _MedicinePriorityRecord, right: _MedicinePriorityRecord) -> bool:
    image_gap = abs(left.image_score - right.image_score)
    if image_gap >= IMAGE_SCORE_DOMINANCE_MARGIN:
        return left.image_score < right.image_score
    return _relevance_tiebreak_key(left) < _relevance_tiebreak_key(right)


def _normalize_for_similarity(value: str | None) -> str:
    if not value:
        return ""
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return "".join(ch for ch in normalized.casefold() if ch.isalnum())


def _name_similarity(left: str | None, right: str | None) -> float:
    normalized_left = _normalize_for_similarity(left)
    normalized_right = _normalize_for_similarity(right)
    if not normalized_left and not normalized_right:
        return 1.0
    if not normalized_left or not normalized_right:
        return 0.0

    left_bigrams = {normalized_left[i : i + 2] for i in range(max(len(normalized_left) - 1, 1))}
    right_bigrams = {normalized_right[i : i + 2] for i in range(max(len(normalized_right) - 1, 1))}
    if not left_bigrams and not right_bigrams:
        return 1.0
    if not left_bigrams or not right_bigrams:
        return 0.0

    intersection = len(left_bigrams & right_bigrams)
    union = len(left_bigrams | right_bigrams)
    if union <= 0:
        return 0.0
    return intersection / union


def _is_auto_correct_allowed(
    *,
    record: _MedicinePriorityRecord,
    judgement: BatchMedicineInterpretationJudgement,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    interpreted_name = (judgement.interpreted_name or "").strip()
    if not interpreted_name:
        reasons.append("missing_interpreted_name")
    if judgement.interpretation_confidence < AUTO_CORRECT_INTERPRETATION_CONFIDENCE:
        reasons.append("low_interpretation_confidence")

    similarity = _name_similarity(record.medicine.name, interpreted_name)
    if similarity < AUTO_CORRECT_SIMILARITY_FLOOR:
        reasons.append("low_name_similarity")

    if judgement.relevance_confidence < AUTO_CORRECT_RELEVANCE_CONFIDENCE_FLOOR:
        reasons.append("low_relevance_confidence")

    if record.image_score < 0.5:
        reasons.append("low_image_support")

    return (len(reasons) == 0, reasons)


def _collect_ambiguous_indices(records: list[_MedicinePriorityRecord]) -> list[int]:
    if len(records) <= 1:
        return []

    image_sorted = sorted(records, key=lambda item: item.image_score, reverse=True)
    ambiguous: list[int] = []
    for idx in range(len(image_sorted) - 1):
        left = image_sorted[idx]
        right = image_sorted[idx + 1]
        if abs(left.image_score - right.image_score) >= IMAGE_SCORE_DOMINANCE_MARGIN:
            continue
        if left.original_index not in ambiguous:
            ambiguous.append(left.original_index)
        if right.original_index not in ambiguous:
            ambiguous.append(right.original_index)

    return ambiguous[:MAX_BATCH_ITEMS]


def _build_db_medicine_hints(db: Session) -> list[dict[str, Any]]:
    repository = ClaimMedicineKnowledgeRepository(db)
    rows = repository.list_recent(limit=MAX_DB_HINTS)
    hints: list[dict[str, Any]] = []
    for row in rows:
        hints.append(
            {
                "medicine_name": row.medicine_name_normalized,
                "support_count": row.support_count,
                "relevant_count": row.relevant_count,
                "uncertain_count": row.uncertain_count,
                "not_relevant_count": row.not_relevant_count,
                "avg_confidence": row.avg_confidence,
                "last_decision_source": row.last_decision_source,
            }
        )
    return hints


def _build_ranked_group_suggestions(ranked_groups: Any, selected_keys: set[str]) -> list[dict[str, Any]]:
    suggestions: list[dict[str, Any]] = []
    if not isinstance(ranked_groups, list):
        return suggestions

    for group in ranked_groups:
        if not isinstance(group, dict):
            continue
        group_key = group.get("key")
        if not isinstance(group_key, str) or group_key in selected_keys:
            continue

        representative = group.get("representative_value") if isinstance(group.get("representative_value"), dict) else {}
        suggestions.append(
            {
                "key": group_key,
                "name": (representative or {}).get("name"),
                "total_price": (representative or {}).get("total_price"),
                "score": _safe_float(group.get("score"), default=0.0),
                "support_count": _safe_int(group.get("support_count"), default=0),
                "producer_count": _safe_int(group.get("producer_count"), default=0),
                "excluded_reason": group.get("excluded_reason"),
            }
        )

    return suggestions[:MAX_GROUP_SUGGESTIONS]


def _coerce_relevance_label(value: str | None) -> str:
    if value in {"relevant", "uncertain", "not_relevant"}:
        return value
    return "uncertain"


def _apply_judgement_to_record(
    record: _MedicinePriorityRecord,
    judgement: BatchMedicineInterpretationJudgement | None,
    *,
    interpretation_mode: str,
) -> tuple[_MedicinePriorityRecord, dict[str, Any]]:
    if judgement is None:
        return record, {
            "original_index": record.original_index,
            "status": "missing_judgement",
            "original_name": record.medicine.name,
        }

    relevance_label = _coerce_relevance_label(judgement.relevance)
    updated_record = _MedicinePriorityRecord(
        original_index=record.original_index,
        medicine=record.medicine,
        image_score=record.image_score,
        support_count=record.support_count,
        producer_count=record.producer_count,
        relevance_bucket=_relevance_bucket(relevance_label),
        relevance_label=relevance_label,
        relevance_confidence=judgement.relevance_confidence,
        relevance_source="llm_batch_image_interpretation",
        interpreted_name=(judgement.interpreted_name or "").strip() or None,
        interpretation_confidence=judgement.interpretation_confidence,
        interpretation_reason=judgement.interpretation_reason,
    )

    log_entry: dict[str, Any] = {
        "original_index": record.original_index,
        "original_name": record.medicine.name,
        "interpreted_name": updated_record.interpreted_name,
        "interpretation_confidence": judgement.interpretation_confidence,
        "interpretation_reason": judgement.interpretation_reason,
        "relevance": relevance_label,
        "relevance_confidence": judgement.relevance_confidence,
        "matched_diagnosis": judgement.matched_diagnosis,
        "matched_anamnesis": judgement.matched_anamnesis,
        "status": "suggested_only",
    }

    if interpretation_mode != "auto_correct_guarded":
        return updated_record, log_entry

    allowed, rejected_reasons = _is_auto_correct_allowed(record=updated_record, judgement=judgement)
    if not allowed:
        log_entry["status"] = "auto_correct_rejected"
        log_entry["auto_correct_rejected_reasons"] = rejected_reasons
        return updated_record, log_entry

    interpreted_name = updated_record.interpreted_name
    if interpreted_name:
        updated_record = _MedicinePriorityRecord(
            original_index=updated_record.original_index,
            medicine=updated_record.medicine.model_copy(update={"name": interpreted_name}),
            image_score=updated_record.image_score,
            support_count=updated_record.support_count,
            producer_count=updated_record.producer_count,
            relevance_bucket=updated_record.relevance_bucket,
            relevance_label=updated_record.relevance_label,
            relevance_confidence=updated_record.relevance_confidence,
            relevance_source=updated_record.relevance_source,
            interpreted_name=updated_record.interpreted_name,
            interpretation_confidence=updated_record.interpretation_confidence,
            interpretation_reason=updated_record.interpretation_reason,
        )
        log_entry["status"] = "auto_correct_applied"

    return updated_record, log_entry


async def prioritize_scan_medicines(
    db: Session,
    scan_result: AIMedicalScanResponse,
    extraction_meta_payload: dict[str, Any] | None,
    *,
    images: list[DocumentImageInput],
    interpretation_mode: str = "suggest_only",
    scan_log_id: int | None = None,
    user_id: int | None = None,
) -> tuple[AIMedicalScanResponse, dict[str, Any]]:
    medicines = list(scan_result.medicines)
    ai_pipeline_log(
        event="medicine_prioritization_started",
        component="scan_medicine_prioritization",
        scan_log_id=scan_log_id,
        user_id=user_id,
        medicine_count=len(medicines),
        interpretation_mode=interpretation_mode,
    )
    if len(medicines) <= 1:
        ai_pipeline_log(
            event="medicine_prioritization_completed",
            component="scan_medicine_prioritization",
            scan_log_id=scan_log_id,
            user_id=user_id,
            medicine_count=len(medicines),
            ambiguous_count=0,
            applied=False,
            success=True,
        )
        return scan_result, {
            "enabled": True,
            "applied": False,
            "reason": "not_enough_medicines",
            "original_order": [medicine.name for medicine in medicines],
            "optimized_order": [medicine.name for medicine in medicines],
            "suggested_medicines_from_document_groups": [],
        }

    list_group_meta = extraction_meta_payload.get("list_group_meta") if isinstance(extraction_meta_payload, dict) else None
    medicines_meta = list_group_meta.get("medicines") if isinstance(list_group_meta, dict) else None
    selected_groups = medicines_meta.get("selected_groups") if isinstance(medicines_meta, dict) else None
    ranked_groups = medicines_meta.get("groups") if isinstance(medicines_meta, dict) else None

    if not isinstance(selected_groups, list) or len(selected_groups) == 0:
        ai_pipeline_log(
            event="medicine_prioritization_completed",
            component="scan_medicine_prioritization",
            scan_log_id=scan_log_id,
            user_id=user_id,
            medicine_count=len(medicines),
            ambiguous_count=0,
            applied=False,
            success=True,
        )
        return scan_result, {
            "enabled": True,
            "applied": False,
            "reason": "missing_image_group_meta",
            "original_order": [medicine.name for medicine in medicines],
            "optimized_order": [medicine.name for medicine in medicines],
            "suggested_medicines_from_document_groups": [],
        }

    group_by_index: dict[int, dict[str, Any]] = {}
    selected_keys: set[str] = set()
    for group in selected_groups:
        if not isinstance(group, dict):
            continue
        output_index = _safe_int(group.get("output_index"), default=-1)
        if output_index >= 0:
            group_by_index[output_index] = group
        key = group.get("key")
        if isinstance(key, str):
            selected_keys.add(key)

    records: list[_MedicinePriorityRecord] = []
    for index, medicine in enumerate(medicines):
        group_meta = group_by_index.get(index, {})
        records.append(
            _MedicinePriorityRecord(
                original_index=index,
                medicine=medicine,
                image_score=_safe_float(group_meta.get("score"), default=0.0),
                support_count=_safe_int(group_meta.get("support_count"), default=0),
                producer_count=_safe_int(group_meta.get("producer_count"), default=0),
                relevance_bucket=1,
                relevance_label="uncertain",
                relevance_confidence=0.5,
                relevance_source="image_priority_default",
            )
        )

    diagnosis_text = _build_diagnosis_text(scan_result)
    anamnesis_text = _build_anamnesis_text(scan_result)
    ambiguous_indices = _collect_ambiguous_indices(records)

    normalized_mode = interpretation_mode.strip() if isinstance(interpretation_mode, str) else "suggest_only"
    interpretation_mode = "auto_correct_guarded" if normalized_mode == "auto_correct_guarded" else "suggest_only"

    judgement_logs: list[dict[str, Any]] = []
    batch_success = False
    batch_error: str | None = None

    if ambiguous_indices:
        ai_pipeline_log(
            event="medicine_prioritization_batch_prepared",
            component="scan_medicine_prioritization",
            scan_log_id=scan_log_id,
            user_id=user_id,
            medicine_count=len(medicines),
            ambiguous_count=len(ambiguous_indices),
            interpretation_mode=interpretation_mode,
        )
        ambiguous_payload: list[dict[str, Any]] = []
        for index in ambiguous_indices:
            record = records[index]
            normalized_name = normalize_medicine_name_for_matching(record.medicine.name)
            ambiguous_payload.append(
                {
                    "original_index": index,
                    "name": record.medicine.name,
                    "normalized_name": normalized_name,
                    "quantity": record.medicine.quantity,
                    "unit": record.medicine.unit,
                    "dosage_instruction": record.medicine.dosage_instruction,
                    "unit_price": record.medicine.unit_price,
                    "total_price": record.medicine.total_price,
                    "image_score": record.image_score,
                    "support_count": record.support_count,
                    "producer_count": record.producer_count,
                }
            )

        ranked_group_suggestions = _build_ranked_group_suggestions(ranked_groups, selected_keys)
        db_medicine_hints = _build_db_medicine_hints(db)

        try:
            judge = ScanMedicineImageInterpretationJudge()
            batch_response = await judge.judge_batch(
                images=images,
                diagnosis_text=diagnosis_text,
                anamnesis_text=anamnesis_text,
                ambiguous_medicines=ambiguous_payload,
                ranked_group_suggestions=ranked_group_suggestions,
                db_medicine_hints=db_medicine_hints,
                ai_call_context={
                    "scan_log_id": scan_log_id,
                    "user_id": user_id,
                },
            )
            judgement_map = {judgement.original_index: judgement for judgement in batch_response.judgements}
            updated_records = records[:]
            for index in ambiguous_indices:
                updated_record, judgement_log = _apply_judgement_to_record(
                    records[index],
                    judgement_map.get(index),
                    interpretation_mode=interpretation_mode,
                )
                updated_records[index] = updated_record
                judgement_logs.append(judgement_log)

            records = updated_records
            batch_success = True
        except Exception as exc:  # noqa: BLE001
            batch_error = str(exc) or exc.__class__.__name__
            ai_pipeline_log(
                event="medicine_prioritization_batch_failed",
                component="scan_medicine_prioritization",
                scan_log_id=scan_log_id,
                user_id=user_id,
                ambiguous_count=len(ambiguous_indices),
                success=False,
                **ai_pipeline_exception_fields(exc),
            )

    sorted_records = records[:]
    changed = True
    while changed:
        changed = False
        for i in range(len(sorted_records) - 1):
            left = sorted_records[i]
            right = sorted_records[i + 1]
            if _should_swap_by_relevance(left, right):
                sorted_records[i], sorted_records[i + 1] = right, left
                changed = True

    sorted_medicines = [record.medicine for record in sorted_records]
    sorted_indices = [record.original_index for record in sorted_records]
    original_order = [medicine.name for medicine in medicines]
    optimized_order = [medicine.name for medicine in sorted_medicines]
    suggested_from_groups = _build_ranked_group_suggestions(ranked_groups, selected_keys)

    prioritized_result = scan_result.model_copy(update={"medicines": sorted_medicines})
    log_payload = {
        "enabled": True,
        "applied": sorted_indices != list(range(len(records))),
        "strategy": {
            "image_score_primary": True,
            "image_score_dominance_margin": IMAGE_SCORE_DOMINANCE_MARGIN,
            "relevance_tiebreak_only": True,
            "reorder_only": True,
            "mode": "batch_image_interpretation",
            "interpretation_mode": interpretation_mode,
            "auto_correct_guarded": True,
        },
        "batch_summary": {
            "ambiguous_indices": ambiguous_indices,
            "batch_item_count": len(ambiguous_indices),
            "batch_success": batch_success,
            "batch_error": batch_error,
        },
        "original_order": original_order,
        "optimized_order": optimized_order,
        "optimized_source_indices": sorted_indices,
        "records": [
            {
                "original_index": record.original_index,
                "medicine_name": record.medicine.name,
                "image_score": record.image_score,
                "support_count": record.support_count,
                "producer_count": record.producer_count,
                "relevance_label": record.relevance_label,
                "relevance_confidence": record.relevance_confidence,
                "relevance_source": record.relevance_source,
                "interpreted_name": record.interpreted_name,
                "interpretation_confidence": record.interpretation_confidence,
                "interpretation_reason": record.interpretation_reason,
            }
            for record in sorted_records
        ],
        "judgement_logs": judgement_logs,
        "suggested_medicines_from_document_groups": suggested_from_groups,
    }
    ai_pipeline_log(
        event="medicine_prioritization_completed",
        component="scan_medicine_prioritization",
        scan_log_id=scan_log_id,
        user_id=user_id,
        medicine_count=len(sorted_medicines),
        ambiguous_count=len(ambiguous_indices),
        applied=sorted_indices != list(range(len(records))),
        batch_success=batch_success,
        success=True,
    )
    return prioritized_result, log_payload
