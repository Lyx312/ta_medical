import base64
import hashlib
import math
from dataclasses import dataclass, replace
from typing import Any, cast

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.api.routes.auth import get_current_user_dependency
from app.core.config import get_settings
from app.db.session import get_db_session
from app.repositories.family_member_repository import FamilyMemberRepository
from app.repositories.scan_log_repository import ScanLogAIRunCreate, ScanLogFileCreate, ScanLogRepository
from app.repositories.user_repository import UserRepository
from app.schemas.ai import AIMedicalScanResponse, AIScanDocumentsResponse
from app.schemas.auth import TokenData
from app.services.ai_ensemble import MedicalExtractionOutcome, ProducerInferenceTrace
from app.services.ai_service import (
    build_validation_packet_hints,
    build_patient_name_verification_error,
    DocumentImageInput,
    DocumentValidationProcessor,
    MedicalDocumentProcessor,
    matches_any_patient_name_candidate,
)
from app.services.image_preprocessing import preprocess_document_image

router = APIRouter(prefix="/ai", tags=["ai"])
processor = MedicalDocumentProcessor(settings=get_settings())
validation_processor = DocumentValidationProcessor(settings=get_settings())


@dataclass(frozen=True)
class BufferedUpload:
    filename: str
    mime_type: str
    content: bytes
    byte_size: int
    sha256_hash: str


def build_patient_name_candidates(user_full_name: str, family_member_names: list[str]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()

    for candidate in [user_full_name, *family_member_names]:
        normalized_candidate = candidate.strip()
        if not normalized_candidate:
            continue

        dedupe_key = normalized_candidate.casefold()
        if dedupe_key in seen:
            continue

        seen.add(dedupe_key)
        names.append(normalized_candidate)

    return names


def normalize_upload_filename(filename: str | None, index: int) -> str:
    candidate = (filename or "").strip()
    if candidate:
        return candidate
    return f"uploaded_file_{index + 1}"


def build_scan_log_error_payload(http_error: HTTPException) -> dict[str, Any]:
    detail = http_error.detail
    if isinstance(detail, dict):
        normalized_detail: Any = detail
    else:
        normalized_detail = {"message": str(detail)}

    return {
        "status_code": http_error.status_code,
        "detail": normalized_detail,
    }


def sanitize_json_compatible(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, list):
        return [sanitize_json_compatible(item) for item in value]
    if isinstance(value, dict):
        return {key: sanitize_json_compatible(item) for key, item in value.items()}
    return value


def normalize_extraction_outcome(
    extraction_result: AIMedicalScanResponse | MedicalExtractionOutcome,
    *,
    fallback_provider: str | None,
    fallback_model: str | None,
) -> MedicalExtractionOutcome:
    if isinstance(extraction_result, MedicalExtractionOutcome):
        return extraction_result

    sanitized_result = AIMedicalScanResponse.model_validate(sanitize_json_compatible(extraction_result.model_dump()))

    return MedicalExtractionOutcome(
        result=sanitized_result,
        overall_confidence=None,
        field_confidence={},
        review_required=False,
        review_reasons=[],
        calibration_version="legacy",
        ai_execution_mode="single_model",
        primary_model_provider=fallback_provider,
        primary_model_name=fallback_model,
        ai_runs=[],
        extraction_meta_payload={
            "fusion_version": "legacy-single-model",
            "calibration_version": "legacy",
            "field_confidence": {},
            "review_required": False,
            "review_reasons": [],
            "overall_confidence": None,
        },
    )


def merge_review_reasons(*reason_groups: list[str]) -> list[str]:
    merged: list[str] = []
    for reasons in reason_groups:
        for reason in reasons:
            normalized_reason = reason.strip()
            if normalized_reason and normalized_reason not in merged:
                merged.append(normalized_reason)
    return merged


def should_run_extraction_name_fallback(validation_payload: dict[str, Any] | None) -> bool:
    if not isinstance(validation_payload, dict):
        return True

    status_value = validation_payload.get("patient_name_match_status")
    if not isinstance(status_value, str):
        return True

    return status_value in {"not_found", "uncertain", "not_checked"}


def resolve_scan_failure_status(
    http_error: HTTPException,
    *,
    validation_payload: dict[str, Any] | None,
) -> str:
    if isinstance(http_error.detail, str) and http_error.detail == "Uploaded file is not a readable image.":
        return "preprocess_failed"
    if isinstance(http_error.detail, dict):
        message = str(http_error.detail.get("message", ""))
        if message == "Document validation failed.":
            return "validation_failed"
        if message == "Patient name verification failed.":
            return "patient_name_mismatch"
    if validation_payload is not None:
        return "extraction_failed"
    return "scan_failed"


def traces_to_ai_run_creates(traces: list[ProducerInferenceTrace] | None) -> list[ScanLogAIRunCreate] | None:
    if not traces:
        return None
    return [
        ScanLogAIRunCreate(
            producer_id=trace.producer_id,
            provider=trace.provider,
            model_name=trace.model,
            producer_rank=trace.producer_rank,
            run_index=trace.run_index,
            temperature=trace.temperature,
            success=trace.success,
            error_message=trace.error,
            payload=trace.payload.model_dump() if trace.payload else None,
        )
        for trace in traces
    ]


async def read_uploaded_files(files: list[UploadFile]) -> list[BufferedUpload]:
    buffered_uploads: list[BufferedUpload] = []
    for index, file in enumerate(files):
        content = await file.read()
        buffered_uploads.append(
            BufferedUpload(
                filename=normalize_upload_filename(file.filename, index),
                mime_type=file.content_type or "application/octet-stream",
                content=content,
                byte_size=len(content),
                sha256_hash=hashlib.sha256(content).hexdigest(),
            )
        )
    return buffered_uploads


@router.post("/scan-documents", response_model=AIScanDocumentsResponse)
async def scan_documents(
    token_data: TokenData = Depends(get_current_user_dependency),
    db: Session = Depends(get_db_session),
    files: list[UploadFile] = File(...),
) -> AIScanDocumentsResponse:
    user_repository = UserRepository(db)
    family_member_repository = FamilyMemberRepository(db)
    scan_log_repository = ScanLogRepository(db)
    user = user_repository.get_by_id(token_data.user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )

    family_members = family_member_repository.get_by_user_id(token_data.user_id)
    buffered_uploads = await read_uploaded_files(files)
    scan_log = scan_log_repository.create_pending(
        user_id=token_data.user_id,
        file_records=[
            ScanLogFileCreate(
                original_filename=upload.filename,
                original_mime_type=upload.mime_type,
                original_byte_size=upload.byte_size,
                original_sha256_hash=upload.sha256_hash,
                original_content=upload.content,
            )
            for upload in buffered_uploads
        ],
    )

    validation_payload: dict[str, Any] | None = None
    extraction_payload: dict[str, Any] | None = None
    extraction_meta_payload: dict[str, Any] | None = None
    ai_runs: list[ProducerInferenceTrace] | None = None
    primary_model_provider: str | None = None
    primary_model_name: str | None = None
    ai_execution_mode: str | None = None

    try:
        images: list[DocumentImageInput] = []
        for upload, scan_log_file in zip(buffered_uploads, scan_log.files):
            processed_image = preprocess_document_image(
                filename=upload.filename,
                content=upload.content,
            )
            scan_log_repository.update_file_processed_content(
                scan_log_file,
                filename=processed_image.filename,
                mime_type=processed_image.mime_type,
                byte_size=len(processed_image.content),
                sha256_hash=hashlib.sha256(processed_image.content).hexdigest(),
                content=processed_image.content,
            )
            images.append(
                DocumentImageInput(
                    filename=processed_image.filename,
                    mime_type=processed_image.mime_type,
                    base64_data=base64.b64encode(processed_image.content).decode("utf-8"),
                )
            )

        patient_name_candidates = build_patient_name_candidates(
            user_full_name=user.full_name,
            family_member_names=[family_member.name for family_member in family_members],
        )
        validation_result = await validation_processor.process(
            images,
            patient_name_candidates=patient_name_candidates,
        )
        validation_payload = validation_result.model_dump()
        validation_hints = build_validation_packet_hints(validation_result)
        if validation_result.patient_name_match_status == "mismatched":
            patient_name_error = build_patient_name_verification_error(
                "Extracted patient name does not match the user or any saved family member. Please review your family members or rescan the document."
            )
            scan_log = scan_log_repository.mark_failed(
                scan_log,
                status="patient_name_mismatch",
                error_payload=build_scan_log_error_payload(patient_name_error),
                validation_payload=validation_payload,
                extraction_payload=extraction_payload,
                ai_execution_mode=ai_execution_mode,
                ai_runs=traces_to_ai_run_creates(ai_runs),
                extraction_meta_payload=extraction_meta_payload,
                primary_model_provider=primary_model_provider,
                primary_model_name=primary_model_name,
            )
            raise patient_name_error

        fallback_model_spec = processor.get_primary_enabled_model_spec()
        primary_model_provider = fallback_model_spec.provider
        primary_model_name = fallback_model_spec.model
        raw_extraction_result = await processor.process(
            images,
            patient_name_candidates=patient_name_candidates,
            validation_hints=validation_hints,
        )
        extraction_result = normalize_extraction_outcome(
            raw_extraction_result,
            fallback_provider=primary_model_provider,
            fallback_model=primary_model_name,
        )
        merged_review_reasons = merge_review_reasons(
            validation_result.review_reasons,
            extraction_result.review_reasons,
        )
        extraction_result = replace(
            extraction_result,
            review_required=bool(merged_review_reasons),
            review_reasons=merged_review_reasons,
            extraction_meta_payload={
                **extraction_result.extraction_meta_payload,
                "review_required": bool(merged_review_reasons),
                "review_reasons": merged_review_reasons,
                "validation_review_required": validation_result.review_required,
                "validation_review_reasons": validation_result.review_reasons,
            },
        )
        extraction_payload = cast(dict[str, Any], sanitize_json_compatible(extraction_result.result.model_dump()))
        extraction_result_payload = AIMedicalScanResponse.model_validate(extraction_payload)
        extraction_meta_payload = extraction_result.extraction_meta_payload
        ai_runs = extraction_result.ai_runs
        primary_model_provider = extraction_result.primary_model_provider
        primary_model_name = extraction_result.primary_model_name
        ai_execution_mode = extraction_result.ai_execution_mode

        extracted_patient_name = extraction_result.result.patient_info.name
        if (
            should_run_extraction_name_fallback(validation_payload)
            and isinstance(extracted_patient_name, str)
            and not matches_any_patient_name_candidate(extracted_patient_name, patient_name_candidates)
        ):
            patient_name_error = build_patient_name_verification_error(
                "Extracted patient name does not match the user or any saved family member. Please review your family members or rescan the document."
            )
            scan_log = scan_log_repository.mark_failed(
                scan_log,
                status="patient_name_mismatch",
                error_payload=build_scan_log_error_payload(patient_name_error),
                validation_payload=validation_payload,
                extraction_payload=extraction_payload,
                ai_execution_mode=ai_execution_mode,
                ai_runs=traces_to_ai_run_creates(ai_runs),
                extraction_meta_payload=extraction_meta_payload,
                primary_model_provider=primary_model_provider,
                primary_model_name=primary_model_name,
            )
            raise patient_name_error

        scan_log = scan_log_repository.mark_succeeded(
            scan_log,
            validation_payload=validation_payload,
            extraction_payload=extraction_payload,
            ai_execution_mode=ai_execution_mode,
            ai_runs=traces_to_ai_run_creates(ai_runs),
            extraction_meta_payload=extraction_meta_payload,
            primary_model_provider=primary_model_provider,
            primary_model_name=primary_model_name,
        )
        return AIScanDocumentsResponse(
            scan_log_id=scan_log.id,
            **extraction_result_payload.model_dump(),
        )
    except HTTPException as exc:
        ai_execution_mode = cast(str | None, getattr(exc, "ai_execution_mode", ai_execution_mode))
        exc_ai_runs = cast(list[ProducerInferenceTrace] | None, getattr(exc, "ai_runs", None))
        if exc_ai_runs is not None:
            ai_runs = exc_ai_runs
        extraction_meta_payload = cast(dict[str, Any] | None, getattr(exc, "extraction_meta_payload", extraction_meta_payload))
        primary_model_provider = cast(str | None, getattr(exc, "primary_model_provider", primary_model_provider))
        primary_model_name = cast(str | None, getattr(exc, "primary_model_name", primary_model_name))
        if scan_log.status == "pending":
            scan_log_repository.mark_failed(
                scan_log,
                status=resolve_scan_failure_status(exc, validation_payload=validation_payload),
                error_payload=build_scan_log_error_payload(exc),
                validation_payload=validation_payload,
                extraction_payload=extraction_payload,
                ai_execution_mode=ai_execution_mode,
                ai_runs=traces_to_ai_run_creates(ai_runs),
                extraction_meta_payload=extraction_meta_payload,
                primary_model_provider=primary_model_provider,
                primary_model_name=primary_model_name,
            )
        raise
    except Exception as exc:
        if scan_log.status == "pending":
            scan_log_repository.mark_failed(
                scan_log,
                status="scan_failed",
                error_payload={
                    "status_code": status.HTTP_500_INTERNAL_SERVER_ERROR,
                    "detail": {"message": str(exc) or "Unexpected scan failure."},
                },
                validation_payload=validation_payload,
                extraction_payload=extraction_payload,
                ai_execution_mode=ai_execution_mode,
                ai_runs=traces_to_ai_run_creates(ai_runs),
                extraction_meta_payload=extraction_meta_payload,
                primary_model_provider=primary_model_provider,
                primary_model_name=primary_model_name,
            )
        raise
