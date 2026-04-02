import base64
import hashlib
from dataclasses import dataclass
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.api.routes.auth import get_current_user_dependency
from app.core.config import get_settings
from app.db.session import get_db_session
from app.repositories.family_member_repository import FamilyMemberRepository
from app.repositories.scan_log_repository import ScanLogFileCreate, ScanLogRepository
from app.repositories.user_repository import UserRepository
from app.schemas.ai import AIScanDocumentsResponse
from app.schemas.auth import TokenData
from app.services.ai_service import (
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
    ai_model_provider: str | None = None
    ai_model_name: str | None = None

    try:
        model_spec = processor.get_primary_enabled_model_spec()
        ai_model_provider = model_spec.provider
        ai_model_name = model_spec.model

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

        validation_result = await validation_processor.process(images)
        validation_payload = validation_result.model_dump()

        patient_name_candidates = build_patient_name_candidates(
            user_full_name=user.full_name,
            family_member_names=[family_member.name for family_member in family_members],
        )
        extraction_result = await processor.process(images, patient_name_candidates=patient_name_candidates)
        extraction_payload = extraction_result.model_dump()

        extracted_patient_name = extraction_result.patient_info.get("name")
        if isinstance(extracted_patient_name, str) and not matches_any_patient_name_candidate(
            extracted_patient_name,
            patient_name_candidates,
        ):
            patient_name_error = HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail={
                    "message": "Patient name verification failed.",
                    "errors": [
                        {
                            "field": "non_field",
                            "message": "Extracted patient name does not match the user or any saved family member. Please review your family members or rescan the document.",
                        }
                    ],
                },
            )
            scan_log = scan_log_repository.mark_failed(
                scan_log,
                status="patient_name_mismatch",
                error_payload=build_scan_log_error_payload(patient_name_error),
                validation_payload=validation_payload,
                extraction_payload=extraction_payload,
                ai_model_provider=ai_model_provider,
                ai_model_name=ai_model_name,
            )
            raise patient_name_error

        scan_log = scan_log_repository.mark_succeeded(
            scan_log,
            validation_payload=validation_payload,
            extraction_payload=extraction_payload,
            ai_model_provider=ai_model_provider,
            ai_model_name=ai_model_name,
        )
        return AIScanDocumentsResponse(
            scan_log_id=scan_log.id,
            **extraction_result.model_dump(),
        )
    except HTTPException as exc:
        if scan_log.status == "pending":
            scan_log_repository.mark_failed(
                scan_log,
                status=resolve_scan_failure_status(exc, validation_payload=validation_payload),
                error_payload=build_scan_log_error_payload(exc),
                validation_payload=validation_payload,
                extraction_payload=extraction_payload,
                ai_model_provider=ai_model_provider,
                ai_model_name=ai_model_name,
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
                ai_model_provider=ai_model_provider,
                ai_model_name=ai_model_name,
            )
        raise
