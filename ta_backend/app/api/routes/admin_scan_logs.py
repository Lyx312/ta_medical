import base64
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.api.routes.auth import get_current_admin_dependency
from app.db.session import get_db_session
from app.repositories.scan_log_repository import ScanLogRepository
from app.schemas.admin_scan_log import (
    AdminScanLogDetailResponse,
    AdminScanLogFileResponse,
    AdminScanLogListItemResponse,
)
from app.schemas.auth import TokenData

router = APIRouter(prefix="/admin/scan-logs", tags=["admin-scan-logs"])


def build_data_url(mime_type: str, content: bytes) -> str:
    return f"data:{mime_type};base64,{base64.b64encode(content).decode('utf-8')}"


def build_scan_log_list_item(scan_log: Any) -> AdminScanLogListItemResponse:
    return AdminScanLogListItemResponse(
        id=scan_log.id,
        user_id=scan_log.user.id,
        user_full_name=scan_log.user.full_name,
        user_email=scan_log.user.email,
        status=scan_log.status,
        file_count=len(scan_log.files),
        linked_claim_ids=[claim.id for claim in scan_log.linked_claims],
        ai_model_provider=scan_log.ai_model_provider,
        ai_model_name=scan_log.ai_model_name,
        created_at=scan_log.created_at,
        updated_at=scan_log.updated_at,
    )


def build_scan_log_file_response(scan_log_file: Any) -> AdminScanLogFileResponse:
    return AdminScanLogFileResponse(
        id=scan_log_file.id,
        original_filename=scan_log_file.original_filename,
        original_mime_type=scan_log_file.original_mime_type,
        original_byte_size=scan_log_file.original_byte_size,
        original_sha256_hash=scan_log_file.original_sha256_hash,
        original_image_data_url=build_data_url(
            scan_log_file.original_mime_type,
            scan_log_file.original_content,
        ),
        processed_filename=scan_log_file.processed_filename,
        processed_mime_type=scan_log_file.processed_mime_type,
        processed_byte_size=scan_log_file.processed_byte_size,
        processed_sha256_hash=scan_log_file.processed_sha256_hash,
        processed_image_data_url=(
            build_data_url(scan_log_file.processed_mime_type, scan_log_file.processed_content)
            if scan_log_file.processed_mime_type and scan_log_file.processed_content
            else None
        ),
    )


@router.get("", response_model=list[AdminScanLogListItemResponse])
async def get_scan_logs(
    token_data: TokenData = Depends(get_current_admin_dependency),
    db: Session = Depends(get_db_session),
    status_filter: str | None = Query(default=None, alias="status"),
    limit: int = Query(default=50, ge=1, le=100),
) -> list[AdminScanLogListItemResponse]:
    del token_data

    repository = ScanLogRepository(db)
    scan_logs = repository.list_for_admin(status=status_filter, limit=limit)
    return [build_scan_log_list_item(scan_log) for scan_log in scan_logs]


@router.get("/{scan_log_id}", response_model=AdminScanLogDetailResponse)
async def get_scan_log(
    scan_log_id: int,
    token_data: TokenData = Depends(get_current_admin_dependency),
    db: Session = Depends(get_db_session),
) -> AdminScanLogDetailResponse:
    del token_data

    repository = ScanLogRepository(db)
    scan_log = repository.get_by_id(scan_log_id)
    if scan_log is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Scan log not found.",
        )

    list_item = build_scan_log_list_item(scan_log)
    return AdminScanLogDetailResponse(
        **list_item.model_dump(),
        validation_payload=scan_log.validation_payload,
        extraction_payload=scan_log.extraction_payload,
        error_payload=scan_log.error_payload,
        files=[build_scan_log_file_response(scan_log_file) for scan_log_file in scan_log.files],
    )
