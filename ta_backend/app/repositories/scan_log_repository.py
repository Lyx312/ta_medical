from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import String, cast, or_, select
from sqlalchemy.orm import Session, selectinload

from app.models.scan_log import ScanLog, ScanLogFile
from app.models.user import User


@dataclass(frozen=True)
class ScanLogFileCreate:
    original_filename: str
    original_mime_type: str
    original_byte_size: int
    original_sha256_hash: str
    original_content: bytes
    processed_filename: str | None = None
    processed_mime_type: str | None = None
    processed_byte_size: int | None = None
    processed_sha256_hash: str | None = None
    processed_content: bytes | None = None


class ScanLogRepository:
    def __init__(self, db: Session) -> None:
        self._db = db

    def create_pending(
        self,
        *,
        user_id: int,
        file_records: list[ScanLogFileCreate],
        ai_model_provider: str | None = None,
        ai_model_name: str | None = None,
    ) -> ScanLog:
        scan_log = ScanLog(
            user_id=user_id,
            status="pending",
            ai_model_provider=ai_model_provider,
            ai_model_name=ai_model_name,
        )
        scan_log.files.extend(
            ScanLogFile(
                original_filename=file_record.original_filename,
                original_mime_type=file_record.original_mime_type,
                original_byte_size=file_record.original_byte_size,
                original_sha256_hash=file_record.original_sha256_hash,
                original_content=file_record.original_content,
                processed_filename=file_record.processed_filename,
                processed_mime_type=file_record.processed_mime_type,
                processed_byte_size=file_record.processed_byte_size,
                processed_sha256_hash=file_record.processed_sha256_hash,
                processed_content=file_record.processed_content,
            )
            for file_record in file_records
        )
        self._db.add(scan_log)
        self._db.commit()
        return self.get_by_id(scan_log.id) or scan_log

    def update_file_processed_content(
        self,
        scan_log_file: ScanLogFile,
        *,
        filename: str,
        mime_type: str,
        byte_size: int,
        sha256_hash: str,
        content: bytes,
    ) -> ScanLogFile:
        scan_log_file.processed_filename = filename
        scan_log_file.processed_mime_type = mime_type
        scan_log_file.processed_byte_size = byte_size
        scan_log_file.processed_sha256_hash = sha256_hash
        scan_log_file.processed_content = content
        self._db.commit()
        self._db.refresh(scan_log_file)
        return scan_log_file

    def mark_succeeded(
        self,
        scan_log: ScanLog,
        *,
        validation_payload: dict[str, Any],
        extraction_payload: dict[str, Any],
        ai_model_provider: str | None,
        ai_model_name: str | None,
    ) -> ScanLog:
        scan_log.status = "succeeded"
        scan_log.ai_model_provider = ai_model_provider
        scan_log.ai_model_name = ai_model_name
        scan_log.validation_payload = validation_payload
        scan_log.extraction_payload = extraction_payload
        scan_log.error_payload = None
        self._db.commit()
        return self.get_by_id(scan_log.id) or scan_log

    def mark_failed(
        self,
        scan_log: ScanLog,
        *,
        status: str,
        error_payload: dict[str, Any],
        validation_payload: dict[str, Any] | None = None,
        extraction_payload: dict[str, Any] | None = None,
        ai_model_provider: str | None = None,
        ai_model_name: str | None = None,
    ) -> ScanLog:
        scan_log.status = status
        scan_log.ai_model_provider = ai_model_provider
        scan_log.ai_model_name = ai_model_name
        scan_log.validation_payload = validation_payload
        scan_log.extraction_payload = extraction_payload
        scan_log.error_payload = error_payload
        self._db.commit()
        return self.get_by_id(scan_log.id) or scan_log

    def get_by_id(self, scan_log_id: int) -> ScanLog | None:
        return self._db.execute(
            select(ScanLog)
            .where(ScanLog.id == scan_log_id)
            .options(
                selectinload(ScanLog.files),
                selectinload(ScanLog.user),
                selectinload(ScanLog.linked_claims),
            )
        ).scalar_one_or_none()

    def get_by_id_and_user_id(self, scan_log_id: int, user_id: int) -> ScanLog | None:
        return self._db.execute(
            select(ScanLog)
            .where(
                ScanLog.id == scan_log_id,
                ScanLog.user_id == user_id,
            )
            .options(
                selectinload(ScanLog.files),
                selectinload(ScanLog.user),
                selectinload(ScanLog.linked_claims),
            )
        ).scalar_one_or_none()

    def list_for_admin(
        self,
        *,
        status: str | None = None,
        q: str | None = None,
        linked: bool | None = None,
        limit: int | None = 50,
    ) -> list[ScanLog]:
        statement = (
            select(ScanLog)
            .join(ScanLog.user)
            .order_by(ScanLog.created_at.desc(), ScanLog.id.desc())
            .options(
                selectinload(ScanLog.files),
                selectinload(ScanLog.user),
                selectinload(ScanLog.linked_claims),
            )
        )
        if status:
            statement = statement.where(ScanLog.status == status)

        normalized_query = (q or "").strip()
        if normalized_query:
            like_pattern = f"%{normalized_query}%"
            statement = statement.where(
                or_(
                    cast(ScanLog.id, String).ilike(like_pattern),
                    User.full_name.ilike(like_pattern),
                    User.email.ilike(like_pattern),
                )
            )

        if linked is True:
            statement = statement.where(ScanLog.linked_claims.any())
        elif linked is False:
            statement = statement.where(~ScanLog.linked_claims.any())

        if limit is not None:
            statement = statement.limit(limit)

        return list(self._db.execute(statement).scalars())

    def summarize_for_admin(
        self,
        *,
        status: str | None = None,
        q: str | None = None,
        linked: bool | None = None,
    ) -> dict[str, int]:
        scan_logs = self.list_for_admin(status=status, q=q, linked=linked, limit=None)
        return {
            "total_logs": len(scan_logs),
            "succeeded_count": sum(1 for log in scan_logs if log.status == "succeeded"),
            "failed_count": sum(1 for log in scan_logs if log.status not in {"pending", "succeeded"}),
            "unlinked_count": sum(1 for log in scan_logs if len(log.linked_claims) == 0),
        }
