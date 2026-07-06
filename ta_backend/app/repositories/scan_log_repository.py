from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import String, cast, or_, select
from sqlalchemy.orm import Session, selectinload

from app.models.scan_log import AIApiCallLog, ScanLog, ScanLogAIRun, ScanLogFile
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


@dataclass(frozen=True)
class ScanLogAIRunCreate:
    producer_id: str
    provider: str
    model_name: str
    producer_rank: int
    run_index: int
    temperature: float
    success: bool
    error_message: str | None = None
    payload: dict[str, Any] | None = None


@dataclass(frozen=True)
class AIApiCallLogCreate:
    provider: str
    model_name: str
    call_kind: str
    success: bool
    scan_log_id: int | None = None
    user_id: int | None = None
    producer_id: str | None = None
    producer_rank: int | None = None
    run_index: int | None = None
    temperature: float | None = None
    error_message: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    latency_ms: int | None = None
    usage_payload: dict[str, Any] | None = None
    request_context_payload: dict[str, Any] | None = None
    prompt_file_path: str | None = None
    prompt_payload: Any | None = None
    raw_response_text: str | None = None
    response_content_payload: Any | None = None
    parsed_response_payload: Any | None = None
    response_validation_error: str | None = None
    result_context_payload: dict[str, Any] | None = None


class ScanLogRepository:
    def __init__(self, db: Session) -> None:
        self._db = db

    def create_pending(
        self,
        *,
        user_id: int,
        file_records: list[ScanLogFileCreate],
        ai_execution_mode: str | None = None,
        progress_payload: dict[str, Any] | None = None,
    ) -> ScanLog:
        scan_log = ScanLog(
            user_id=user_id,
            status="pending",
            ai_execution_mode=ai_execution_mode,
            progress_payload=progress_payload,
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

    def update_progress(
        self,
        scan_log: ScanLog,
        *,
        progress_payload: dict[str, Any],
        status: str | None = None,
    ) -> ScanLog:
        if status is not None:
            scan_log.status = status
        scan_log.progress_payload = progress_payload
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
        primary_model_provider: str | None,
        primary_model_name: str | None,
        ai_execution_mode: str | None = None,
        ai_runs: list[ScanLogAIRunCreate] | None = None,
        extraction_meta_payload: dict[str, Any] | None = None,
    ) -> ScanLog:
        scan_log.status = "succeeded"
        scan_log.primary_model_provider = primary_model_provider
        scan_log.primary_model_name = primary_model_name
        scan_log.ai_execution_mode = ai_execution_mode
        scan_log.validation_payload = validation_payload
        scan_log.extraction_payload = extraction_payload
        scan_log.extraction_meta_payload = extraction_meta_payload
        if scan_log.progress_payload is not None:
            scan_log.progress_payload = {
                **scan_log.progress_payload,
                "stage": "completed",
                "title": "Ready to review",
                "detail": "Results are ready for you to review.",
                "percent": 100,
                "status": "succeeded",
            }
        scan_log.error_payload = None
        if ai_runs:
            self._persist_ai_runs(scan_log, ai_runs)
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
        primary_model_provider: str | None = None,
        primary_model_name: str | None = None,
        ai_execution_mode: str | None = None,
        ai_runs: list[ScanLogAIRunCreate] | None = None,
        extraction_meta_payload: dict[str, Any] | None = None,
    ) -> ScanLog:
        scan_log.status = status
        scan_log.primary_model_provider = primary_model_provider
        scan_log.primary_model_name = primary_model_name
        scan_log.ai_execution_mode = ai_execution_mode
        scan_log.validation_payload = validation_payload
        scan_log.extraction_payload = extraction_payload
        scan_log.extraction_meta_payload = extraction_meta_payload
        current_progress = scan_log.progress_payload or {}
        scan_log.progress_payload = {
            **current_progress,
            "stage": "failed",
            "title": "Scan failed",
            "detail": _extract_progress_error_message(error_payload),
            "percent": min(int(current_progress.get("percent") or 0), 97),
            "status": "failed",
        }
        scan_log.error_payload = error_payload
        if ai_runs:
            self._persist_ai_runs(scan_log, ai_runs)
        self._db.commit()
        return self.get_by_id(scan_log.id) or scan_log

    def mark_cancelled(self, scan_log: ScanLog) -> ScanLog:
        """Mark a scan as cancelled by the user."""
        scan_log.status = "cancelled"
        current_progress = scan_log.progress_payload or {}
        scan_log.progress_payload = {
            **current_progress,
            "stage": "failed",
            "title": "Scan cancelled",
            "detail": "Scan was cancelled by the user.",
            "percent": min(int(current_progress.get("percent") or 0), 97),
            "status": "failed",
        }
        scan_log.error_payload = {
            "status_code": 499,
            "detail": {"message": "Scan was cancelled by the user."},
        }
        self._db.commit()
        return self.get_by_id(scan_log.id) or scan_log

    def _persist_ai_runs(self, scan_log: ScanLog, runs: list[ScanLogAIRunCreate]) -> None:
        for run in runs:
            self._db.add(
                ScanLogAIRun(
                    scan_log_id=scan_log.id,
                    producer_id=run.producer_id,
                    provider=run.provider,
                    model_name=run.model_name,
                    producer_rank=run.producer_rank,
                    run_index=run.run_index,
                    temperature=run.temperature,
                    success=run.success,
                    error_message=run.error_message,
                    payload=run.payload,
                )
            )

    def create_ai_api_call_log(self, payload: AIApiCallLogCreate) -> AIApiCallLog:
        record = AIApiCallLog(
            scan_log_id=payload.scan_log_id,
            user_id=payload.user_id,
            provider=payload.provider,
            model_name=payload.model_name,
            call_kind=payload.call_kind,
            producer_id=payload.producer_id,
            producer_rank=payload.producer_rank,
            run_index=payload.run_index,
            temperature=payload.temperature,
            success=payload.success,
            error_message=payload.error_message,
            prompt_tokens=payload.prompt_tokens,
            completion_tokens=payload.completion_tokens,
            total_tokens=payload.total_tokens,
            latency_ms=payload.latency_ms,
            usage_payload=payload.usage_payload,
            request_context_payload=payload.request_context_payload,
            prompt_file_path=payload.prompt_file_path,
            prompt_payload=payload.prompt_payload,
            raw_response_text=payload.raw_response_text,
            response_content_payload=payload.response_content_payload,
            parsed_response_payload=payload.parsed_response_payload,
            response_validation_error=payload.response_validation_error,
            result_context_payload=payload.result_context_payload,
        )
        self._db.add(record)
        self._db.commit()
        self._db.refresh(record)
        return record

    def update_ai_api_call_log(
        self,
        ai_api_call_log_id: int,
        *,
        parsed_response_payload: Any | None = None,
        response_validation_error: str | None = None,
        result_context_payload: dict[str, Any] | None = None,
    ) -> AIApiCallLog | None:
        record = self._db.get(AIApiCallLog, ai_api_call_log_id)
        if record is None:
            return None

        record.parsed_response_payload = parsed_response_payload
        record.response_validation_error = response_validation_error
        record.result_context_payload = result_context_payload
        self._db.commit()
        self._db.refresh(record)
        return record

    def get_by_id(self, scan_log_id: int) -> ScanLog | None:
        return self._db.execute(
            select(ScanLog)
            .where(ScanLog.id == scan_log_id)
            .options(
                selectinload(ScanLog.files),
                selectinload(ScanLog.user),
                selectinload(ScanLog.linked_claims),
                selectinload(ScanLog.ai_runs),
                selectinload(ScanLog.ai_api_call_logs),
            )
        ).scalar_one_or_none()

    def get_by_id_for_eval_export(self, scan_log_id: int) -> ScanLog | None:
        return self._db.execute(
            select(ScanLog)
            .where(ScanLog.id == scan_log_id)
            .options(
                selectinload(ScanLog.ai_runs),
                selectinload(ScanLog.ai_api_call_logs),
            )
        ).scalar_one_or_none()

    def get_latest_for_user_for_eval_export(self, user_id: int) -> ScanLog | None:
        return self._db.execute(
            select(ScanLog)
            .where(ScanLog.user_id == user_id)
            .order_by(ScanLog.created_at.desc(), ScanLog.id.desc())
            .limit(1)
            .options(
                selectinload(ScanLog.ai_runs),
                selectinload(ScanLog.ai_api_call_logs),
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
                selectinload(ScanLog.ai_runs),
                selectinload(ScanLog.ai_api_call_logs),
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
                selectinload(ScanLog.ai_runs),
                selectinload(ScanLog.ai_api_call_logs),
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


def _extract_progress_error_message(error_payload: dict[str, Any]) -> str:
    detail = error_payload.get("detail")
    if isinstance(detail, dict):
        raw_errors = detail.get("errors")
        if isinstance(raw_errors, list):
            for error in raw_errors:
                if isinstance(error, dict):
                    message = str(error.get("message", "")).strip()
                    if message:
                        return message
        message = str(detail.get("message", "")).strip()
        if message:
            return message
    return "Scan did not complete."
