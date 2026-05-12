from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, JSON, LargeBinary, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.claim import Claim
    from app.models.user import User


class ScanLog(Base):
    __tablename__ = "scan_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True, nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="pending")
    ai_execution_mode: Mapped[str | None] = mapped_column(String(50), nullable=True)
    primary_model_provider: Mapped[str | None] = mapped_column(String(50), nullable=True)
    primary_model_name: Mapped[str | None] = mapped_column(String(150), nullable=True)
    validation_payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    extraction_payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    extraction_meta_payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    progress_payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    error_payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    user: Mapped[User] = relationship("User")
    files: Mapped[list["ScanLogFile"]] = relationship(
        back_populates="scan_log",
        cascade="all, delete-orphan",
        order_by="ScanLogFile.id.asc()",
    )
    ai_runs: Mapped[list["ScanLogAIRun"]] = relationship(
        back_populates="scan_log",
        cascade="all, delete-orphan",
        order_by="ScanLogAIRun.id.asc()",
    )
    ai_api_call_logs: Mapped[list["AIApiCallLog"]] = relationship(
        back_populates="scan_log",
        cascade="all, delete-orphan",
        order_by="AIApiCallLog.id.asc()",
    )
    linked_claims: Mapped[list[Claim]] = relationship(
        "Claim",
        back_populates="source_scan_log",
        foreign_keys="Claim.source_scan_log_id",
        order_by="Claim.id.asc()",
    )


class ScanLogFile(Base):
    __tablename__ = "scan_log_files"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    scan_log_id: Mapped[int] = mapped_column(ForeignKey("scan_logs.id"), index=True, nullable=False)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    original_mime_type: Mapped[str] = mapped_column(String(255), nullable=False)
    original_byte_size: Mapped[int] = mapped_column(Integer, nullable=False)
    original_sha256_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    original_content: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    processed_filename: Mapped[str | None] = mapped_column(String(255), nullable=True)
    processed_mime_type: Mapped[str | None] = mapped_column(String(255), nullable=True)
    processed_byte_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    processed_sha256_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    processed_content: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)

    scan_log: Mapped[ScanLog] = relationship(back_populates="files")


class ScanLogAIRun(Base):
    __tablename__ = "scan_log_ai_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    scan_log_id: Mapped[int] = mapped_column(ForeignKey("scan_logs.id"), index=True, nullable=False)
    producer_id: Mapped[str] = mapped_column(String(200), nullable=False)
    provider: Mapped[str] = mapped_column(String(50), nullable=False)
    model_name: Mapped[str] = mapped_column(String(150), nullable=False)
    producer_rank: Mapped[int] = mapped_column(Integer, nullable=False)
    run_index: Mapped[int] = mapped_column(Integer, nullable=False)
    temperature: Mapped[float] = mapped_column(Float, nullable=False)
    success: Mapped[bool] = mapped_column(Boolean, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    scan_log: Mapped[ScanLog] = relationship(back_populates="ai_runs")


class AIApiCallLog(Base):
    __tablename__ = "ai_api_call_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    scan_log_id: Mapped[int | None] = mapped_column(ForeignKey("scan_logs.id"), index=True, nullable=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), index=True, nullable=True)
    provider: Mapped[str] = mapped_column(String(50), nullable=False)
    model_name: Mapped[str] = mapped_column(String(150), nullable=False)
    call_kind: Mapped[str] = mapped_column(String(80), nullable=False)
    producer_id: Mapped[str | None] = mapped_column(String(200), nullable=True)
    producer_rank: Mapped[int | None] = mapped_column(Integer, nullable=True)
    run_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    temperature: Mapped[float | None] = mapped_column(Float, nullable=True)
    success: Mapped[bool] = mapped_column(Boolean, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    usage_payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    request_context_payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    scan_log: Mapped[ScanLog | None] = relationship(back_populates="ai_api_call_logs")
    user: Mapped[User | None] = relationship("User")
