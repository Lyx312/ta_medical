from __future__ import annotations

from datetime import date, datetime, timezone

from sqlalchemy import Date, DateTime, Float, ForeignKey, Integer, LargeBinary, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.models.scan_log import ScanLog


class Claim(Base):
    __tablename__ = "claims"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True, nullable=False)
    user_email: Mapped[str] = mapped_column(String(255), nullable=False)
    nama_pasien: Mapped[str] = mapped_column(String(255), nullable=False)
    tanggal_pelayanan: Mapped[date] = mapped_column(Date, nullable=False)
    fasilitas_kesehatan: Mapped[str] = mapped_column(String(255), nullable=False)
    jenis_manfaat: Mapped[str] = mapped_column(String(50), nullable=False)
    nominal_klaim: Mapped[float] = mapped_column(Float, nullable=False)
    keterangan_penyakit: Mapped[str] = mapped_column(String, nullable=False)
    kronologi: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="pending")
    source_scan_log_id: Mapped[int | None] = mapped_column(
        ForeignKey("scan_logs.id"),
        index=True,
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    files: Mapped[list["ClaimFile"]] = relationship(
        back_populates="claim",
        cascade="all, delete-orphan",
        order_by="ClaimFile.id.asc()",
    )
    source_scan_log: Mapped["ScanLog | None"] = relationship(
        "ScanLog",
        back_populates="linked_claims",
        foreign_keys=[source_scan_log_id],
    )


class ClaimFile(Base):
    __tablename__ = "claim_files"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    claim_id: Mapped[int] = mapped_column(ForeignKey("claims.id"), index=True, nullable=False)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    mime_type: Mapped[str] = mapped_column(String(255), nullable=False)
    byte_size: Mapped[int] = mapped_column(Integer, nullable=False)
    sha256_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    content: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)

    claim: Mapped[Claim] = relationship(back_populates="files")
