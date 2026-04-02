from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.models.claim import Claim, ClaimFile


@dataclass(frozen=True)
class ClaimFileCreate:
    filename: str
    mime_type: str
    byte_size: int
    sha256_hash: str
    content: bytes


class ClaimRepository:
    def __init__(self, db: Session) -> None:
        self._db = db

    def create(
        self,
        *,
        user_id: int,
        user_email: str,
        nama_pasien: str,
        tanggal_pelayanan: date,
        fasilitas_kesehatan: str,
        jenis_manfaat: str,
        nominal_klaim: float,
        keterangan_penyakit: str,
        kronologi: str,
        status: str,
        created_at: datetime,
        files: list[ClaimFileCreate],
        source_scan_log_id: int | None = None,
    ) -> Claim:
        claim = Claim(
            user_id=user_id,
            user_email=user_email,
            nama_pasien=nama_pasien.strip(),
            tanggal_pelayanan=tanggal_pelayanan,
            fasilitas_kesehatan=fasilitas_kesehatan.strip(),
            jenis_manfaat=jenis_manfaat,
            nominal_klaim=nominal_klaim,
            keterangan_penyakit=keterangan_penyakit.strip(),
            kronologi=kronologi.strip(),
            status=status,
            created_at=created_at,
            source_scan_log_id=source_scan_log_id,
        )
        claim.files.extend(
            ClaimFile(
                filename=file_record.filename,
                mime_type=file_record.mime_type,
                byte_size=file_record.byte_size,
                sha256_hash=file_record.sha256_hash,
                content=file_record.content,
            )
            for file_record in files
        )
        self._db.add(claim)
        self._db.commit()
        return self.get_by_id(claim.id) or claim

    def get_by_id(self, claim_id: int) -> Claim | None:
        return self._db.execute(
            select(Claim)
            .where(Claim.id == claim_id)
            .options(
                selectinload(Claim.files),
                selectinload(Claim.source_scan_log),
            )
        ).scalar_one_or_none()

    def get_by_user_id(self, user_id: int) -> list[Claim]:
        return list(
            self._db.execute(
                select(Claim)
                .where(Claim.user_id == user_id)
                .order_by(Claim.created_at.desc(), Claim.id.desc())
                .options(
                    selectinload(Claim.files),
                    selectinload(Claim.source_scan_log),
                )
            ).scalars()
        )

    def list_all(self) -> list[Claim]:
        return list(
            self._db.execute(
                select(Claim)
                .order_by(Claim.created_at.desc(), Claim.id.desc())
                .options(
                    selectinload(Claim.files),
                    selectinload(Claim.source_scan_log),
                )
            ).scalars()
        )
