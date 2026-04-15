from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from sqlalchemy import String, cast, or_, select
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
        medicines: str | None = None,
        medicines_payload: list[dict[str, Any]] | None = None,
        requires_additional_review: bool = False,
        review_note: str | None = None,
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
            medicines=medicines.strip() if medicines and medicines.strip() else None,
            medicines_payload=medicines_payload,
            requires_additional_review=requires_additional_review,
            review_note=review_note.strip() if review_note else None,
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

    def _build_admin_list_statement(
        self,
        *,
        q: str | None = None,
        linked: bool | None = None,
        sort: str = "newest",
        limit: int | None = None,
    ):
        statement = select(Claim).options(
            selectinload(Claim.files),
            selectinload(Claim.source_scan_log),
        )

        normalized_query = (q or "").strip()
        if normalized_query:
            like_pattern = f"%{normalized_query}%"
            statement = statement.where(
                or_(
                    cast(Claim.id, String).ilike(like_pattern),
                    Claim.nama_pasien.ilike(like_pattern),
                    Claim.user_email.ilike(like_pattern),
                )
            )

        if linked is True:
            statement = statement.where(Claim.source_scan_log_id.is_not(None))
        elif linked is False:
            statement = statement.where(Claim.source_scan_log_id.is_(None))

        if sort == "oldest":
            statement = statement.order_by(Claim.created_at.asc(), Claim.id.asc())
        elif sort == "amount_desc":
            statement = statement.order_by(Claim.nominal_klaim.desc(), Claim.created_at.desc(), Claim.id.desc())
        elif sort == "amount_asc":
            statement = statement.order_by(Claim.nominal_klaim.asc(), Claim.created_at.desc(), Claim.id.desc())
        else:
            statement = statement.order_by(Claim.created_at.desc(), Claim.id.desc())

        if limit is not None:
            statement = statement.limit(limit)

        return statement

    def list_for_admin(
        self,
        *,
        q: str | None = None,
        linked: bool | None = None,
        sort: str = "newest",
        limit: int | None = 100,
    ) -> list[Claim]:
        return list(
            self._db.execute(
                self._build_admin_list_statement(
                    q=q,
                    linked=linked,
                    sort=sort,
                    limit=limit,
                )
            ).scalars()
        )

    def summarize_for_admin(
        self,
        *,
        q: str | None = None,
        linked: bool | None = None,
    ) -> dict[str, float | int]:
        claims = self.list_for_admin(q=q, linked=linked, limit=None)
        return {
            "total_claims": len(claims),
            "total_claim_amount": float(sum(claim.nominal_klaim for claim in claims)),
            "linked_scan_count": sum(1 for claim in claims if claim.source_scan_log_id is not None),
            "manual_submission_count": sum(1 for claim in claims if claim.source_scan_log_id is None),
        }

    def list_all(self) -> list[Claim]:
        return self.list_for_admin(limit=None)
