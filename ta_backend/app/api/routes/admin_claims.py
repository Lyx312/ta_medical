import base64
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.routes.auth import get_current_admin_dependency
from app.db.session import get_db_session
from app.repositories.claim_repository import ClaimRepository
from app.schemas.admin_claim import AdminClaimDetailResponse, AdminClaimFileResponse
from app.schemas.auth import TokenData
from app.schemas.claim import ClaimResponse

router = APIRouter(prefix="/admin/claims", tags=["admin-claims"])


def build_data_url(mime_type: str, content: bytes) -> str:
    return f"data:{mime_type};base64,{base64.b64encode(content).decode('utf-8')}"


def build_claim_response(claim: Any) -> ClaimResponse:
    return ClaimResponse(
        id=claim.id,
        user_id=claim.user_id,
        user_email=claim.user_email,
        nama_pasien=claim.nama_pasien,
        tanggal_pelayanan=claim.tanggal_pelayanan,
        fasilitas_kesehatan=claim.fasilitas_kesehatan,
        jenis_manfaat=claim.jenis_manfaat,
        nominal_klaim=claim.nominal_klaim,
        keterangan_penyakit=claim.keterangan_penyakit,
        kronologi=claim.kronologi,
        file_count=len(claim.files),
        status=claim.status,
        source_scan_log_id=claim.source_scan_log_id,
        created_at=claim.created_at,
    )


@router.get("", response_model=list[ClaimResponse])
async def get_admin_claims(
    token_data: TokenData = Depends(get_current_admin_dependency),
    db: Session = Depends(get_db_session),
) -> list[ClaimResponse]:
    del token_data

    repository = ClaimRepository(db)
    claims = repository.list_all()
    return [build_claim_response(claim) for claim in claims]


@router.get("/{claim_id}", response_model=AdminClaimDetailResponse)
async def get_admin_claim(
    claim_id: int,
    token_data: TokenData = Depends(get_current_admin_dependency),
    db: Session = Depends(get_db_session),
) -> AdminClaimDetailResponse:
    del token_data

    repository = ClaimRepository(db)
    claim = repository.get_by_id(claim_id)
    if claim is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Claim not found.",
        )

    return AdminClaimDetailResponse(
        id=claim.id,
        user_id=claim.user_id,
        user_email=claim.user_email,
        nama_pasien=claim.nama_pasien,
        tanggal_pelayanan=claim.tanggal_pelayanan,
        fasilitas_kesehatan=claim.fasilitas_kesehatan,
        jenis_manfaat=claim.jenis_manfaat,
        nominal_klaim=claim.nominal_klaim,
        keterangan_penyakit=claim.keterangan_penyakit,
        kronologi=claim.kronologi,
        file_count=len(claim.files),
        status=claim.status,
        source_scan_log_id=claim.source_scan_log_id,
        created_at=claim.created_at,
        files=[
            AdminClaimFileResponse(
                id=claim_file.id,
                filename=claim_file.filename,
                mime_type=claim_file.mime_type,
                byte_size=claim_file.byte_size,
                sha256_hash=claim_file.sha256_hash,
                file_data_url=build_data_url(claim_file.mime_type, claim_file.content),
            )
            for claim_file in claim.files
        ],
    )
