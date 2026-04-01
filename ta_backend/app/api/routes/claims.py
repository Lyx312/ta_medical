import base64
from datetime import date, datetime, timezone

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from app.api.routes.auth import get_current_user_dependency
from app.models.claim import Claim, ClaimRepository, _next_id
from app.schemas.auth import TokenData
from app.schemas.claim import ClaimResponse, parse_claim_date

router = APIRouter(prefix="/claims", tags=["claims"])
claim_repository = ClaimRepository()


@router.post("", response_model=ClaimResponse, status_code=status.HTTP_201_CREATED)
async def create_claim(
    token_data: TokenData = Depends(get_current_user_dependency),
    nama_pasien: str = Form(...),
    tanggal_pelayanan: str = Form(...),
    fasilitas_kesehatan: str = Form(...),
    jenis_manfaat: str = Form(...),
    nominal_klaim: float = Form(...),
    keterangan_penyakit: str = Form(...),
    kronologi: str = Form(...),
    files: list[UploadFile] = File(default=[]),
) -> ClaimResponse:
    valid_types = ["rawat_jalan", "rawat_inap", "emergency", "maternity"]
    if jenis_manfaat not in valid_types:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Invalid jenis_manfaat. Must be one of: {', '.join(valid_types)}",
        )

    try:
        parsed_tanggal_pelayanan = parse_claim_date(tanggal_pelayanan)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Invalid tanggal_pelayanan. Expected format: dd-mm-yyyy",
        ) from exc

    file_contents: list[str] = []
    for file in files:
        content = await file.read()
        file_contents.append(base64.b64encode(content).decode("utf-8"))

    claim = claim_repository.add(Claim(
        id=_next_id(),
        user_id=token_data.user_id,
        user_email=token_data.email,
        nama_pasien=nama_pasien.strip(),
        tanggal_pelayanan=parsed_tanggal_pelayanan,
        fasilitas_kesehatan=fasilitas_kesehatan.strip(),
        jenis_manfaat=jenis_manfaat,  # type: ignore
        nominal_klaim=nominal_klaim,
        keterangan_penyakit=keterangan_penyakit.strip(),
        kronologi=kronologi.strip(),
        files=file_contents,
        status="pending",
        created_at=datetime.now(timezone.utc),
    ))

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
        created_at=claim.created_at,
    )


@router.get("", response_model=list[ClaimResponse])
async def get_claims(
    token_data: TokenData = Depends(get_current_user_dependency),
) -> list[ClaimResponse]:
    claims = claim_repository.get_by_user_id(token_data.user_id)
    claims.sort(key=lambda c: c.created_at, reverse=True)

    return [
        ClaimResponse(
            id=c.id,
            user_id=c.user_id,
            user_email=c.user_email,
            nama_pasien=c.nama_pasien,
            tanggal_pelayanan=c.tanggal_pelayanan,
            fasilitas_kesehatan=c.fasilitas_kesehatan,
            jenis_manfaat=c.jenis_manfaat,
            nominal_klaim=c.nominal_klaim,
            keterangan_penyakit=c.keterangan_penyakit,
            kronologi=c.kronologi,
            file_count=len(c.files),
            status=c.status,
            created_at=c.created_at,
        )
        for c in claims
    ]


@router.get("/{claim_id}", response_model=ClaimResponse)
async def get_claim(
    claim_id: int,
    token_data: TokenData = Depends(get_current_user_dependency),
) -> ClaimResponse:
    claim = claim_repository.get_by_id(claim_id)
    if claim is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Claim not found.",
        )
    if claim.user_id != token_data.user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to view this claim.",
        )

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
        created_at=claim.created_at,
    )
