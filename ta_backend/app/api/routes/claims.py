import base64
from datetime import date, datetime, timezone

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, UploadFile, status
from jose import JWTError, jwt

from app.core.config import get_settings
from app.models.claim import Claim, ClaimRepository, _next_id
from app.schemas.auth import TokenData
from app.schemas.claim import ClaimResponse

router = APIRouter(prefix="/claims", tags=["claims"])
claim_repository = ClaimRepository()
settings = get_settings()


def get_current_user_dependency(authorization: str | None = Header(None)) -> TokenData:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = authorization[7:]
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
        user_id = payload.get("sub")
        email = payload.get("email")
        role = payload.get("role")
        if user_id is None or email is None or role is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials.",
            )
        return TokenData(user_id=int(user_id), email=str(email), role=str(role))
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials.",
        )


def require_auth(authorization: str = "") -> TokenData:
    return get_current_user_dependency(authorization)


@router.post("", response_model=ClaimResponse, status_code=status.HTTP_201_CREATED)
async def create_claim(
    token_data: TokenData = Depends(require_auth),
    tanggal_pelayanan: date = Form(...),
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
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid jenis_manfaat. Must be one of: {', '.join(valid_types)}",
        )

    file_contents: list[str] = []
    for file in files:
        content = await file.read()
        file_contents.append(base64.b64encode(content).decode("utf-8"))

    claim = claim_repository.add(Claim(
        id=_next_id(),
        user_id=token_data.user_id,
        user_email=token_data.email,
        tanggal_pelayanan=tanggal_pelayanan,
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
    token_data: TokenData = Depends(require_auth),
) -> list[ClaimResponse]:
    claims = claim_repository.get_by_user_id(token_data.user_id)
    claims.sort(key=lambda c: c.created_at, reverse=True)

    return [
        ClaimResponse(
            id=c.id,
            user_id=c.user_id,
            user_email=c.user_email,
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
    token_data: TokenData = Depends(require_auth),
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
