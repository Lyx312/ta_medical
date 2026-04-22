import base64
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import TypeAdapter, ValidationError
from sqlalchemy.orm import Session

from app.api.routes.auth import get_current_user_dependency
from app.core.config import get_settings
from app.db.session import get_db_session
from app.repositories.claim_repository import ClaimFileCreate, ClaimRepository
from app.repositories.scan_log_repository import ScanLogRepository
from app.schemas.auth import TokenData
from app.schemas.claim import ClaimDetailResponse, ClaimMedicine, ClaimMedicineInput, ClaimResponse, parse_claim_date
from app.services.ai_service import DocumentImageInput, DocumentValidationProcessor
from app.services.ai_service import build_patient_name_verification_error
from app.services.ai_service import matches_any_patient_name_candidate
from app.services.claim_medicine_relevance import evaluate_claim_medicines, persist_claim_medicine_knowledge
from app.services.image_preprocessing import preprocess_document_image

router = APIRouter(prefix="/claims", tags=["claims"])
validation_processor = DocumentValidationProcessor(settings=get_settings())
claim_medicines_adapter = TypeAdapter(list[ClaimMedicine])
claim_medicine_input_adapter = TypeAdapter(list[ClaimMedicineInput])


@dataclass(frozen=True)
class BufferedUpload:
    filename: str
    mime_type: str
    content: bytes
    byte_size: int
    sha256_hash: str


def build_claim_response(claim: Any) -> ClaimResponse:
    medicines_payload = claim.medicines_payload if isinstance(claim.medicines_payload, list) else []
    try:
        medicines = claim_medicines_adapter.validate_python(medicines_payload)
    except ValidationError:
        medicines = []

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
        medicines=medicines,
        requires_additional_review=bool(claim.requires_additional_review),
        review_note=claim.review_note,
        file_count=len(claim.files),
        status=claim.status,
        source_scan_log_id=claim.source_scan_log_id,
        created_at=claim.created_at,
    )


def build_claim_detail_response(claim: Any) -> ClaimDetailResponse:
    medicines_payload = claim.medicines_payload if isinstance(claim.medicines_payload, list) else []
    try:
        medicines = claim_medicines_adapter.validate_python(medicines_payload)
    except ValidationError:
        medicines = []

    source_scan_log = getattr(claim, "source_scan_log", None)

    return ClaimDetailResponse(
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
        medicines=medicines,
        requires_additional_review=bool(claim.requires_additional_review),
        review_note=claim.review_note,
        file_count=len(claim.files),
        status=claim.status,
        source_scan_log_id=claim.source_scan_log_id,
        created_at=claim.created_at,
        source_scan_extraction_payload=(
            source_scan_log.extraction_payload if source_scan_log is not None else None
        ),
    )


def normalize_upload_filename(filename: str | None, index: int) -> str:
    candidate = (filename or "").strip()
    if candidate:
        return candidate
    return f"uploaded_file_{index + 1}"


def build_sorted_hashes(items: list[str]) -> list[str]:
    return sorted(items)


def build_invalid_scan_log_link_error() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        detail={
            "message": "Input validation failed.",
            "errors": [
                {
                    "field": "source_scan_log_id",
                    "message": "The selected scan log cannot be linked to this claim.",
                }
            ],
        },
    )


async def read_uploaded_files(files: list[UploadFile]) -> list[BufferedUpload]:
    buffered_uploads: list[BufferedUpload] = []
    for index, file in enumerate(files):
        content = await file.read()
        buffered_uploads.append(
            BufferedUpload(
                filename=normalize_upload_filename(file.filename, index),
                mime_type=file.content_type or "application/octet-stream",
                content=content,
                byte_size=len(content),
                sha256_hash=hashlib.sha256(content).hexdigest(),
            )
        )
    return buffered_uploads


def scan_log_matches_uploaded_files(scan_log: Any, uploads: list[BufferedUpload]) -> bool:
    return build_sorted_hashes([scan_file.original_sha256_hash for scan_file in scan_log.files]) == build_sorted_hashes(
        [upload.sha256_hash for upload in uploads]
    )


def can_reuse_scan_log_validation(scan_log: Any, uploads: list[BufferedUpload]) -> bool:
    return (
        scan_log.status == "succeeded"
        and scan_log.validation_payload is not None
        and scan_log_matches_uploaded_files(scan_log, uploads)
    )


def get_validation_payload_patient_name(validation_payload: dict[str, Any] | None) -> tuple[str | None, str | None]:
    if not isinstance(validation_payload, dict):
        return None, None

    packet_patient_name = validation_payload.get("packet_patient_name")
    patient_name_match_status = validation_payload.get("patient_name_match_status")
    return (
        packet_patient_name.strip() if isinstance(packet_patient_name, str) and packet_patient_name.strip() else None,
        patient_name_match_status if isinstance(patient_name_match_status, str) else None,
    )


def has_usable_validation_patient_name(validation_payload: dict[str, Any] | None) -> bool:
    packet_patient_name, patient_name_match_status = get_validation_payload_patient_name(validation_payload)
    return packet_patient_name is not None and patient_name_match_status in {"matched", "mismatched"}


def build_selected_patient_name_error() -> HTTPException:
    return build_patient_name_verification_error(
        "Uploaded documents do not match the selected patient. Please choose the correct patient or upload the correct documents."
    )


def build_invalid_medicines_payload_error() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        detail={
            "message": "Input validation failed.",
            "errors": [
                {
                    "field": "medicines_payload",
                    "message": "Invalid medicines payload format.",
                }
            ],
        },
    )


def parse_claim_medicines_payload(raw_payload: str | None) -> list[ClaimMedicineInput]:
    if raw_payload is None or not raw_payload.strip():
        return []

    try:
        parsed = json.loads(raw_payload)
    except json.JSONDecodeError as exc:
        raise build_invalid_medicines_payload_error() from exc

    try:
        medicines = claim_medicine_input_adapter.validate_python(parsed)
    except ValidationError as exc:
        raise build_invalid_medicines_payload_error() from exc

    normalized: list[ClaimMedicineInput] = []
    for item in medicines:
        name = item.name.strip()
        if not name:
            continue

        normalized.append(
            ClaimMedicineInput(
                name=name,
                quantity=item.quantity,
                unit=item.unit.strip() if item.unit else None,
                dosage_instruction=item.dosage_instruction.strip() if item.dosage_instruction else None,
                source=item.source,
            )
        )

    return normalized


@router.post("", response_model=ClaimDetailResponse, status_code=status.HTTP_201_CREATED)
async def create_claim(
    token_data: TokenData = Depends(get_current_user_dependency),
    db: Session = Depends(get_db_session),
    nama_pasien: str = Form(...),
    tanggal_pelayanan: str = Form(...),
    fasilitas_kesehatan: str = Form(...),
    jenis_manfaat: str = Form(...),
    nominal_klaim: float = Form(...),
    keterangan_penyakit: str = Form(...),
    kronologi: str = Form(...),
    medicines_payload: str | None = Form(default=None),
    source_scan_log_id: int | None = Form(default=None),
    files: list[UploadFile] = File(default=[]),
) -> ClaimDetailResponse:
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

    medicines_input = parse_claim_medicines_payload(medicines_payload)
    evaluated_medicines, requires_additional_review, review_note = await evaluate_claim_medicines(
        db,
        medicines_input,
        keterangan_penyakit,
        kronologi,
        persist=False,
    )

    buffered_uploads = await read_uploaded_files(files)
    source_scan_log: Any | None = None
    should_run_validation = source_scan_log_id is None
    should_reuse_existing_validation = source_scan_log_id is not None

    if source_scan_log_id is not None:
        scan_log_repository = ScanLogRepository(db)
        source_scan_log = scan_log_repository.get_by_id_and_user_id(source_scan_log_id, token_data.user_id)
        if source_scan_log is None or not scan_log_matches_uploaded_files(source_scan_log, buffered_uploads):
            raise build_invalid_scan_log_link_error()
        if not can_reuse_scan_log_validation(source_scan_log, buffered_uploads):
            should_run_validation = True
            should_reuse_existing_validation = False
        elif has_usable_validation_patient_name(source_scan_log.validation_payload):
            packet_patient_name, _ = get_validation_payload_patient_name(source_scan_log.validation_payload)
            if not matches_any_patient_name_candidate(packet_patient_name, [nama_pasien]):
                raise build_selected_patient_name_error()
        else:
            should_run_validation = True
            should_reuse_existing_validation = False

    if should_run_validation:
        validation_images: list[DocumentImageInput] = []
        for upload in buffered_uploads:
            processed_image = preprocess_document_image(
                filename=upload.filename,
                content=upload.content,
            )
            validation_images.append(
                DocumentImageInput(
                    filename=processed_image.filename,
                    mime_type=processed_image.mime_type,
                    base64_data=base64.b64encode(processed_image.content).decode("utf-8"),
                )
            )

        validation_result = await validation_processor.process(
            validation_images,
            expected_patient_name=nama_pasien,
        )
        if validation_result.patient_name_match_status == "mismatched":
            raise build_selected_patient_name_error()

    if source_scan_log is not None and should_reuse_existing_validation:
        packet_patient_name, _ = get_validation_payload_patient_name(source_scan_log.validation_payload)
        if packet_patient_name is not None and not matches_any_patient_name_candidate(packet_patient_name, [nama_pasien]):
            raise build_selected_patient_name_error()

    claim_repository = ClaimRepository(db)
    claim = claim_repository.create(
        user_id=token_data.user_id,
        user_email=token_data.email,
        nama_pasien=nama_pasien,
        tanggal_pelayanan=parsed_tanggal_pelayanan,
        fasilitas_kesehatan=fasilitas_kesehatan,
        jenis_manfaat=jenis_manfaat,
        nominal_klaim=nominal_klaim,
        keterangan_penyakit=keterangan_penyakit,
        kronologi=kronologi,
        medicines="\n".join(item.name for item in evaluated_medicines) if evaluated_medicines else None,
        medicines_payload=[item.model_dump(mode="json") for item in evaluated_medicines],
        requires_additional_review=requires_additional_review,
        review_note=review_note,
        status="pending",
        created_at=datetime.now(timezone.utc),
        source_scan_log_id=source_scan_log_id,
        files=[
            ClaimFileCreate(
                filename=upload.filename,
                mime_type=upload.mime_type,
                byte_size=upload.byte_size,
                sha256_hash=upload.sha256_hash,
                content=upload.content,
            )
            for upload in buffered_uploads
        ],
    )

    if evaluated_medicines:
        persist_claim_medicine_knowledge(
            db,
            evaluated_medicines,
            keterangan_penyakit,
            kronologi,
            claim_id=claim.id,
        )
        db.commit()

    return build_claim_detail_response(claim)


@router.get("", response_model=list[ClaimResponse])
async def get_claims(
    token_data: TokenData = Depends(get_current_user_dependency),
    db: Session = Depends(get_db_session),
) -> list[ClaimResponse]:
    claim_repository = ClaimRepository(db)
    claims = claim_repository.get_by_user_id(token_data.user_id)
    return [build_claim_response(claim) for claim in claims]


@router.get("/{claim_id}", response_model=ClaimDetailResponse)
async def get_claim(
    claim_id: int,
    token_data: TokenData = Depends(get_current_user_dependency),
    db: Session = Depends(get_db_session),
) -> ClaimDetailResponse:
    claim_repository = ClaimRepository(db)
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

    return build_claim_detail_response(claim)
