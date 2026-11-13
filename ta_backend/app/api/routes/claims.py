import base64
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile, status
from pydantic import TypeAdapter, ValidationError
from sqlalchemy.orm import Session

from app.api.routes.auth import get_current_user_dependency
from app.core.config import get_settings
from app.db.session import get_db_session
from app.repositories.claim_repository import ClaimFileCreate, ClaimRepository
from app.repositories.family_member_repository import FamilyMemberRepository
from app.repositories.scan_log_repository import ScanLogRepository
from app.repositories.user_repository import UserRepository
from app.schemas.auth import TokenData
from app.schemas.claim import (
    ClaimDetailResponse,
    ClaimFileResponse,
    ClaimMedicine,
    ClaimMedicineInput,
    ClaimResponse,
    parse_claim_date,
)
from app.services.ai_service import DocumentImageInput, DocumentValidationProcessor
from app.services.ai_service import build_patient_name_verification_error
from app.services.ai_service import matches_any_patient_name_candidate
from app.services.claim_medicine_relevance import evaluate_claim_medicines, persist_claim_medicine_knowledge
from app.services.image_preprocessing import preprocess_document_image
from app.services.inhealth_claim_form import InHealthClaimFormData, build_inhealth_claim_form_pdf
from app.services.profile_completion import get_missing_family_profile_fields, get_missing_user_profile_fields

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


@dataclass(frozen=True)
class ClaimPatientSnapshot:
    selected_patient_type: str
    selected_family_member_id: int | None
    patient_relation: str
    patient_card_number: str | None
    patient_phone_number: str | None
    patient_mobile_number: str | None
    patient_email: str | None
    user_full_name: str
    user_nip: str | None
    user_company_name: str | None
    user_bank_account_number: str | None
    user_bank_account_holder_name: str | None
    user_bank_name: str | None
    insurance_card_filename: str | None
    insurance_card_mime_type: str | None
    insurance_card_byte_size: int | None
    insurance_card_sha256_hash: str | None
    insurance_card_content: bytes | None


def build_data_url(mime_type: str, content: bytes) -> str:
    return f"data:{mime_type};base64,{base64.b64encode(content).decode('utf-8')}"


def build_profile_required_error(message: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        detail={
            "message": "Input validation failed.",
            "errors": [{"field": "profile", "message": message}],
        },
    )


def normalize_patient_relation(value: str | None) -> str | None:
    normalized = (value or "").strip().lower()
    relation_map = {
        "self": "self",
        "saya": "self",
        "peserta": "self",
        "husband": "husband",
        "suami": "husband",
        "wife": "wife",
        "istri": "wife",
        "child": "child",
        "anak": "child",
        "son": "child",
        "daughter": "child",
    }
    return relation_map.get(normalized)


def build_patient_card_file(snapshot: ClaimPatientSnapshot) -> ClaimFileCreate | None:
    if not snapshot.insurance_card_content:
        return None

    filename = snapshot.insurance_card_filename or "insurance_card"
    return ClaimFileCreate(
        filename=f"patient_card_{filename}",
        mime_type=snapshot.insurance_card_mime_type or "application/octet-stream",
        byte_size=snapshot.insurance_card_byte_size or len(snapshot.insurance_card_content),
        sha256_hash=snapshot.insurance_card_sha256_hash or hashlib.sha256(snapshot.insurance_card_content).hexdigest(),
        content=snapshot.insurance_card_content,
    )


def resolve_claim_patient_snapshot(
    db: Session,
    *,
    token_data: TokenData,
    nama_pasien: str,
    selected_patient_type: str | None,
    selected_family_member_id: int | None,
) -> ClaimPatientSnapshot:
    user_repository = UserRepository(db)
    user = user_repository.get_by_id(token_data.user_id)
    if user is None:
        raise build_profile_required_error("User profile was not found.")

    family_member_repository = FamilyMemberRepository(db)
    selected_type = (selected_patient_type or "").strip().lower()

    if selected_type == "self":
        if not matches_any_patient_name_candidate(nama_pasien, [user.full_name]):
            raise build_profile_required_error("Selected patient does not match the user profile.")
        missing_fields = get_missing_user_profile_fields(user)
        if missing_fields:
            raise build_profile_required_error(f"Complete user profile before submitting claim: {', '.join(missing_fields)}.")
        return ClaimPatientSnapshot(
            selected_patient_type="self",
            selected_family_member_id=None,
            patient_relation="self",
            patient_card_number=user.insurance_card_number,
            patient_phone_number=user.phone_number,
            patient_mobile_number=user.mobile_number,
            patient_email=user.email,
            user_full_name=user.full_name,
            user_nip=user.nip,
            user_company_name=user.company_name,
            user_bank_account_number=user.bank_account_number,
            user_bank_account_holder_name=user.bank_account_holder_name,
            user_bank_name=user.bank_name,
            insurance_card_filename=user.insurance_card_filename,
            insurance_card_mime_type=user.insurance_card_mime_type,
            insurance_card_byte_size=user.insurance_card_byte_size,
            insurance_card_sha256_hash=user.insurance_card_sha256_hash,
            insurance_card_content=user.insurance_card_content,
        )

    if selected_type == "family":
        if selected_family_member_id is None:
            raise build_profile_required_error("Family member ID is required for family patient claims.")
        family_member = family_member_repository.get_by_id_and_user_id(selected_family_member_id, token_data.user_id)
        if family_member is None:
            raise build_profile_required_error("Selected family member was not found.")
        if not matches_any_patient_name_candidate(nama_pasien, [family_member.name]):
            raise build_profile_required_error("Selected patient does not match the family member profile.")
        patient_relation = normalize_patient_relation(family_member.relation)
        if patient_relation not in {"husband", "wife", "child"}:
            raise build_profile_required_error("Family member relation must be husband, wife, or child.")
        missing_user_fields = get_missing_user_profile_fields(user)
        missing_family_fields = get_missing_family_profile_fields(family_member)
        if missing_user_fields or missing_family_fields:
            missing_parts = []
            if missing_user_fields:
                missing_parts.append(f"user profile: {', '.join(missing_user_fields)}")
            if missing_family_fields:
                missing_parts.append(f"{family_member.name}: {', '.join(missing_family_fields)}")
            raise build_profile_required_error(f"Complete profile before submitting claim: {'; '.join(missing_parts)}.")
        return ClaimPatientSnapshot(
            selected_patient_type="family",
            selected_family_member_id=family_member.id,
            patient_relation=patient_relation,
            patient_card_number=family_member.insurance_card_number,
            patient_phone_number=family_member.phone_number,
            patient_mobile_number=family_member.mobile_number,
            patient_email=family_member.email,
            user_full_name=user.full_name,
            user_nip=user.nip,
            user_company_name=user.company_name,
            user_bank_account_number=user.bank_account_number,
            user_bank_account_holder_name=user.bank_account_holder_name,
            user_bank_name=user.bank_name,
            insurance_card_filename=family_member.insurance_card_filename,
            insurance_card_mime_type=family_member.insurance_card_mime_type,
            insurance_card_byte_size=family_member.insurance_card_byte_size,
            insurance_card_sha256_hash=family_member.insurance_card_sha256_hash,
            insurance_card_content=family_member.insurance_card_content,
        )

    family_members = family_member_repository.get_by_user_id(token_data.user_id)
    candidates: list[tuple[str, int | None]] = []
    if matches_any_patient_name_candidate(nama_pasien, [user.full_name]):
        candidates.append(("self", None))
    for family_member in family_members:
        if matches_any_patient_name_candidate(nama_pasien, [family_member.name]):
            candidates.append(("family", family_member.id))

    if len(candidates) != 1:
        raise build_profile_required_error("Selected patient identity is required for this claim.")

    fallback_type, fallback_family_member_id = candidates[0]
    return resolve_claim_patient_snapshot(
        db,
        token_data=token_data,
        nama_pasien=nama_pasien,
        selected_patient_type=fallback_type,
        selected_family_member_id=fallback_family_member_id,
    )


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
        files=[
            ClaimFileResponse(
                id=claim_file.id,
                filename=claim_file.filename,
                mime_type=claim_file.mime_type,
                byte_size=claim_file.byte_size,
                sha256_hash=claim_file.sha256_hash,
                file_data_url=build_data_url(claim_file.mime_type, claim_file.content),
            )
            for claim_file in claim.files
        ],
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


def build_claim_export_error(message: str, *, status_code: int = status.HTTP_422_UNPROCESSABLE_CONTENT) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={
            "message": "Input validation failed.",
            "errors": [{"field": "inhealth_form", "message": message}],
        },
    )


def build_inhealth_form_data(db: Session, claim: Any, token_data: TokenData) -> InHealthClaimFormData:
    if claim.patient_relation and claim.user_full_name_snapshot:
        form_data = InHealthClaimFormData(
            employee_name=claim.user_full_name_snapshot or "",
            nip=claim.user_nip_snapshot or "",
            company_name=claim.user_company_name_snapshot or "",
            patient_name=claim.nama_pasien,
            patient_relation=claim.patient_relation or "",
            patient_card_number=claim.patient_card_number or "",
            patient_phone_number=claim.patient_phone_number or "",
            patient_mobile_number=claim.patient_mobile_number or "",
            patient_email=claim.patient_email or "",
            bank_account_number=claim.user_bank_account_number_snapshot or "",
            bank_account_holder_name=claim.user_bank_account_holder_name_snapshot or "",
            bank_name=claim.user_bank_name_snapshot or "",
            claim_amount=claim.nominal_klaim,
            receipt_date=claim.tanggal_pelayanan,
            export_place="Surabaya",
            export_date=datetime.now(timezone.utc).date(),
            signature_name=claim.user_full_name_snapshot or "",
        )
    else:
        snapshot = resolve_claim_patient_snapshot(
            db,
            token_data=token_data,
            nama_pasien=claim.nama_pasien,
            selected_patient_type=claim.selected_patient_type,
            selected_family_member_id=claim.selected_family_member_id,
        )
        form_data = InHealthClaimFormData(
            employee_name=snapshot.user_full_name,
            nip=snapshot.user_nip or "",
            company_name=snapshot.user_company_name or "",
            patient_name=claim.nama_pasien,
            patient_relation=snapshot.patient_relation,
            patient_card_number=snapshot.patient_card_number or "",
            patient_phone_number=snapshot.patient_phone_number or "",
            patient_mobile_number=snapshot.patient_mobile_number or "",
            patient_email=snapshot.patient_email or "",
            bank_account_number=snapshot.user_bank_account_number or "",
            bank_account_holder_name=snapshot.user_bank_account_holder_name or "",
            bank_name=snapshot.user_bank_name or "",
            claim_amount=claim.nominal_klaim,
            receipt_date=claim.tanggal_pelayanan,
            export_place="Surabaya",
            export_date=datetime.now(timezone.utc).date(),
            signature_name=snapshot.user_full_name,
        )

    missing_fields = [
        label
        for label, value in (
            ("Nama karyawan", form_data.employee_name),
            ("NIP", form_data.nip),
            ("Nama perusahaan", form_data.company_name),
            ("Nama pasien", form_data.patient_name),
            ("Relasi pasien", form_data.patient_relation),
            ("No. kartu", form_data.patient_card_number),
            ("No. telepon", form_data.patient_phone_number),
            ("HP", form_data.patient_mobile_number),
            ("Email", form_data.patient_email),
            ("No. rekening", form_data.bank_account_number),
            ("Atas nama rekening", form_data.bank_account_holder_name),
            ("Bank", form_data.bank_name),
        )
        if not str(value or "").strip()
    ]
    if missing_fields:
        raise build_claim_export_error(f"Cannot export InHealth form. Missing: {', '.join(missing_fields)}.")

    return form_data


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
    selected_patient_type: str | None = Form(default=None),
    family_member_id: int | None = Form(default=None),
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
    patient_snapshot: ClaimPatientSnapshot | None = None
    if selected_patient_type is not None and selected_patient_type.strip():
        patient_snapshot = resolve_claim_patient_snapshot(
            db,
            token_data=token_data,
            nama_pasien=nama_pasien,
            selected_patient_type=selected_patient_type,
            selected_family_member_id=family_member_id,
        )
    else:
        try:
            patient_snapshot = resolve_claim_patient_snapshot(
                db,
                token_data=token_data,
                nama_pasien=nama_pasien,
                selected_patient_type=None,
                selected_family_member_id=None,
            )
        except HTTPException:
            patient_snapshot = None

    evaluated_medicines, requires_additional_review, review_note = await evaluate_claim_medicines(
        db,
        medicines_input,
        keterangan_penyakit,
        kronologi,
        user_id=token_data.user_id,
        scan_log_id=source_scan_log_id,
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
    claim_files = [
        ClaimFileCreate(
            filename=upload.filename,
            mime_type=upload.mime_type,
            byte_size=upload.byte_size,
            sha256_hash=upload.sha256_hash,
            content=upload.content,
        )
        for upload in buffered_uploads
    ]
    if patient_snapshot is not None:
        patient_card_file = build_patient_card_file(patient_snapshot)
        if patient_card_file is not None:
            claim_files.append(patient_card_file)

    claim = claim_repository.create(
        user_id=token_data.user_id,
        user_email=token_data.email,
        nama_pasien=nama_pasien,
        selected_patient_type=patient_snapshot.selected_patient_type if patient_snapshot else None,
        selected_family_member_id=patient_snapshot.selected_family_member_id if patient_snapshot else None,
        patient_relation=patient_snapshot.patient_relation if patient_snapshot else None,
        patient_card_number=patient_snapshot.patient_card_number if patient_snapshot else None,
        patient_phone_number=patient_snapshot.patient_phone_number if patient_snapshot else None,
        patient_mobile_number=patient_snapshot.patient_mobile_number if patient_snapshot else None,
        patient_email=patient_snapshot.patient_email if patient_snapshot else None,
        user_full_name_snapshot=patient_snapshot.user_full_name if patient_snapshot else None,
        user_nip_snapshot=patient_snapshot.user_nip if patient_snapshot else None,
        user_company_name_snapshot=patient_snapshot.user_company_name if patient_snapshot else None,
        user_bank_account_number_snapshot=patient_snapshot.user_bank_account_number if patient_snapshot else None,
        user_bank_account_holder_name_snapshot=patient_snapshot.user_bank_account_holder_name if patient_snapshot else None,
        user_bank_name_snapshot=patient_snapshot.user_bank_name if patient_snapshot else None,
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
        files=claim_files,
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


@router.get("/{claim_id}/inhealth-form")
async def download_inhealth_claim_form(
    claim_id: int,
    token_data: TokenData = Depends(get_current_user_dependency),
    db: Session = Depends(get_db_session),
) -> Response:
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
            detail="You do not have permission to export this claim.",
        )

    form_data = build_inhealth_form_data(db, claim, token_data)
    try:
        pdf_bytes = build_inhealth_claim_form_pdf(form_data)
    except FileNotFoundError as exc:
        raise build_claim_export_error("InHealth claim form template is missing.", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR) from exc
    except RuntimeError as exc:
        raise build_claim_export_error(str(exc), status_code=status.HTTP_500_INTERNAL_SERVER_ERROR) from exc
    except ValueError as exc:
        raise build_claim_export_error("InHealth claim form template is invalid.", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR) from exc

    filename = f"inhealth-claim-{claim.id}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


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
