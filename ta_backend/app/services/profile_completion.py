import base64
import hashlib
from dataclasses import dataclass
from io import BytesIO
from typing import Any

from fastapi import HTTPException, UploadFile, status
from PIL import Image, UnidentifiedImageError

USER_PROFILE_FIELDS: tuple[tuple[str, str], ...] = (
    ("nip", "NIP"),
    ("company_name", "Nama perusahaan"),
    ("insurance_card_number", "No. Kartu Asuransi"),
    ("phone_number", "Nomor telepon"),
    ("mobile_number", "Nomor HP"),
    ("bank_account_number", "Nomor rekening"),
    ("bank_account_holder_name", "Atas nama rekening"),
    ("bank_name", "Bank"),
)

FAMILY_PROFILE_FIELDS: tuple[tuple[str, str], ...] = (
    ("insurance_card_number", "No. Kartu Asuransi"),
    ("phone_number", "Nomor telepon"),
    ("mobile_number", "Nomor HP"),
    ("email", "Email"),
)


@dataclass(frozen=True)
class InsuranceCardUpload:
    filename: str
    mime_type: str
    content: bytes
    byte_size: int
    sha256_hash: str


def build_data_url(mime_type: str | None, content: bytes | None) -> str | None:
    if not mime_type or not content:
        return None
    return f"data:{mime_type};base64,{base64.b64encode(content).decode('utf-8')}"


def get_missing_user_profile_fields(user: Any) -> list[str]:
    missing_fields = [
        label
        for field_name, label in USER_PROFILE_FIELDS
        if _is_blank(getattr(user, field_name, None))
    ]
    if not getattr(user, "insurance_card_content", None):
        missing_fields.append("Foto kartu Asuransi")
    return missing_fields


def get_missing_family_profile_fields(family_member: Any) -> list[str]:
    missing_fields = [
        label
        for field_name, label in FAMILY_PROFILE_FIELDS
        if _is_blank(getattr(family_member, field_name, None))
    ]
    if not getattr(family_member, "insurance_card_content", None):
        missing_fields.append("Foto Kartu member asuransi")
    return missing_fields


async def read_insurance_card_upload(file: UploadFile) -> InsuranceCardUpload:
    content = await file.read()
    mime_type = file.content_type or "application/octet-stream"

    if not content or not mime_type.startswith("image/"):
        raise _invalid_image_error()

    try:
        with Image.open(BytesIO(content)) as image:
            image.verify()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise _invalid_image_error() from exc

    filename = (file.filename or "insurance_card").strip() or "insurance_card"
    return InsuranceCardUpload(
        filename=filename,
        mime_type=mime_type,
        content=content,
        byte_size=len(content),
        sha256_hash=hashlib.sha256(content).hexdigest(),
    )


def _is_blank(value: object) -> bool:
    return not isinstance(value, str) or not value.strip()


def _invalid_image_error() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        detail={
            "message": "Input validation failed.",
            "errors": [{"field": "insurance_card", "message": "Uploaded insurance card must be a readable image."}],
        },
    )
