from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field, field_serializer, field_validator

CLAIM_DATE_FORMAT = "%d-%m-%Y"


def parse_claim_date(value: date | str) -> date:
    if isinstance(value, date):
        return value

    return datetime.strptime(value, CLAIM_DATE_FORMAT).date()


class ClaimCreate(BaseModel):
    nama_pasien: str = Field(min_length=1, max_length=255)
    tanggal_pelayanan: date
    fasilitas_kesehatan: str = Field(min_length=1, max_length=255)
    jenis_manfaat: Literal["rawat_jalan", "rawat_inap", "emergency", "maternity"]
    nominal_klaim: float = Field(ge=0)
    keterangan_penyakit: str = Field(min_length=1)
    kronologi: str = Field(min_length=1)

    @field_validator("tanggal_pelayanan", mode="before")
    @classmethod
    def validate_tanggal_pelayanan(cls, value: date | str) -> date:
        return parse_claim_date(value)


class ClaimResponse(BaseModel):
    id: int
    user_id: int
    user_email: str
    nama_pasien: str
    tanggal_pelayanan: date
    fasilitas_kesehatan: str
    jenis_manfaat: str
    nominal_klaim: float
    keterangan_penyakit: str
    kronologi: str
    file_count: int
    status: str
    created_at: datetime

    @field_serializer("tanggal_pelayanan")
    def serialize_tanggal_pelayanan(self, value: date) -> str:
        return value.strftime(CLAIM_DATE_FORMAT)
