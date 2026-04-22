from datetime import date, datetime
from typing import Any
from typing import Literal

from pydantic import BaseModel, Field, field_serializer, field_validator

CLAIM_DATE_FORMAT = "%d-%m-%Y"


def parse_claim_date(value: date | str) -> date:
    if isinstance(value, date):
        return value

    return datetime.strptime(value, CLAIM_DATE_FORMAT).date()


class ClaimMedicineInput(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    quantity: float | None = Field(default=None, ge=0)
    unit: str | None = Field(default=None, max_length=50)
    dosage_instruction: str | None = Field(default=None, max_length=255)
    source: Literal["ai_extracted", "user_input"] = "user_input"


class ClaimMedicineEvaluation(BaseModel):
    relevance: Literal["relevant", "uncertain", "not_relevant"]
    confidence: float = Field(ge=0, le=1)
    reason: str = Field(min_length=1)
    matched_diagnosis: bool
    matched_anamnesis: bool
    decision_source: Literal["history_strong", "llm_judge", "heuristic_fallback"] | None = None


class ClaimMedicine(ClaimMedicineInput):
    evaluation: ClaimMedicineEvaluation | None = None


class ClaimCreate(BaseModel):
    nama_pasien: str = Field(min_length=1, max_length=255)
    tanggal_pelayanan: date
    fasilitas_kesehatan: str = Field(min_length=1, max_length=255)
    jenis_manfaat: Literal["rawat_jalan", "rawat_inap", "emergency", "maternity"]
    nominal_klaim: float = Field(ge=0)
    keterangan_penyakit: str = Field(min_length=1)
    kronologi: str = Field(min_length=1)
    medicines: list[ClaimMedicineInput] = Field(default_factory=list)

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
    medicines: list[ClaimMedicine] = Field(default_factory=list)
    requires_additional_review: bool = False
    review_note: str | None = None
    file_count: int
    status: str
    source_scan_log_id: int | None = None
    created_at: datetime

    @field_serializer("tanggal_pelayanan")
    def serialize_tanggal_pelayanan(self, value: date) -> str:
        return value.strftime(CLAIM_DATE_FORMAT)


class ClaimDetailResponse(ClaimResponse):
    source_scan_extraction_payload: dict[str, Any] | None = None
