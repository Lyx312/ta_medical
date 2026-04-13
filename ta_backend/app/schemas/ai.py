from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class AIScanSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    detected_documents: list[str] = Field(default_factory=list)
    medical_facility_name: str | None = None
    service_date: str | None = None
    transaction_date: str | None = None
    currency: str | None = None
    total_amount: float | int | None = None


class AIScanPatientInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    date_of_birth: str | None = None
    age: float | int | None = None
    gender: str | None = None
    address: str | None = None


class AIScanDiagnosis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str | None = None
    doctor_name: str | None = None


class AIScanMedicalContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    anamnesis: str | None = None
    diagnoses: list["AIScanDiagnosis"] = Field(default_factory=list)
    therapy_note: str | None = None


class AIScanMedicine(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    quantity: float | int | None = None
    unit: str | None = None
    dosage_instruction: str | None = None
    unit_price: float | int | None = None
    total_price: float | int | None = None


class AIScanOtherCharge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: str | None = None
    description: str | None = None
    amount: float | int | None = None


class AIScanFinancialRecap(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_medicines: float | int | None = None
    total_other_charges: float | int | None = None
    grand_total: float | int | None = None


class AIMedicalScanResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: AIScanSummary
    patient_info: AIScanPatientInfo
    medical_context: AIScanMedicalContext
    medicines: list[AIScanMedicine]
    other_charges: list[AIScanOtherCharge]
    financial_recap: AIScanFinancialRecap


class AIScanDocumentsResponse(AIMedicalScanResponse):
    scan_log_id: int


ValidationDocumentType = Literal[
    "receipt",
    "medical_form",
    "doctor_note",
    "prescription",
    "lab_result",
    "radiology_result",
    "non_medical",
    "unknown",
]
ValidationDecision = Literal["valid", "invalid", "uncertain"]
ValidationPatientNameMatchStatus = Literal["matched", "mismatched", "not_found", "uncertain", "not_checked"]


class DocumentValidationItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_index: int = Field(ge=0)
    document_type: ValidationDocumentType
    is_medical_document: bool
    readability_ok: bool
    facility_identity_visible: bool
    stamp_detected: bool
    signature_detected: bool
    decision: ValidationDecision
    reasons: list[str]


class ImageValidationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_index: int = Field(ge=0)
    filename: str = Field(min_length=1)
    document_count_detected: int = Field(ge=0)
    documents: list[DocumentValidationItem]


class DocumentValidationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    images: list[ImageValidationResult]
    packet_patient_name: str | None = None
    patient_name_match_status: ValidationPatientNameMatchStatus = "not_checked"
    patient_name_match_reason: str | None = None
    review_required: bool = False
    review_reasons: list[str] = Field(default_factory=list)
