from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class AIMedicalScanResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: dict[str, Any]
    patient_info: dict[str, Any]
    medical_context: dict[str, Any]
    medicines: list[dict[str, Any]]
    other_charges: list[dict[str, Any]]
    financial_recap: dict[str, Any]


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
