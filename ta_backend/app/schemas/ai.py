from typing import Any

from pydantic import BaseModel, ConfigDict


class AIMedicalScanResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: dict[str, Any]
    patient_info: dict[str, Any]
    medical_context: dict[str, Any]
    medicines: list[dict[str, Any]]
    other_charges: list[dict[str, Any]]
    financial_recap: dict[str, Any]
