from typing import Any

from pydantic import BaseModel

from app.schemas.claim import ClaimDetailResponse
from app.schemas.claim import ClaimFileResponse
from app.schemas.claim import ClaimResponse


class AdminClaimFileResponse(ClaimFileResponse):
    pass


class AdminClaimDetailResponse(ClaimDetailResponse):
    files: list[AdminClaimFileResponse]
    metadata: dict[str, Any] | None = None


class AdminClaimSummaryResponse(BaseModel):
    total_claims: int
    total_claim_amount: float
    linked_scan_count: int
    manual_submission_count: int
