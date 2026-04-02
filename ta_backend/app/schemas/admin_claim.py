from typing import Any

from pydantic import BaseModel

from app.schemas.claim import ClaimResponse


class AdminClaimFileResponse(BaseModel):
    id: int
    filename: str
    mime_type: str
    byte_size: int
    sha256_hash: str
    file_data_url: str


class AdminClaimDetailResponse(ClaimResponse):
    files: list[AdminClaimFileResponse]
    metadata: dict[str, Any] | None = None
