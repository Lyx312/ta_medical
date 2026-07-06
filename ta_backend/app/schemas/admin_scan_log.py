from datetime import datetime
from typing import Any

from pydantic import BaseModel


class AdminScanLogListItemResponse(BaseModel):
    id: int
    user_id: int
    user_full_name: str
    user_email: str
    status: str
    file_count: int
    linked_claim_ids: list[int]
    primary_model_provider: str | None
    primary_model_name: str | None
    created_at: datetime
    updated_at: datetime


class AdminScanLogFileResponse(BaseModel):
    id: int
    original_filename: str
    original_mime_type: str
    original_byte_size: int
    original_sha256_hash: str
    original_image_data_url: str
    processed_filename: str | None
    processed_mime_type: str | None
    processed_byte_size: int | None
    processed_sha256_hash: str | None
    processed_image_data_url: str | None


class AdminScanLogAIRunResponse(BaseModel):
    producer_id: str
    provider: str
    model_name: str
    producer_rank: int
    run_index: int
    temperature: float
    success: bool
    error_message: str | None
    payload: dict[str, Any] | None


class AdminAIApiCallLogResponse(BaseModel):
    provider: str
    model_name: str
    call_kind: str
    producer_id: str | None
    producer_rank: int | None
    run_index: int | None
    temperature: float | None
    success: bool
    error_message: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    latency_ms: int | None
    usage_payload: dict[str, Any] | None
    request_context_payload: dict[str, Any] | None
    prompt_file_path: str | None
    prompt_payload: Any | None
    raw_response_text: str | None
    response_content_payload: Any | None
    parsed_response_payload: Any | None
    response_validation_error: str | None
    result_context_payload: dict[str, Any] | None
    created_at: datetime


class AdminScanLogDetailResponse(AdminScanLogListItemResponse):
    ai_execution_mode: str | None = None
    validation_payload: dict[str, Any] | None
    extraction_payload: dict[str, Any] | None
    ai_model_runs: list[AdminScanLogAIRunResponse] = []
    ai_api_calls: list[AdminAIApiCallLogResponse] = []
    extraction_meta_payload: dict[str, Any] | None = None
    error_payload: dict[str, Any] | None
    files: list[AdminScanLogFileResponse]


class AdminScanLogSummaryResponse(BaseModel):
    total_logs: int
    succeeded_count: int
    failed_count: int
    unlinked_count: int
