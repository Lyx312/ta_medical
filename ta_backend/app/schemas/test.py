from typing import Any

from pydantic import BaseModel, Field


class EchoRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1000)


class EchoResponse(BaseModel):
    echoed_text: str


class HealthResponse(BaseModel):
    status: str
    service: str


class GreetRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)


class GreetResponse(BaseModel):
    message: str


class ModelInvokeTestResponse(BaseModel):
    success: bool
    provider: str
    model_name: str
    supports_image_input: bool
    image_supplied: bool
    latency_ms: int
    raw_text: str
    parsed_json: dict[str, Any] | list[Any] | None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
