from pydantic import BaseModel, Field, model_validator


class AIModelResponse(BaseModel):
    id: int
    display_name: str
    model_name: str
    provider: str
    priority: int
    is_active: bool
    supports_image_input: bool
    is_scan_compatible: bool
    activation_block_reason: str | None = None


class AIModelUpdateRequest(BaseModel):
    is_active: bool | None = Field(default=None)
    priority: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_has_change(self) -> "AIModelUpdateRequest":
        if self.is_active is None and self.priority is None:
            raise ValueError("At least one field must be provided.")
        return self
