from pydantic import BaseModel, Field


class AIModelResponse(BaseModel):
    id: int
    display_name: str
    model_name: str
    provider: str
    is_active: bool


class AIModelUpdateRequest(BaseModel):
    is_active: bool = Field()
