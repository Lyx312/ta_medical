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
