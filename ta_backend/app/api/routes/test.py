from fastapi import APIRouter

from app.core.config import get_settings
from app.schemas.test import (
    EchoRequest,
    EchoResponse,
    GreetRequest,
    GreetResponse,
    HealthResponse,
)
from app.services.greeting_service import GreetingService

router = APIRouter(prefix="/test", tags=["test"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    return HealthResponse(status="ok", service=get_settings().app_name)


@router.post("/echo", response_model=EchoResponse)
async def echo_message(request: EchoRequest) -> EchoResponse:
    return EchoResponse(echoed_text=request.text)


@router.post("/greet", response_model=GreetResponse)
async def greet_user(request: GreetRequest) -> GreetResponse:
    settings = get_settings()
    greeting_service = GreetingService(settings=settings)
    message = greeting_service.build_greeting(name=request.name)
    return GreetResponse(message=message)
