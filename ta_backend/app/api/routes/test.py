import base64

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app.core.config import get_settings
from app.schemas.test import (
    EchoRequest,
    EchoResponse,
    GreetRequest,
    GreetResponse,
    HealthResponse,
    ModelInvokeTestResponse,
)
from app.services.ai_service import DocumentImageInput
from app.services.greeting_service import GreetingService
from app.services.model_invocation_test import ModelInvocationTestService

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


@router.post("/invoke-model", response_model=ModelInvokeTestResponse)
async def invoke_model_for_testing(
    message: str = Form(...),
    provider: str = Form(...),
    model_name: str = Form(...),
    image: UploadFile | None = File(default=None),
) -> ModelInvokeTestResponse:
    normalized_message = message.strip()
    if not normalized_message:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={
                "message": "Input validation failed.",
                "errors": [
                    {
                        "field": "message",
                        "message": "Message must not be empty.",
                    }
                ],
            },
        )

    document_image: DocumentImageInput | None = None
    if image is not None:
        image_bytes = await image.read()
        document_image = DocumentImageInput(
            filename=image.filename or "uploaded-image",
            mime_type=image.content_type or "application/octet-stream",
            base64_data=base64.b64encode(image_bytes).decode("utf-8"),
        )

    service = ModelInvocationTestService(settings=get_settings())
    return await service.invoke_model(
        provider=provider.strip(),
        model_name=model_name.strip(),
        message=normalized_message,
        image=document_image,
    )
