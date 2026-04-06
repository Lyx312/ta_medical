from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.routes.auth import get_current_admin_dependency
from app.db.session import get_db_session
from app.repositories.ai_model_repository import AIModelRepository
from app.schemas.ai_model import AIModelResponse, AIModelUpdateRequest
from app.schemas.auth import TokenData
from app.services.ai_provider_catalog import (
    get_ai_provider_catalog_entry,
    get_provider_activation_block_reason,
)

router = APIRouter(prefix="/admin/ai-models", tags=["admin-ai-models"])


def build_ai_model_response(model: object) -> AIModelResponse:
    provider = str(getattr(model, "provider"))
    provider_entry = get_ai_provider_catalog_entry(provider)
    return AIModelResponse(
        id=int(getattr(model, "id")),
        display_name=str(getattr(model, "display_name")),
        model_name=str(getattr(model, "model_name")),
        provider=provider,
        priority=int(getattr(model, "priority")),
        is_active=bool(getattr(model, "is_active")),
        supports_image_input=bool(provider_entry and provider_entry.supports_image_input),
        is_scan_compatible=bool(provider_entry and provider_entry.is_scan_compatible),
        activation_block_reason=get_provider_activation_block_reason(provider),
    )


@router.get("", response_model=list[AIModelResponse])
async def get_ai_models(
    token_data: TokenData = Depends(get_current_admin_dependency),
    db: Session = Depends(get_db_session),
) -> list[AIModelResponse]:
    del token_data

    repository = AIModelRepository(db)
    models = repository.list_all()
    return [build_ai_model_response(model) for model in models]


@router.put("/{model_id}", response_model=AIModelResponse)
async def update_ai_model(
    model_id: int,
    request: AIModelUpdateRequest,
    token_data: TokenData = Depends(get_current_admin_dependency),
    db: Session = Depends(get_db_session),
) -> AIModelResponse:
    del token_data

    repository = AIModelRepository(db)
    ai_model = repository.get_by_id(model_id)
    if ai_model is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="AI model not found.",
        )

    provider_entry = get_ai_provider_catalog_entry(ai_model.provider)
    if request.is_active is True and not (provider_entry and provider_entry.is_scan_compatible):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={
                "message": "Input validation failed.",
                "errors": [
                    {
                        "field": "is_active",
                        "message": get_provider_activation_block_reason(ai_model.provider)
                        or "This provider cannot be activated for the current scan pipeline.",
                    }
                ],
            },
        )

    updated_model = repository.update_configuration(
        ai_model,
        is_active=request.is_active,
        priority=request.priority,
    )
    return build_ai_model_response(updated_model)
