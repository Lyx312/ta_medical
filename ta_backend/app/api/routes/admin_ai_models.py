from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.routes.auth import get_current_admin_dependency
from app.db.session import get_db_session
from app.repositories.ai_model_repository import AIModelRepository
from app.schemas.ai_model import AIModelResponse, AIModelUpdateRequest
from app.schemas.auth import TokenData

router = APIRouter(prefix="/admin/ai-models", tags=["admin-ai-models"])


@router.get("", response_model=list[AIModelResponse])
async def get_ai_models(
    token_data: TokenData = Depends(get_current_admin_dependency),
    db: Session = Depends(get_db_session),
) -> list[AIModelResponse]:
    del token_data

    repository = AIModelRepository(db)
    models = repository.list_all()
    return [
        AIModelResponse(
            id=model.id,
            display_name=model.display_name,
            model_name=model.model_name,
            provider=model.provider,
            is_active=model.is_active,
        )
        for model in models
    ]


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

    updated_model = repository.update_active_status(ai_model, is_active=request.is_active)
    return AIModelResponse(
        id=updated_model.id,
        display_name=updated_model.display_name,
        model_name=updated_model.model_name,
        provider=updated_model.provider,
        is_active=updated_model.is_active,
    )
