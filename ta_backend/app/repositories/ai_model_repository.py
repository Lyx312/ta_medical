from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.ai_model import AIModel


@dataclass(frozen=True)
class SeedAIModelRecord:
    display_name: str
    model_name: str
    provider: str
    priority: int
    is_active: bool


class AIModelRepository:
    def __init__(self, db: Session) -> None:
        self._db = db

    def list_all(self) -> list[AIModel]:
        return list(
            self._db.execute(
                select(AIModel).order_by(AIModel.priority.asc(), AIModel.id.asc())
            ).scalars()
        )

    def list_active(self) -> list[AIModel]:
        return list(
            self._db.execute(
                select(AIModel)
                .where(AIModel.is_active.is_(True))
                .order_by(AIModel.priority.asc(), AIModel.id.asc())
            ).scalars()
        )

    def get_by_id(self, model_id: int) -> AIModel | None:
        return self._db.get(AIModel, model_id)

    def update_configuration(
        self,
        ai_model: AIModel,
        *,
        is_active: bool | None = None,
        priority: int | None = None,
    ) -> AIModel:
        if is_active is not None:
            ai_model.is_active = is_active
        if priority is not None:
            ai_model.priority = priority
        self._db.commit()
        self._db.refresh(ai_model)
        return ai_model

    def upsert_seed_catalog(self, seed_rows: list[SeedAIModelRecord]) -> None:
        for seed_row in seed_rows:
            existing_row = self._db.execute(
                select(AIModel).where(
                    AIModel.provider == seed_row.provider,
                    AIModel.model_name == seed_row.model_name,
                )
            ).scalar_one_or_none()

            if existing_row is None:
                self._db.add(
                    AIModel(
                        display_name=seed_row.display_name,
                        model_name=seed_row.model_name,
                        provider=seed_row.provider,
                        priority=seed_row.priority,
                        is_active=seed_row.is_active,
                    )
                )
                continue

            existing_row.display_name = seed_row.display_name
            existing_row.priority = seed_row.priority

        self._db.commit()
