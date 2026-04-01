from sqlalchemy import select

from app.db.base import Base
from app.models.ai_model import AIModel
from app.models.family_member import FamilyMember
from app.db.session import get_engine, get_session_factory
from app.models.user import User, hash_password
from app.repositories.ai_model_repository import AIModelRepository, SeedAIModelRecord

DEFAULT_AI_MODEL_CATALOG = [
    SeedAIModelRecord(
        display_name="Gemini 3.1 Flash Lite Preview",
        model_name="gemini-3.1-flash-lite-preview",
        provider="google_genai",
        is_active=True,
    ),
    SeedAIModelRecord(
        display_name="Gemini 3 Flash Preview",
        model_name="gemini-3-flash-preview",
        provider="google_genai",
        is_active=True,
    ),
]


def initialize_database() -> None:
    Base.metadata.create_all(bind=get_engine())
    seed_default_admin()
    seed_default_ai_models()


def seed_default_admin() -> None:
    session_factory = get_session_factory()
    with session_factory() as db:
        existing_admin = db.execute(
            select(User).where(User.email == "admin@mail.com")
        ).scalar_one_or_none()
        if existing_admin is not None:
            return

        db.add(
            User(
                full_name="Admin User",
                email="admin@mail.com",
                password_hash=hash_password("admin123"),
                role="admin",
            )
        )
        db.commit()


def seed_default_ai_models() -> None:
    session_factory = get_session_factory()
    with session_factory() as db:
        repository = AIModelRepository(db)
        repository.upsert_seed_catalog(DEFAULT_AI_MODEL_CATALOG)
