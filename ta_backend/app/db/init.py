from sqlalchemy import inspect, select, text

from app.db.base import Base
from app.db.session import get_engine, get_session_factory
from app.models.user import User, hash_password
from app.repositories.ai_model_repository import AIModelRepository, SeedAIModelRecord

DEFAULT_AI_MODEL_CATALOG = [
    SeedAIModelRecord(
        display_name="Gemini 3.1 Flash Lite Preview",
        model_name="gemini-3.1-flash-lite-preview",
        provider="google_genai",
        priority=10,
        is_active=True,
    ),
    SeedAIModelRecord(
        display_name="Gemini 3 Flash Preview",
        model_name="gemini-3-flash-preview",
        provider="google_genai",
        priority=20,
        is_active=True,
    ),
    SeedAIModelRecord(
        display_name="OpenRouter Llama 3.2 11B Vision Instruct",
        model_name="meta-llama/llama-3.2-11b-vision-instruct",
        provider="openrouter",
        priority=30,
        is_active=False,
    ),
    SeedAIModelRecord(
        display_name="Ollama Llama 3.2 Vision",
        model_name="llama3.2-vision",
        provider="ollama",
        priority=40,
        is_active=False,
    ),
    SeedAIModelRecord(
        display_name="Groq Llama 3.3 70B Versatile",
        model_name="llama-3.3-70b-versatile",
        provider="groq",
        priority=50,
        is_active=False,
    ),
    SeedAIModelRecord(
        display_name="Mistral Small Latest",
        model_name="mistral-small-latest",
        provider="mistralai",
        priority=60,
        is_active=False,
    ),
    SeedAIModelRecord(
        display_name="DeepSeek Chat",
        model_name="deepseek-chat",
        provider="deepseek",
        priority=70,
        is_active=False,
    ),
    SeedAIModelRecord(
        display_name="Cohere Command R Plus",
        model_name="command-r-plus",
        provider="cohere",
        priority=80,
        is_active=False,
    ),
    SeedAIModelRecord(
        display_name="Cerebras Llama 3.3 70B",
        model_name="llama-3.3-70b",
        provider="cerebras",
        priority=90,
        is_active=False,
    ),
]


def initialize_database() -> None:
    Base.metadata.create_all(bind=get_engine())
    ensure_ai_model_schema()
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


def ensure_ai_model_schema() -> None:
    engine = get_engine()
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    if "ai_models" not in tables:
        return

    column_names = {column["name"] for column in inspector.get_columns("ai_models")}
    if "priority" in column_names:
        return

    with engine.begin() as connection:
        connection.execute(text("ALTER TABLE ai_models ADD COLUMN priority INTEGER NOT NULL DEFAULT 100"))
