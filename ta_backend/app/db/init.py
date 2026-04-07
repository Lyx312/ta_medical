from typing import Any

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
    ensure_scan_log_schema()
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


def ensure_scan_log_schema() -> None:
    """
    Version-aware migration for scan_logs table.

    Detects legacy schema (has ai_model_provider, ai_model_name, ai_model_runs_payload)
    and rebuilds the table with new schema (primary_model_provider, primary_model_name).
    Creates scan_log_ai_runs table for structured per-run data.
    Does NOT backfill ai_model_runs_payload data - historical run debug data is discarded.
    """
    engine = get_engine()
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())

    if "scan_logs" not in tables:
        return

    column_names = {column["name"] for column in inspector.get_columns("scan_logs")}

    is_legacy = "ai_model_provider" in column_names
    has_new_schema = "primary_model_provider" in column_names

    if has_new_schema and not is_legacy:
        _ensure_scan_log_ai_runs_table(engine, inspector)
        return

    if not is_legacy:
        _ensure_scan_log_ai_runs_table(engine, inspector)
        return

    _migrate_scan_logs_to_new_schema(engine)
    _ensure_scan_log_ai_runs_table(engine, inspector)


def _ensure_scan_log_ai_runs_table(engine: Any, inspector: Any) -> None:
    """Create scan_log_ai_runs table if it doesn't exist."""
    tables = set(inspector.get_table_names())
    if "scan_log_ai_runs" in tables:
        return

    with engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE TABLE scan_log_ai_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scan_log_id INTEGER NOT NULL REFERENCES scan_logs(id),
                    producer_id VARCHAR(200) NOT NULL,
                    provider VARCHAR(50) NOT NULL,
                    model_name VARCHAR(150) NOT NULL,
                    producer_rank INTEGER NOT NULL,
                    run_index INTEGER NOT NULL,
                    temperature REAL NOT NULL,
                    success BOOLEAN NOT NULL,
                    error_message TEXT,
                    payload JSON,
                    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
        )
        connection.execute(text("CREATE INDEX ix_scan_log_ai_runs_scan_log_id ON scan_log_ai_runs(scan_log_id)"))


def _migrate_scan_logs_to_new_schema(engine: Any) -> None:
    """
    Rebuild scan_logs table with new schema, preserving main scan-log data.

    SQLite does not support DROP COLUMN, so we rebuild the table:
    1. Create scan_logs_new with new schema
    2. Copy data (mapping ai_model_provider -> primary_model_provider, etc.)
    3. Drop old scan_logs
    4. Rename scan_logs_new to scan_logs
    5. Recreate indexes
    """
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE TABLE scan_logs_new (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL REFERENCES users(id),
                    status VARCHAR(50) NOT NULL DEFAULT 'pending',
                    ai_execution_mode VARCHAR(50),
                    primary_model_provider VARCHAR(50),
                    primary_model_name VARCHAR(150),
                    validation_payload JSON,
                    extraction_payload JSON,
                    extraction_meta_payload JSON,
                    error_payload JSON,
                    created_at DATETIME NOT NULL,
                    updated_at DATETIME NOT NULL
                )
                """
            )
        )

        connection.execute(
            text(
                """
                INSERT INTO scan_logs_new (
                    id, user_id, status, ai_execution_mode,
                    primary_model_provider, primary_model_name,
                    validation_payload, extraction_payload,
                    extraction_meta_payload, error_payload,
                    created_at, updated_at
                )
                SELECT
                    id, user_id, status, ai_execution_mode,
                    ai_model_provider, ai_model_name,
                    validation_payload, extraction_payload,
                    extraction_meta_payload, error_payload,
                    created_at, updated_at
                FROM scan_logs
                """
            )
        )

        connection.execute(text("DROP TABLE scan_logs"))

        connection.execute(text("ALTER TABLE scan_logs_new RENAME TO scan_logs"))

        connection.execute(text("CREATE INDEX ix_scan_logs_user_id ON scan_logs(user_id)"))
