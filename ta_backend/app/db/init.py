from typing import Any

from sqlalchemy import inspect, select, text

from app.db.base import Base
from app.db.session import get_engine, get_session_factory
from app.models.scan_log import AIApiCallLog  # noqa: F401
from app.models.user import User, hash_password
from app.repositories.ai_model_repository import AIModelRepository, SeedAIModelRecord

DEFAULT_AI_MODEL_CATALOG = [
    SeedAIModelRecord(
        display_name="Gemini 3.1 Flash Lite",
        model_name="gemini-3.1-flash-lite",
        provider="google_genai",
        priority=10,
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
    ensure_user_profile_schema()
    ensure_family_member_profile_schema()
    ensure_ai_model_schema()
    ensure_scan_log_schema()
    ensure_ai_api_call_log_schema()
    ensure_claim_schema()
    ensure_claim_medicine_knowledge_schema()
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


def _binary_column_type() -> str:
    return "BYTEA" if get_engine().dialect.name == "postgresql" else "BLOB"


def ensure_user_profile_schema() -> None:
    engine = get_engine()
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    if "users" not in tables:
        return

    existing_columns = {column["name"] for column in inspector.get_columns("users")}
    desired_columns = {
        "nip": "VARCHAR(100)",
        "company_name": "VARCHAR(150)",
        "insurance_card_number": "VARCHAR(100)",
        "phone_number": "VARCHAR(50)",
        "mobile_number": "VARCHAR(50)",
        "bank_account_number": "VARCHAR(100)",
        "bank_account_holder_name": "VARCHAR(150)",
        "bank_name": "VARCHAR(100)",
        "insurance_card_filename": "VARCHAR(255)",
        "insurance_card_mime_type": "VARCHAR(100)",
        "insurance_card_byte_size": "INTEGER",
        "insurance_card_sha256_hash": "VARCHAR(64)",
        "insurance_card_content": _binary_column_type(),
    }

    with engine.begin() as connection:
        for column_name, column_type in desired_columns.items():
            if column_name not in existing_columns:
                connection.execute(text(f"ALTER TABLE users ADD COLUMN {column_name} {column_type}"))


def ensure_family_member_profile_schema() -> None:
    engine = get_engine()
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    if "family_members" not in tables:
        return

    existing_columns = {column["name"] for column in inspector.get_columns("family_members")}
    desired_columns = {
        "insurance_card_number": "VARCHAR(100)",
        "phone_number": "VARCHAR(50)",
        "mobile_number": "VARCHAR(50)",
        "email": "VARCHAR(255)",
        "insurance_card_filename": "VARCHAR(255)",
        "insurance_card_mime_type": "VARCHAR(100)",
        "insurance_card_byte_size": "INTEGER",
        "insurance_card_sha256_hash": "VARCHAR(64)",
        "insurance_card_content": _binary_column_type(),
    }

    with engine.begin() as connection:
        for column_name, column_type in desired_columns.items():
            if column_name not in existing_columns:
                connection.execute(text(f"ALTER TABLE family_members ADD COLUMN {column_name} {column_type}"))


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

    with engine.begin() as connection:
        if "progress_payload" not in column_names:
            connection.execute(text("ALTER TABLE scan_logs ADD COLUMN progress_payload JSON"))

    if has_new_schema and not is_legacy:
        _ensure_scan_log_ai_runs_table(engine, inspector)
        return

    if not is_legacy:
        _ensure_scan_log_ai_runs_table(engine, inspector)
        return

    _migrate_scan_logs_to_new_schema(engine)
    _ensure_scan_log_ai_runs_table(engine, inspector)


def ensure_claim_schema() -> None:
    engine = get_engine()
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())

    if "claims" not in tables:
        return

    column_names = {column["name"] for column in inspector.get_columns("claims")}
    with engine.begin() as connection:
        if "medicines" not in column_names:
            connection.execute(text("ALTER TABLE claims ADD COLUMN medicines TEXT"))
        if "medicines_payload" not in column_names:
            connection.execute(text("ALTER TABLE claims ADD COLUMN medicines_payload JSON"))
        if "requires_additional_review" not in column_names:
            connection.execute(text("ALTER TABLE claims ADD COLUMN requires_additional_review BOOLEAN NOT NULL DEFAULT FALSE"))
        if "review_note" not in column_names:
            connection.execute(text("ALTER TABLE claims ADD COLUMN review_note TEXT"))
        claim_snapshot_columns = {
            "selected_patient_type": "VARCHAR(20)",
            "selected_family_member_id": "INTEGER",
            "patient_relation": "VARCHAR(20)",
            "patient_card_number": "VARCHAR(100)",
            "patient_phone_number": "VARCHAR(50)",
            "patient_mobile_number": "VARCHAR(50)",
            "patient_email": "VARCHAR(255)",
            "user_full_name_snapshot": "VARCHAR(150)",
            "user_nip_snapshot": "VARCHAR(100)",
            "user_company_name_snapshot": "VARCHAR(150)",
            "user_bank_account_number_snapshot": "VARCHAR(100)",
            "user_bank_account_holder_name_snapshot": "VARCHAR(150)",
            "user_bank_name_snapshot": "VARCHAR(100)",
        }
        for snapshot_column_name, snapshot_column_type in claim_snapshot_columns.items():
            if snapshot_column_name not in column_names:
                connection.execute(text(f"ALTER TABLE claims ADD COLUMN {snapshot_column_name} {snapshot_column_type}"))

        if "requires_manual_review" in column_names:
            connection.execute(
                text(
                    """
                    UPDATE claims
                    SET requires_additional_review = TRUE
                    WHERE COALESCE(requires_additional_review, FALSE) = FALSE
                      AND COALESCE(requires_manual_review, FALSE) = TRUE
                    """
                )
            )
        if "manual_review_reason" in column_names:
            connection.execute(
                text(
                    """
                    UPDATE claims
                    SET review_note = manual_review_reason
                    WHERE (review_note IS NULL OR TRIM(review_note) = '')
                      AND manual_review_reason IS NOT NULL
                      AND TRIM(manual_review_reason) <> ''
                    """
                )
            )


def ensure_claim_medicine_knowledge_schema() -> None:
    engine = get_engine()
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())

    if "claim_medicine_knowledge" in tables:
        return

    with engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE TABLE claim_medicine_knowledge (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    context_signature VARCHAR(64) NOT NULL,
                    medicine_name_normalized VARCHAR(255) NOT NULL,
                    diagnosis_text_normalized TEXT NOT NULL,
                    anamnesis_text_normalized TEXT NOT NULL,
                    diagnosis_terms_json JSON NOT NULL,
                    anamnesis_terms_json JSON NOT NULL,
                    support_count INTEGER NOT NULL DEFAULT 0,
                    relevant_count REAL NOT NULL DEFAULT 0,
                    uncertain_count REAL NOT NULL DEFAULT 0,
                    not_relevant_count REAL NOT NULL DEFAULT 0,
                    avg_confidence REAL NOT NULL DEFAULT 0,
                    last_reason TEXT,
                    last_decision_source VARCHAR(32) NOT NULL DEFAULT 'heuristic_fallback',
                    first_seen_claim_id INTEGER,
                    last_seen_claim_id INTEGER,
                    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    last_used_at DATETIME,
                    CONSTRAINT uq_claim_medicine_knowledge_context_medicine
                        UNIQUE(context_signature, medicine_name_normalized)
                )
                """
            )
        )
        connection.execute(
            text(
                "CREATE INDEX ix_claim_medicine_knowledge_medicine_name_normalized "
                "ON claim_medicine_knowledge(medicine_name_normalized)"
            )
        )
        connection.execute(
            text(
                "CREATE INDEX ix_claim_medicine_knowledge_updated_at "
                "ON claim_medicine_knowledge(updated_at)"
            )
        )


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


def ensure_ai_api_call_log_schema() -> None:
    engine = get_engine()
    inspector = inspect(engine)
    _ensure_ai_api_call_logs_table(engine, inspector)


def _ensure_ai_api_call_logs_table(engine: Any, inspector: Any) -> None:
    tables = set(inspector.get_table_names())
    if "ai_api_call_logs" in tables:
        _ensure_ai_api_call_logs_columns(engine, inspector)
        return

    with engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE TABLE ai_api_call_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scan_log_id INTEGER REFERENCES scan_logs(id),
                    user_id INTEGER REFERENCES users(id),
                    provider VARCHAR(50) NOT NULL,
                    model_name VARCHAR(150) NOT NULL,
                    call_kind VARCHAR(80) NOT NULL,
                    producer_id VARCHAR(200),
                    producer_rank INTEGER,
                    run_index INTEGER,
                    temperature REAL,
                    success BOOLEAN NOT NULL,
                    error_message TEXT,
                    prompt_tokens INTEGER,
                    completion_tokens INTEGER,
                    total_tokens INTEGER,
                    latency_ms INTEGER,
                    usage_payload JSON,
                    request_context_payload JSON,
                    prompt_file_path VARCHAR(500),
                    prompt_payload JSON,
                    raw_response_text TEXT,
                    response_content_payload JSON,
                    parsed_response_payload JSON,
                    response_validation_error TEXT,
                    result_context_payload JSON,
                    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
        )
        connection.execute(text("CREATE INDEX ix_ai_api_call_logs_scan_log_id ON ai_api_call_logs(scan_log_id)"))
        connection.execute(text("CREATE INDEX ix_ai_api_call_logs_user_id ON ai_api_call_logs(user_id)"))
        connection.execute(text("CREATE INDEX ix_ai_api_call_logs_call_kind ON ai_api_call_logs(call_kind)"))


def _ensure_ai_api_call_logs_columns(engine: Any, inspector: Any) -> None:
    existing_columns = {column["name"] for column in inspector.get_columns("ai_api_call_logs")}
    desired_columns = {
        "prompt_file_path": "ALTER TABLE ai_api_call_logs ADD COLUMN prompt_file_path VARCHAR(500)",
        "prompt_payload": "ALTER TABLE ai_api_call_logs ADD COLUMN prompt_payload JSON",
        "raw_response_text": "ALTER TABLE ai_api_call_logs ADD COLUMN raw_response_text TEXT",
        "response_content_payload": "ALTER TABLE ai_api_call_logs ADD COLUMN response_content_payload JSON",
        "parsed_response_payload": "ALTER TABLE ai_api_call_logs ADD COLUMN parsed_response_payload JSON",
        "response_validation_error": "ALTER TABLE ai_api_call_logs ADD COLUMN response_validation_error TEXT",
        "result_context_payload": "ALTER TABLE ai_api_call_logs ADD COLUMN result_context_payload JSON",
    }

    missing_column_ddls = [
        ddl
        for column_name, ddl in desired_columns.items()
        if column_name not in existing_columns
    ]
    if not missing_column_ddls:
        return

    with engine.begin() as connection:
        for ddl in missing_column_ddls:
            connection.execute(text(ddl))


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
