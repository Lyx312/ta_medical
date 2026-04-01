from collections.abc import Generator
from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings


def normalize_database_url(database_url: str) -> str:
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+psycopg://", 1)
    return database_url


def _build_engine(database_url: str) -> Engine:
    normalized_database_url = normalize_database_url(database_url)
    connect_args: dict[str, object] = {}
    if normalized_database_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
    return create_engine(normalized_database_url, future=True, connect_args=connect_args)


@lru_cache
def get_engine() -> Engine:
    return _build_engine(get_settings().database_url)


@lru_cache
def get_session_factory() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), autoflush=False, autocommit=False, expire_on_commit=False)


def get_db_session() -> Generator[Session, None, None]:
    db = get_session_factory()()
    try:
        yield db
    finally:
        db.close()


def clear_database_caches() -> None:
    try:
        get_engine().dispose()
    except Exception:
        pass
    get_session_factory.cache_clear()
    get_engine.cache_clear()
