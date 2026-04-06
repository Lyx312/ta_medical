import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    app_name: str
    app_env: str
    openai_api_key: str
    google_api_key: str
    jwt_secret: str
    access_token_expire_minutes: int
    database_url: str
    openrouter_api_key: str = ""
    ollama_api_key: str = ""
    groq_api_key: str = ""
    mistral_api_key: str = ""
    deepseek_api_key: str = ""
    cohere_api_key: str = ""
    cerebras_api_key: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings(
        app_name=os.getenv("APP_NAME", "python-backend"),
        app_env=os.getenv("APP_ENV", "dev"),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        google_api_key=os.getenv("GOOGLE_API_KEY", ""),
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
        ollama_api_key=os.getenv("OLLAMA_API_KEY", ""),
        groq_api_key=os.getenv("GROQ_API_KEY", ""),
        mistral_api_key=os.getenv("MISTRAL_API_KEY", ""),
        deepseek_api_key=os.getenv("DEEPSEEK_API_KEY", ""),
        cohere_api_key=os.getenv("COHERE_API_KEY", ""),
        cerebras_api_key=os.getenv("CEREBRAS_API_KEY", ""),
        jwt_secret=os.getenv("JWT_SECRET", "fallback_dev_secret_change_this"),
        access_token_expire_minutes=int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "10080")),
        database_url=os.getenv("DATABASE_URL", "sqlite:///./app.db"),
    )
