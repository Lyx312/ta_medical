from app.core.config import Settings


class GreetingService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def build_greeting(self, name: str) -> str:
        return f"Hello {name}! Welcome to {self._settings.app_name} ({self._settings.app_env})."
