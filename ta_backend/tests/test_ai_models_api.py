from pathlib import Path
from typing import Generator

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.db.session import clear_database_caches, get_session_factory
from app.main import create_app
from app.repositories.ai_model_repository import AIModelRepository


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient, None, None]:
    database_path = tmp_path / "ai-models-test.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{database_path.as_posix()}")
    monkeypatch.setenv("APP_ENV", "test")
    get_settings.cache_clear()
    clear_database_caches()

    app = create_app()
    with TestClient(app) as test_client:
        yield test_client

    clear_database_caches()
    get_settings.cache_clear()


def register_and_login_user(client: TestClient, email: str) -> str:
    register = client.post(
        "/api/auth/register",
        json={
            "full_name": "Normal User",
            "email": email,
            "password": "password123",
            "confirm_password": "password123",
        },
    )
    assert register.status_code == 201

    login = client.post(
        "/api/auth/login",
        json={"email": email, "password": "password123"},
    )
    assert login.status_code == 200
    return login.json()["access_token"]


def login_admin(client: TestClient) -> str:
    response = client.post(
        "/api/auth/login",
        json={"email": "admin@mail.com", "password": "admin123"},
    )
    assert response.status_code == 200
    return response.json()["access_token"]


def test_startup_seed_creates_ai_model_rows_once(client: TestClient) -> None:
    session_factory = get_session_factory()
    with session_factory() as db:
        repository = AIModelRepository(db)
        first_rows = repository.list_all()

    assert len(first_rows) == 1
    assert first_rows[0].display_name == "Gemini 3.1 Flash Lite Preview"
    assert first_rows[0].model_name == "gemini-3.1-flash-lite-preview"
    assert first_rows[0].provider == "google_genai"
    assert first_rows[0].is_active is True

    app = create_app()
    with TestClient(app):
        pass

    with session_factory() as db:
        repository = AIModelRepository(db)
        second_rows = repository.list_all()

    assert len(second_rows) == 1
    assert second_rows[0].id == first_rows[0].id


def test_admin_can_list_and_update_ai_models(client: TestClient) -> None:
    admin_token = login_admin(client)

    list_response = client.get(
        "/api/admin/ai-models",
        headers={"Authorization": f"Bearer {admin_token}"},
    )

    assert list_response.status_code == 200
    models = list_response.json()
    assert len(models) == 1
    assert models[0]["display_name"] == "Gemini 3.1 Flash Lite Preview"
    assert models[0]["is_active"] is True

    update_response = client.put(
        f"/api/admin/ai-models/{models[0]['id']}",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"is_active": False},
    )

    assert update_response.status_code == 200
    updated_model = update_response.json()
    assert updated_model["is_active"] is False

    final_list_response = client.get(
        "/api/admin/ai-models",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert final_list_response.status_code == 200
    assert final_list_response.json()[0]["is_active"] is False


def test_non_admin_cannot_access_ai_model_admin_routes(client: TestClient) -> None:
    user_token = register_and_login_user(client, "user@example.com")

    list_response = client.get(
        "/api/admin/ai-models",
        headers={"Authorization": f"Bearer {user_token}"},
    )
    assert list_response.status_code == 403
    assert list_response.json()["errors"] == [
        {"field": "non_field", "message": "You do not have permission to access this resource."}
    ]

    update_response = client.put(
        "/api/admin/ai-models/1",
        headers={"Authorization": f"Bearer {user_token}"},
        json={"is_active": False},
    )
    assert update_response.status_code == 403
    assert update_response.json()["errors"] == [
        {"field": "non_field", "message": "You do not have permission to access this resource."}
    ]
