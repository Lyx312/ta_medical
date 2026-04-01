from pathlib import Path
from typing import Generator

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.db.session import clear_database_caches
from app.main import create_app


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient, None, None]:
    database_path = tmp_path / "auth-test.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{database_path.as_posix()}")
    monkeypatch.setenv("APP_ENV", "test")
    get_settings.cache_clear()
    clear_database_caches()

    app = create_app()
    with TestClient(app) as test_client:
        yield test_client

    clear_database_caches()
    get_settings.cache_clear()


def test_register_success(client: TestClient) -> None:
    response = client.post(
        "/api/auth/register",
        json={
            "full_name": "Alice",
            "email": "alice@example.com",
            "password": "password123",
            "confirm_password": "password123",
        },
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["user"]["email"] == "alice@example.com"
    assert payload["user"]["role"] == "user"


def test_register_duplicate_email(client: TestClient) -> None:
    first = client.post(
        "/api/auth/register",
        json={
            "full_name": "Bob",
            "email": "bob@example.com",
            "password": "password123",
            "confirm_password": "password123",
        },
    )
    assert first.status_code == 201

    second = client.post(
        "/api/auth/register",
        json={
            "full_name": "Bobby",
            "email": "bob@example.com",
            "password": "password123",
            "confirm_password": "password123",
        },
    )

    assert second.status_code == 409
    payload = second.json()
    assert payload["message"] == "Input validation failed."
    assert payload["errors"] == [{"field": "email", "message": "Email is already registered."}]


def test_login_success(client: TestClient) -> None:
    register = client.post(
        "/api/auth/register",
        json={
            "full_name": "Carol",
            "email": "carol@example.com",
            "password": "password123",
            "confirm_password": "password123",
        },
    )
    assert register.status_code == 201

    login = client.post(
        "/api/auth/login",
        json={"email": "carol@example.com", "password": "password123"},
    )

    assert login.status_code == 200
    assert login.json()["user"]["email"] == "carol@example.com"


def test_login_wrong_password(client: TestClient) -> None:
    register = client.post(
        "/api/auth/register",
        json={
            "full_name": "David",
            "email": "david@example.com",
            "password": "password123",
            "confirm_password": "password123",
        },
    )
    assert register.status_code == 201

    login = client.post(
        "/api/auth/login",
        json={"email": "david@example.com", "password": "badpass123"},
    )

    assert login.status_code == 401
    payload = login.json()
    assert payload["message"] == "Input validation failed."
    assert payload["errors"] == [{"field": "credentials", "message": "Wrong email or password."}]


def test_http_exception_with_plain_string_detail_uses_global_error_envelope(client: TestClient) -> None:
    response = client.get("/api/does-not-exist")

    assert response.status_code == 404
    payload = response.json()
    assert payload["message"] == "Input validation failed."
    assert payload["errors"] == [{"field": "non_field", "message": "Not Found"}]


def test_register_invalid_email_returns_custom_validation_message(client: TestClient) -> None:
    response = client.post(
        "/api/auth/register",
        json={
            "full_name": "Eve",
            "email": "invalid-email",
            "password": "password123",
            "confirm_password": "password123",
        },
    )

    assert response.status_code == 422
    payload = response.json()
    assert payload["message"] == "Input validation failed."
    assert {"field": "email", "message": "Please enter a valid email address."} in payload["errors"]


def test_register_password_mismatch_returns_custom_validation_message(client: TestClient) -> None:
    response = client.post(
        "/api/auth/register",
        json={
            "full_name": "Frank",
            "email": "frank@example.com",
            "password": "password123",
            "confirm_password": "password999",
        },
    )

    assert response.status_code == 422
    payload = response.json()
    assert payload["message"] == "Input validation failed."
    assert any(
        error["message"] == "Password and confirm password do not match."
        for error in payload["errors"]
    )


def test_get_me_returns_persisted_user(client: TestClient) -> None:
    register = client.post(
        "/api/auth/register",
        json={
            "full_name": "Grace",
            "email": "grace@example.com",
            "password": "password123",
            "confirm_password": "password123",
        },
    )
    assert register.status_code == 201

    login = client.post(
        "/api/auth/login",
        json={"email": "grace@example.com", "password": "password123"},
    )
    assert login.status_code == 200

    token = login.json()["access_token"]
    response = client.get(
        "/api/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["email"] == "grace@example.com"
    assert payload["role"] == "user"


def test_startup_seed_does_not_create_duplicate_admin_rows(client: TestClient) -> None:
    first_login = client.post(
        "/api/auth/login",
        json={"email": "admin@mail.com", "password": "admin123"},
    )
    assert first_login.status_code == 200

    app = create_app()
    with TestClient(app) as second_client:
        second_login = second_client.post(
            "/api/auth/login",
            json={"email": "admin@mail.com", "password": "admin123"},
        )

    assert second_login.status_code == 200
