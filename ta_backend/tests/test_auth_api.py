from io import BytesIO
from pathlib import Path
from typing import Generator

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.core.config import get_settings
from app.db.session import clear_database_caches
from app.main import create_app


def build_png_bytes() -> bytes:
    output = BytesIO()
    Image.new("RGB", (8, 8), color="white").save(output, format="PNG")
    return output.getvalue()


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
    assert payload["access_token"]
    assert payload["token_type"] == "bearer"
    assert payload["user"]["email"] == "alice@example.com"
    assert payload["user"]["role"] == "user"

    me = client.get(
        "/api/auth/me",
        headers={"Authorization": f"Bearer {payload['access_token']}"},
    )

    assert me.status_code == 200
    assert me.json()["email"] == "alice@example.com"


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


def test_user_profile_update_and_insurance_card_completes_profile(client: TestClient) -> None:
    register = client.post(
        "/api/auth/register",
        json={
            "full_name": "Profile User",
            "email": "profile@example.com",
            "password": "password123",
            "confirm_password": "password123",
        },
    )
    assert register.status_code == 201
    token = register.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    update = client.put(
        "/api/auth/me/profile",
        headers=headers,
        json={
            "nip": "NIP-001",
            "company_name": "Acme Corp",
            "insurance_card_number": "CARD-001",
            "phone_number": "031123456",
            "mobile_number": "08123456789",
            "bank_account_number": "1234567890",
            "bank_account_holder_name": "Profile User",
            "bank_name": "Bank Test",
        },
    )

    assert update.status_code == 200
    update_payload = update.json()
    assert update_payload["nip"] == "NIP-001"
    assert update_payload["company_name"] == "Acme Corp"
    assert update_payload["profile_complete"] is False
    assert "Foto kartu Asuransi" in update_payload["missing_profile_fields"]

    upload = client.post(
        "/api/auth/me/insurance-card",
        headers=headers,
        files={"file": ("insurance.png", build_png_bytes(), "image/png")},
    )

    assert upload.status_code == 200
    upload_payload = upload.json()
    assert upload_payload["insurance_card_filename"] == "insurance.png"
    assert upload_payload["insurance_card_mime_type"] == "image/png"
    assert upload_payload["insurance_card_data_url"].startswith("data:image/png;base64,")
    assert upload_payload["profile_complete"] is True
    assert upload_payload["missing_profile_fields"] == []


def test_user_insurance_card_upload_rejects_non_image_bytes(client: TestClient) -> None:
    register = client.post(
        "/api/auth/register",
        json={
            "full_name": "Bad Image User",
            "email": "bad-image@example.com",
            "password": "password123",
            "confirm_password": "password123",
        },
    )
    assert register.status_code == 201
    token = register.json()["access_token"]

    response = client.post(
        "/api/auth/me/insurance-card",
        headers={"Authorization": f"Bearer {token}"},
        files={"file": ("bad.txt", b"not an image", "text/plain")},
    )

    assert response.status_code == 422
    payload = response.json()
    assert {"field": "insurance_card", "message": "Uploaded insurance card must be a readable image."} in payload["errors"]


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
