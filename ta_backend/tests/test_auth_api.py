from fastapi.testclient import TestClient

from app.api.routes.auth import user_repository
from app.main import create_app


app = create_app()


def setup_function() -> None:
    user_repository.clear()


def test_register_success() -> None:
    client = TestClient(app)

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


def test_register_duplicate_email() -> None:
    client = TestClient(app)

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


def test_login_success() -> None:
    client = TestClient(app)

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


def test_login_wrong_password() -> None:
    client = TestClient(app)

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


def test_http_exception_with_plain_string_detail_uses_global_error_envelope() -> None:
    client = TestClient(app)

    response = client.get("/api/does-not-exist")

    assert response.status_code == 404
    payload = response.json()
    assert payload["message"] == "Input validation failed."
    assert payload["errors"] == [{"field": "non_field", "message": "Not Found"}]


def test_register_invalid_email_returns_custom_validation_message() -> None:
    client = TestClient(app)

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


def test_register_password_mismatch_returns_custom_validation_message() -> None:
    client = TestClient(app)

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
