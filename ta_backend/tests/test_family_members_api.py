from pathlib import Path
from typing import Generator

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.db.session import clear_database_caches
from app.main import create_app


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient, None, None]:
    database_path = tmp_path / "family-members-test.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{database_path.as_posix()}")
    monkeypatch.setenv("APP_ENV", "test")
    get_settings.cache_clear()
    clear_database_caches()

    app = create_app()
    with TestClient(app) as test_client:
        yield test_client

    clear_database_caches()
    get_settings.cache_clear()


def register_and_login(client: TestClient, *, email: str, full_name: str) -> str:
    register = client.post(
        "/api/auth/register",
        json={
            "full_name": full_name,
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


def test_family_member_crud_flow(client: TestClient) -> None:
    token = register_and_login(client, email="family@example.com", full_name="Family User")
    headers = {"Authorization": f"Bearer {token}"}

    create_response = client.post(
        "/api/family-members",
        headers=headers,
        json={"name": "Budi Santoso", "relation": "Child"},
    )

    assert create_response.status_code == 201
    created_family_member = create_response.json()
    assert created_family_member["name"] == "Budi Santoso"
    assert created_family_member["relation"] == "Child"

    list_response = client.get("/api/family-members", headers=headers)
    assert list_response.status_code == 200
    listed_family_members = list_response.json()
    assert len(listed_family_members) == 1
    assert listed_family_members[0]["name"] == "Budi Santoso"

    update_response = client.put(
        f"/api/family-members/{created_family_member['id']}",
        headers=headers,
        json={"name": "Budi Pratama", "relation": "Son"},
    )
    assert update_response.status_code == 200
    assert update_response.json()["name"] == "Budi Pratama"
    assert update_response.json()["relation"] == "Son"

    delete_response = client.delete(
        f"/api/family-members/{created_family_member['id']}",
        headers=headers,
    )
    assert delete_response.status_code == 204

    final_list_response = client.get("/api/family-members", headers=headers)
    assert final_list_response.status_code == 200
    assert final_list_response.json() == []


def test_family_member_validation_rejects_empty_fields(client: TestClient) -> None:
    token = register_and_login(client, email="invalid-family@example.com", full_name="Family User")

    response = client.post(
        "/api/family-members",
        headers={"Authorization": f"Bearer {token}"},
        json={"name": "   ", "relation": "   "},
    )

    assert response.status_code == 422
    payload = response.json()
    assert payload["message"] == "Input validation failed."
    assert {"field": "name", "message": "Field cannot be empty."} in payload["errors"]
    assert {"field": "relation", "message": "Field cannot be empty."} in payload["errors"]


def test_family_member_routes_reject_other_users_access(client: TestClient) -> None:
    owner_token = register_and_login(client, email="owner@example.com", full_name="Owner User")
    other_token = register_and_login(client, email="other@example.com", full_name="Other User")

    create_response = client.post(
        "/api/family-members",
        headers={"Authorization": f"Bearer {owner_token}"},
        json={"name": "Nina Owner", "relation": "Spouse"},
    )
    assert create_response.status_code == 201
    family_member_id = create_response.json()["id"]

    update_response = client.put(
        f"/api/family-members/{family_member_id}",
        headers={"Authorization": f"Bearer {other_token}"},
        json={"name": "Hijacked Name", "relation": "Other"},
    )
    assert update_response.status_code == 404

    delete_response = client.delete(
        f"/api/family-members/{family_member_id}",
        headers={"Authorization": f"Bearer {other_token}"},
    )
    assert delete_response.status_code == 404
