import mimetypes
from pathlib import Path

from fastapi.testclient import TestClient


def guess_mime_type(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(path))
    return mime_type or "application/octet-stream"


def register_test_user(client: TestClient, email: str, full_name: str) -> str:
    register = client.post(
        "/api/auth/register",
        json={
            "full_name": full_name,
            "email": email,
            "password": "password123",
            "confirm_password": "password123",
        },
    )
    if register.status_code == 201:
        pass
    elif register.status_code in {400, 409} and "already" in register.text.lower():
        pass
    else:
        raise RuntimeError(f"Register failed: {register.status_code} {register.text}")

    login = client.post(
        "/api/auth/login",
        json={"email": email, "password": "password123"},
    )
    if login.status_code != 200:
        raise RuntimeError(f"Login failed: {login.status_code} {login.text}")
    return str(login.json()["access_token"])


def ensure_family_candidates(
    client: TestClient,
    token: str,
    candidates: list[str],
    primary_name: str,
) -> None:
    existing_response = client.get(
        "/api/family-members",
        headers={"Authorization": f"Bearer {token}"},
    )
    if existing_response.status_code != 200:
        raise RuntimeError(
            f"Failed to list family members: {existing_response.status_code} {existing_response.text}"
        )

    existing_names = {
        str(member.get("name", "")).strip().casefold()
        for member in existing_response.json()
        if isinstance(member, dict)
    }

    for candidate in candidates:
        normalized = candidate.strip()
        if (
            not normalized
            or normalized.casefold() == primary_name.strip().casefold()
            or normalized.casefold() in existing_names
        ):
            continue

        response = client.post(
            "/api/family-members",
            headers={"Authorization": f"Bearer {token}"},
            json={"name": normalized, "relation": "Family"},
        )
        if response.status_code != 201:
            raise RuntimeError(
                f"Failed to create family member '{normalized}': {response.status_code} {response.text}"
            )
        existing_names.add(normalized.casefold())
