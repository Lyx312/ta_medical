from pathlib import Path
from typing import Generator

import pytest
from fastapi.testclient import TestClient

import app.models.claim as claim_model
from app.api.routes.claims import claim_repository
from app.core.config import get_settings
from app.db.session import clear_database_caches
from app.main import create_app


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient, None, None]:
    database_path = tmp_path / "claims-test.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{database_path.as_posix()}")
    monkeypatch.setenv("APP_ENV", "test")
    get_settings.cache_clear()
    clear_database_caches()
    claim_repository.clear()
    claim_model._counter = 0

    app = create_app()
    with TestClient(app) as test_client:
        yield test_client

    claim_repository.clear()
    clear_database_caches()
    get_settings.cache_clear()


def register_and_login(client: TestClient, email: str) -> str:
    register = client.post(
        "/api/auth/register",
        json={
            "full_name": "Claim User",
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


def test_create_and_list_claims_include_nama_pasien(client: TestClient) -> None:
    token = register_and_login(client, "claims@example.com")

    create_response = client.post(
        "/api/claims",
        headers={"Authorization": f"Bearer {token}"},
        data={
            "nama_pasien": "Budi Santoso",
            "tanggal_pelayanan": "31-03-2026",
            "fasilitas_kesehatan": "RS Sehat Selalu",
            "jenis_manfaat": "rawat_jalan",
            "nominal_klaim": "150000",
            "keterangan_penyakit": "Demam",
            "kronologi": "Pemeriksaan rawat jalan",
        },
    )

    assert create_response.status_code == 201
    created_payload = create_response.json()
    assert created_payload["nama_pasien"] == "Budi Santoso"
    assert created_payload["tanggal_pelayanan"] == "31-03-2026"

    list_response = client.get(
        "/api/claims",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert list_response.status_code == 200
    claims = list_response.json()
    assert len(claims) == 1
    assert claims[0]["nama_pasien"] == "Budi Santoso"
    assert claims[0]["tanggal_pelayanan"] == "31-03-2026"


def test_create_claim_rejects_non_dmy_date_format(client: TestClient) -> None:
    token = register_and_login(client, "claims-invalid-date@example.com")

    response = client.post(
        "/api/claims",
        headers={"Authorization": f"Bearer {token}"},
        data={
            "nama_pasien": "Budi Santoso",
            "tanggal_pelayanan": "2026-03-31",
            "fasilitas_kesehatan": "RS Sehat Selalu",
            "jenis_manfaat": "rawat_jalan",
            "nominal_klaim": "150000",
            "keterangan_penyakit": "Demam",
            "kronologi": "Pemeriksaan rawat jalan",
        },
    )

    assert response.status_code == 422
    payload = response.json()
    assert payload["message"] == "Input validation failed."
    assert payload["errors"] == [
        {"field": "non_field", "message": "Invalid tanggal_pelayanan. Expected format: dd-mm-yyyy"}
    ]
