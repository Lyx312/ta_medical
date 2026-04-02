from pathlib import Path
from typing import Generator

import pytest
from fastapi.testclient import TestClient

from app.api.routes import claims as claims_route
from app.core.config import get_settings
from app.db.session import clear_database_caches
from app.main import create_app
from app.schemas.ai import DocumentValidationResponse
from app.services.ai_service import DocumentImageInput
from app.services.image_preprocessing import ProcessedDocumentImage


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient, None, None]:
    database_path = tmp_path / "admin-claims-test.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{database_path.as_posix()}")
    monkeypatch.setenv("APP_ENV", "test")
    get_settings.cache_clear()
    clear_database_caches()

    app = create_app()
    with TestClient(app) as test_client:
        yield test_client

    clear_database_caches()
    get_settings.cache_clear()


def register_and_login(client: TestClient, email: str, *, full_name: str = "Admin Claim User") -> str:
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


def login_admin(client: TestClient) -> str:
    response = client.post(
        "/api/auth/login",
        json={"email": "admin@mail.com", "password": "admin123"},
    )
    assert response.status_code == 200
    return response.json()["access_token"]


def build_valid_validation_response(filename: str) -> DocumentValidationResponse:
    return DocumentValidationResponse.model_validate(
        {
            "images": [
                {
                    "image_index": 0,
                    "filename": filename,
                    "document_count_detected": 1,
                    "documents": [
                        {
                            "document_index": 0,
                            "document_type": "receipt",
                            "is_medical_document": True,
                            "readability_ok": True,
                            "facility_identity_visible": True,
                            "stamp_detected": True,
                            "signature_detected": False,
                            "decision": "valid",
                            "reasons": ["Visible facility stamp detected."],
                        }
                    ],
                }
            ]
        }
    )


def test_admin_can_list_and_view_claims_with_documents(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_token = register_and_login(client, "admin-claims-user@example.com")
    admin_token = login_admin(client)

    def fake_preprocess_document_image(filename: str, content: bytes) -> ProcessedDocumentImage:
        assert filename == "receipt.png"
        assert content == b"fake-image"
        return ProcessedDocumentImage(
            filename="receipt.jpg",
            mime_type="image/jpeg",
            content=b"processed-image",
        )

    async def fake_validation_process(images: list[DocumentImageInput]) -> DocumentValidationResponse:
        return build_valid_validation_response(images[0].filename)

    monkeypatch.setattr(claims_route, "preprocess_document_image", fake_preprocess_document_image)
    monkeypatch.setattr(claims_route.validation_processor, "process", fake_validation_process)

    create_response = client.post(
        "/api/claims",
        headers={"Authorization": f"Bearer {user_token}"},
        data={
            "nama_pasien": "Budi Santoso",
            "tanggal_pelayanan": "31-03-2026",
            "fasilitas_kesehatan": "RS Sehat Selalu",
            "jenis_manfaat": "rawat_jalan",
            "nominal_klaim": "150000",
            "keterangan_penyakit": "Demam",
            "kronologi": "Pemeriksaan rawat jalan",
        },
        files=[("files", ("receipt.png", b"fake-image", "image/png"))],
    )

    assert create_response.status_code == 201
    claim_id = create_response.json()["id"]

    list_response = client.get(
        "/api/admin/claims",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert list_response.status_code == 200
    list_payload = list_response.json()
    assert len(list_payload) == 1
    assert list_payload[0]["id"] == claim_id
    assert list_payload[0]["file_count"] == 1

    detail_response = client.get(
        f"/api/admin/claims/{claim_id}",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert detail_response.status_code == 200
    detail_payload = detail_response.json()
    assert detail_payload["id"] == claim_id
    assert detail_payload["files"][0]["filename"] == "receipt.png"
    assert detail_payload["files"][0]["mime_type"] == "image/png"
    assert detail_payload["files"][0]["file_data_url"].startswith("data:image/png;base64,")


def test_non_admin_cannot_access_admin_claim_routes(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_token = register_and_login(client, "admin-claims-non-admin@example.com")

    def fake_preprocess_document_image(filename: str, content: bytes) -> ProcessedDocumentImage:
        return ProcessedDocumentImage(
            filename="receipt.jpg",
            mime_type="image/jpeg",
            content=b"processed-image",
        )

    async def fake_validation_process(images: list[DocumentImageInput]) -> DocumentValidationResponse:
        return build_valid_validation_response(images[0].filename)

    monkeypatch.setattr(claims_route, "preprocess_document_image", fake_preprocess_document_image)
    monkeypatch.setattr(claims_route.validation_processor, "process", fake_validation_process)

    create_response = client.post(
        "/api/claims",
        headers={"Authorization": f"Bearer {user_token}"},
        data={
            "nama_pasien": "Budi Santoso",
            "tanggal_pelayanan": "31-03-2026",
            "fasilitas_kesehatan": "RS Sehat Selalu",
            "jenis_manfaat": "rawat_jalan",
            "nominal_klaim": "150000",
            "keterangan_penyakit": "Demam",
            "kronologi": "Pemeriksaan rawat jalan",
        },
        files=[("files", ("receipt.png", b"fake-image", "image/png"))],
    )
    assert create_response.status_code == 201
    claim_id = create_response.json()["id"]

    list_response = client.get(
        "/api/admin/claims",
        headers={"Authorization": f"Bearer {user_token}"},
    )
    assert list_response.status_code == 403
    assert list_response.json()["errors"] == [
        {"field": "non_field", "message": "You do not have permission to access this resource."}
    ]

    detail_response = client.get(
        f"/api/admin/claims/{claim_id}",
        headers={"Authorization": f"Bearer {user_token}"},
    )
    assert detail_response.status_code == 403
    assert detail_response.json()["errors"] == [
        {"field": "non_field", "message": "You do not have permission to access this resource."}
    ]
