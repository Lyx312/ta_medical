from pathlib import Path
from typing import Generator

import pytest
from fastapi.testclient import TestClient

from app.api.routes import ai as ai_route
from app.api.routes import claims as claims_route
from app.api.routes.ai import processor as ai_processor
from app.core.config import get_settings
from app.db.session import clear_database_caches
from app.main import create_app
from app.schemas.ai import AIMedicalScanResponse, DocumentValidationResponse
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


def test_admin_claim_filters_and_summary(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_token = register_and_login(client, "admin-claims-filter@example.com", full_name="Siti Admin Filter")
    admin_token = login_admin(client)

    def fake_preprocess_document_image(filename: str, content: bytes) -> ProcessedDocumentImage:
        return ProcessedDocumentImage(
            filename=filename.replace(".png", ".jpg"),
            mime_type="image/jpeg",
            content=b"processed-image",
        )

    async def fake_validation_process(images: list[DocumentImageInput]) -> DocumentValidationResponse:
        return build_valid_validation_response(images[0].filename)

    async def fake_scan_process(
        images: list[DocumentImageInput],
        patient_name_candidates: list[str] | None = None,
    ) -> AIMedicalScanResponse:
        assert patient_name_candidates == ["Siti Admin Filter"]
        return AIMedicalScanResponse.model_validate(
            {
                "summary": {
                    "detected_documents": ["HOSPITAL_INVOICE"],
                    "medical_facility_name": "RS Sehat Sentosa",
                    "service_date": "2026-03-31",
                    "currency": "IDR",
                    "total_amount": 200000,
                },
                "patient_info": {
                    "name": "Siti Admin Filter",
                    "date_of_birth": "1990-01-01",
                    "age": 36,
                    "gender": "FEMALE",
                    "address": "Jl. Testing No.1",
                },
                "medical_context": {
                    "anamnesis": "Demam ringan",
                    "diagnoses": [{"description": "Influenza", "doctor_name": "dr. Maya"}],
                    "therapy_note": "Istirahat",
                },
                "medicines": [],
                "other_charges": [],
                "financial_recap": {
                    "total_medicines": 0,
                    "total_other_charges": 200000,
                    "grand_total": 200000,
                },
            }
        )

    monkeypatch.setattr(claims_route, "preprocess_document_image", fake_preprocess_document_image)
    monkeypatch.setattr(claims_route.validation_processor, "process", fake_validation_process)
    monkeypatch.setattr(ai_route, "preprocess_document_image", fake_preprocess_document_image)
    monkeypatch.setattr(ai_route.validation_processor, "process", fake_validation_process)
    monkeypatch.setattr(ai_processor, "process", fake_scan_process)

    scan_response = client.post(
        "/api/ai/scan-documents",
        headers={"Authorization": f"Bearer {user_token}"},
        files=[("files", ("linked-claim.png", b"linked-image", "image/png"))],
    )
    assert scan_response.status_code == 200
    scan_log_id = scan_response.json()["scan_log_id"]

    linked_claim_response = client.post(
        "/api/claims",
        headers={"Authorization": f"Bearer {user_token}"},
        data={
            "nama_pasien": "Siti Admin Filter",
            "tanggal_pelayanan": "31-03-2026",
            "fasilitas_kesehatan": "RS Sehat Sentosa",
            "jenis_manfaat": "rawat_jalan",
            "nominal_klaim": "200000",
            "keterangan_penyakit": "Influenza",
            "kronologi": "Rawat jalan dengan scan AI",
            "source_scan_log_id": str(scan_log_id),
        },
        files=[("files", ("linked-claim.png", b"linked-image", "image/png"))],
    )
    assert linked_claim_response.status_code == 201

    manual_claim_response = client.post(
        "/api/claims",
        headers={"Authorization": f"Bearer {user_token}"},
        data={
            "nama_pasien": "Budi Manual",
            "tanggal_pelayanan": "01-04-2026",
            "fasilitas_kesehatan": "Klinik Manual",
            "jenis_manfaat": "emergency",
            "nominal_klaim": "350000",
            "keterangan_penyakit": "Cedera ringan",
            "kronologi": "Datang ke IGD",
        },
        files=[("files", ("manual-claim.png", b"manual-image", "image/png"))],
    )
    assert manual_claim_response.status_code == 201

    linked_list_response = client.get(
        "/api/admin/claims?linked=true",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert linked_list_response.status_code == 200
    linked_payload = linked_list_response.json()
    assert len(linked_payload) == 1
    assert linked_payload[0]["source_scan_log_id"] == scan_log_id

    manual_search_response = client.get(
        "/api/admin/claims?linked=false&q=Budi",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert manual_search_response.status_code == 200
    manual_payload = manual_search_response.json()
    assert len(manual_payload) == 1
    assert manual_payload[0]["nama_pasien"] == "Budi Manual"
    assert manual_payload[0]["source_scan_log_id"] is None

    summary_response = client.get(
        "/api/admin/claims/summary",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert summary_response.status_code == 200
    assert summary_response.json() == {
        "total_claims": 2,
        "total_claim_amount": 550000.0,
        "linked_scan_count": 1,
        "manual_submission_count": 1,
    }
