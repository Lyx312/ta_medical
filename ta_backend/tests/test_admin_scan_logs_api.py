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

SCAN_RESULT = {
    "summary": {
        "detected_documents": ["HOSPITAL_INVOICE"],
        "medical_facility_name": "RS Sehat Sentosa",
        "service_date": "2024-08-10",
        "currency": "IDR",
        "total_amount": 125000,
    },
    "patient_info": {
        "name": "Admin Log User",
        "date_of_birth": "1985-05-15",
        "age": 39,
        "gender": "MALE",
        "address": "JL. Pejuang No.1",
    },
    "medical_context": {
        "anamnesis": "Demam dua hari",
        "diagnoses": [{"description": "Influenza", "doctor_name": "dr. Budi"}],
        "therapy_note": "Istirahat dan minum obat",
    },
    "medicines": [],
    "other_charges": [],
    "financial_recap": {
        "total_medicines": 0,
        "total_other_charges": 125000,
        "grand_total": 125000,
    },
}


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient, None, None]:
    database_path = tmp_path / "admin-scan-logs-test.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{database_path.as_posix()}")
    monkeypatch.setenv("APP_ENV", "test")
    get_settings.cache_clear()
    clear_database_caches()

    app = create_app()
    with TestClient(app) as test_client:
        yield test_client

    clear_database_caches()
    get_settings.cache_clear()


def register_and_login(client: TestClient, email: str, *, full_name: str = "Admin Log User") -> str:
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


def test_admin_can_list_and_view_scan_logs(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_token = register_and_login(client, "admin-scan-logs-user@example.com")
    admin_token = login_admin(client)

    def fake_preprocess_document_image(filename: str, content: bytes) -> ProcessedDocumentImage:
        assert filename == "scan.png"
        assert content == b"abc"
        return ProcessedDocumentImage(
            filename="scan.jpg",
            mime_type="image/jpeg",
            content=b"processed-image",
        )

    async def fake_validation_process(images: list[DocumentImageInput]) -> DocumentValidationResponse:
        return build_valid_validation_response(images[0].filename)

    async def fake_scan_process(
        images: list[DocumentImageInput],
        patient_name_candidates: list[str] | None = None,
    ) -> AIMedicalScanResponse:
        assert patient_name_candidates == ["Admin Log User"]
        return AIMedicalScanResponse.model_validate(SCAN_RESULT)

    monkeypatch.setattr(ai_route, "preprocess_document_image", fake_preprocess_document_image)
    monkeypatch.setattr(ai_route.validation_processor, "process", fake_validation_process)
    monkeypatch.setattr(ai_processor, "process", fake_scan_process)

    scan_response = client.post(
        "/api/ai/scan-documents",
        headers={"Authorization": f"Bearer {user_token}"},
        files=[("files", ("scan.png", b"abc", "image/png"))],
    )

    assert scan_response.status_code == 200
    scan_log_id = scan_response.json()["scan_log_id"]

    list_response = client.get(
        "/api/admin/scan-logs?status=succeeded",
        headers={"Authorization": f"Bearer {admin_token}"},
    )

    assert list_response.status_code == 200
    payload = list_response.json()
    assert len(payload) == 1
    assert payload[0]["id"] == scan_log_id
    assert payload[0]["status"] == "succeeded"
    assert payload[0]["file_count"] == 1
    assert payload[0]["linked_claim_ids"] == []
    assert payload[0]["user_email"] == "admin-scan-logs-user@example.com"

    detail_response = client.get(
        f"/api/admin/scan-logs/{scan_log_id}",
        headers={"Authorization": f"Bearer {admin_token}"},
    )

    assert detail_response.status_code == 200
    detail_payload = detail_response.json()
    assert detail_payload["id"] == scan_log_id
    assert detail_payload["validation_payload"]["images"][0]["filename"] == "scan.jpg"
    assert detail_payload["extraction_payload"] == SCAN_RESULT
    assert detail_payload["files"][0]["original_image_data_url"].startswith("data:image/png;base64,")
    assert detail_payload["files"][0]["processed_image_data_url"].startswith("data:image/jpeg;base64,")


def test_non_admin_cannot_access_scan_log_admin_routes(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_token = register_and_login(client, "scan-log-non-admin@example.com")

    def fake_preprocess_document_image(filename: str, content: bytes) -> ProcessedDocumentImage:
        return ProcessedDocumentImage(
            filename="scan.jpg",
            mime_type="image/jpeg",
            content=b"processed-image",
        )

    async def fake_validation_process(images: list[DocumentImageInput]) -> DocumentValidationResponse:
        return build_valid_validation_response(images[0].filename)

    async def fake_scan_process(
        images: list[DocumentImageInput],
        patient_name_candidates: list[str] | None = None,
    ) -> AIMedicalScanResponse:
        return AIMedicalScanResponse.model_validate(SCAN_RESULT)

    monkeypatch.setattr(ai_route, "preprocess_document_image", fake_preprocess_document_image)
    monkeypatch.setattr(ai_route.validation_processor, "process", fake_validation_process)
    monkeypatch.setattr(ai_processor, "process", fake_scan_process)

    scan_response = client.post(
        "/api/ai/scan-documents",
        headers={"Authorization": f"Bearer {user_token}"},
        files=[("files", ("scan.png", b"abc", "image/png"))],
    )
    assert scan_response.status_code == 200
    scan_log_id = scan_response.json()["scan_log_id"]

    list_response = client.get(
        "/api/admin/scan-logs",
        headers={"Authorization": f"Bearer {user_token}"},
    )
    assert list_response.status_code == 403
    assert list_response.json()["errors"] == [
        {"field": "non_field", "message": "You do not have permission to access this resource."}
    ]

    detail_response = client.get(
        f"/api/admin/scan-logs/{scan_log_id}",
        headers={"Authorization": f"Bearer {user_token}"},
    )
    assert detail_response.status_code == 403
    assert detail_response.json()["errors"] == [
        {"field": "non_field", "message": "You do not have permission to access this resource."}
    ]


def test_admin_scan_log_filters_and_summary(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_token = register_and_login(client, "queue-user@example.com", full_name="Queue User")
    second_user_token = register_and_login(client, "second-user@example.com", full_name="Second Person")
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
        candidate_name = patient_name_candidates[0] if patient_name_candidates else "Unknown"
        extracted_name = "Unexpected Person" if images[0].filename == "queue-failed.jpg" else candidate_name
        return AIMedicalScanResponse.model_validate(
            {
                **SCAN_RESULT,
                "patient_info": {
                    **SCAN_RESULT["patient_info"],
                    "name": extracted_name,
                },
            }
        )

    monkeypatch.setattr(ai_route, "preprocess_document_image", fake_preprocess_document_image)
    monkeypatch.setattr(ai_route.validation_processor, "process", fake_validation_process)
    monkeypatch.setattr(ai_processor, "process", fake_scan_process)
    monkeypatch.setattr(claims_route, "preprocess_document_image", fake_preprocess_document_image)
    monkeypatch.setattr(claims_route.validation_processor, "process", fake_validation_process)

    linked_scan_response = client.post(
        "/api/ai/scan-documents",
        headers={"Authorization": f"Bearer {user_token}"},
        files=[("files", ("queue-linked.png", b"linked", "image/png"))],
    )
    assert linked_scan_response.status_code == 200
    linked_scan_log_id = linked_scan_response.json()["scan_log_id"]

    linked_claim_response = client.post(
        "/api/claims",
        headers={"Authorization": f"Bearer {user_token}"},
        data={
            "nama_pasien": "Queue User",
            "tanggal_pelayanan": "31-03-2026",
            "fasilitas_kesehatan": "RS Sehat Sentosa",
            "jenis_manfaat": "rawat_jalan",
            "nominal_klaim": "125000",
            "keterangan_penyakit": "Influenza",
            "kronologi": "Scan linked claim",
            "source_scan_log_id": str(linked_scan_log_id),
        },
        files=[("files", ("queue-linked.png", b"linked", "image/png"))],
    )
    assert linked_claim_response.status_code == 201

    failed_scan_response = client.post(
        "/api/ai/scan-documents",
        headers={"Authorization": f"Bearer {second_user_token}"},
        files=[("files", ("queue-failed.png", b"failed", "image/png"))],
    )
    assert failed_scan_response.status_code == 422

    unlinked_scan_response = client.post(
        "/api/ai/scan-documents",
        headers={"Authorization": f"Bearer {second_user_token}"},
        files=[("files", ("queue-unlinked.png", b"unlinked", "image/png"))],
    )
    assert unlinked_scan_response.status_code == 200
    unlinked_scan_log_id = unlinked_scan_response.json()["scan_log_id"]

    search_response = client.get(
        "/api/admin/scan-logs?q=Second",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert search_response.status_code == 200
    search_payload = search_response.json()
    assert len(search_payload) == 2
    assert all(item["user_full_name"] == "Second Person" for item in search_payload)

    unlinked_response = client.get(
        "/api/admin/scan-logs?linked=false",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert unlinked_response.status_code == 200
    unlinked_payload = unlinked_response.json()
    assert {item["id"] for item in unlinked_payload} == {item["id"] for item in search_payload}

    summary_response = client.get(
        "/api/admin/scan-logs/summary",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert summary_response.status_code == 200
    assert summary_response.json() == {
        "total_logs": 3,
        "succeeded_count": 2,
        "failed_count": 1,
        "unlinked_count": 2,
    }
