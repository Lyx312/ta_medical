"""
Proposal-aligned black-box API tests using equivalence partitioning.

These tests cover the backend-visible partitions from the TA proposal:
- valid multipart medical image upload creates a scan job;
- invalid bytes are rejected as unreadable image input;
- low-quality/unreadable document validation fails;
- non-medical document validation fails;
- successful scan output can be submitted as a persisted claim linked to the scan log.

Camera/gallery source, frontend-visible latency, and Smart Auto-fill UI assertions are
frontend/system concerns. The backend only sees multipart uploads.
"""

from __future__ import annotations

import hashlib
import os
from io import BytesIO
from pathlib import Path
from typing import Any, Generator
from uuid import uuid4

import pytest
from fastapi import HTTPException, status
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import select
from sqlalchemy.engine import make_url

from app.api.routes import ai as ai_route
from app.core.config import get_settings
from app.db.session import clear_database_caches, get_engine, get_session_factory
from app.main import create_app
from app.models.claim import Claim
from app.repositories.scan_log_repository import ScanLogRepository
from app.schemas.ai import AIMedicalScanResponse, DocumentValidationResponse
from app.services.ai_service import DocumentImageInput
from app.services.image_preprocessing import ProcessedDocumentImage


SCAN_RESULT = {
    "summary": {
        "detected_documents": ["HOSPITAL_INVOICE"],
        "medical_facility_name": "RS Sehat Sentosa",
        "service_date": "2024-08-10",
        "transaction_date": "2024-08-10",
        "currency": "IDR",
        "total_amount": 125000,
    },
    "patient_info": {
        "name": "Claim User",
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
    database_path = tmp_path / "proposal-black-box.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{database_path.as_posix()}")
    monkeypatch.setenv("APP_ENV", "test")
    get_settings.cache_clear()
    clear_database_caches()

    app = create_app()
    with TestClient(app) as test_client:
        yield test_client

    clear_database_caches()
    get_settings.cache_clear()


def build_png_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (16, 16), color=(255, 255, 255)).save(buffer, format="PNG")
    return buffer.getvalue()


def register_and_login(client: TestClient, email: str, *, full_name: str = "Claim User") -> str:
    register_response = client.post(
        "/api/auth/register",
        json={
            "full_name": full_name,
            "email": email,
            "password": "password123",
            "confirm_password": "password123",
        },
    )
    assert register_response.status_code == 201

    login_response = client.post(
        "/api/auth/login",
        json={"email": email, "password": "password123"},
    )
    assert login_response.status_code == 200
    return str(login_response.json()["access_token"])


def get_scan_log(scan_log_id: int) -> Any:
    session_factory = get_session_factory()
    with session_factory() as db:
        scan_log = ScanLogRepository(db).get_by_id(scan_log_id)
        assert scan_log is not None
        return scan_log


def get_claim(claim_id: int) -> Claim:
    session_factory = get_session_factory()
    with session_factory() as db:
        claim = db.execute(select(Claim).where(Claim.id == claim_id)).scalar_one_or_none()
        assert claim is not None
        return claim


def build_valid_validation_response(filename: str, *, patient_name: str = "Claim User") -> DocumentValidationResponse:
    return DocumentValidationResponse.model_validate(
        {
            "packet_patient_name": patient_name,
            "patient_name_match_status": "matched",
            "patient_name_match_reason": None,
            "review_required": False,
            "review_reasons": [],
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
            ],
        }
    )


def install_successful_scan_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_preprocess_document_image(filename: str, content: bytes) -> ProcessedDocumentImage:
        return ProcessedDocumentImage(
            filename=filename.rsplit(".", 1)[0] + ".jpg",
            mime_type="image/jpeg",
            content=b"processed-image",
        )

    async def fake_validation_process(
        images: list[DocumentImageInput],
        **_: object,
    ) -> DocumentValidationResponse:
        assert images
        return build_valid_validation_response(images[0].filename)

    async def fake_scan_process(
        images: list[DocumentImageInput],
        **_: object,
    ) -> AIMedicalScanResponse:
        assert images
        return AIMedicalScanResponse.model_validate(SCAN_RESULT)

    async def fake_prioritize_scan_medicines(
        _db: Any,
        scan_result: AIMedicalScanResponse,
        extraction_meta_payload: dict[str, Any] | None,
        **_: object,
    ) -> tuple[AIMedicalScanResponse, dict[str, Any]]:
        del extraction_meta_payload
        return scan_result, {
            "enabled": True,
            "applied": False,
            "reason": "test fixture has no medicines",
        }

    monkeypatch.setattr(ai_route, "preprocess_document_image", fake_preprocess_document_image)
    monkeypatch.setattr(ai_route.validation_processor, "process", fake_validation_process)
    monkeypatch.setattr(ai_route.processor, "process", fake_scan_process)
    monkeypatch.setattr(ai_route, "prioritize_scan_medicines", fake_prioritize_scan_medicines)


def build_document_validation_failure(message: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        detail={
            "message": "Document validation failed.",
            "errors": [
                {
                    "field": "non_field",
                    "message": message,
                }
            ],
        },
    )


def install_validation_failure_stubs(monkeypatch: pytest.MonkeyPatch, *, validation_message: str) -> None:
    def fake_preprocess_document_image(filename: str, content: bytes) -> ProcessedDocumentImage:
        return ProcessedDocumentImage(
            filename=filename.rsplit(".", 1)[0] + ".jpg",
            mime_type="image/jpeg",
            content=b"processed-image",
        )

    async def fake_validation_process(
        images: list[DocumentImageInput],
        **_: object,
    ) -> DocumentValidationResponse:
        assert images
        raise build_document_validation_failure(validation_message)

    async def extraction_should_not_run(images: list[DocumentImageInput], **_: object) -> AIMedicalScanResponse:
        raise AssertionError("Extraction should not run after document validation failure.")

    monkeypatch.setattr(ai_route, "preprocess_document_image", fake_preprocess_document_image)
    monkeypatch.setattr(ai_route.validation_processor, "process", fake_validation_process)
    monkeypatch.setattr(ai_route.processor, "process", extraction_should_not_run)


def test_valid_multipart_scan_job_creates_queued_scan_log(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = register_and_login(client, "proposal-valid-job@example.com")
    image_bytes = build_png_bytes()

    async def no_op_background_job(scan_log_id: int, user_id: int) -> None:
        del scan_log_id, user_id

    monkeypatch.setattr(ai_route, "run_scan_documents_job", no_op_background_job)

    response = client.post(
        "/api/ai/scan-documents/jobs",
        headers={"Authorization": f"Bearer {token}"},
        files=[("files", ("gallery-valid.png", image_bytes, "image/png"))],
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["progress"]["stage"] == "queued"
    assert payload["progress"]["status"] == "pending"
    assert payload["progress"]["image_count"] == 1

    scan_log = get_scan_log(payload["scan_log_id"])
    assert scan_log.status == "pending"
    assert len(scan_log.files) == 1
    assert scan_log.files[0].original_filename == "gallery-valid.png"
    assert scan_log.files[0].original_sha256_hash == hashlib.sha256(image_bytes).hexdigest()
    assert scan_log.files[0].processed_content is None


def test_invalid_image_bytes_are_rejected_and_scan_log_records_preprocess_failure(
    client: TestClient,
) -> None:
    token = register_and_login(client, "proposal-invalid-bytes@example.com")

    response = client.post(
        "/api/ai/scan-documents",
        headers={"Authorization": f"Bearer {token}"},
        files=[("files", ("broken.png", b"not-an-image", "image/png"))],
    )

    assert response.status_code == 422
    payload = response.json()
    assert payload["message"] == "Input validation failed."
    assert payload["errors"] == [
        {"field": "non_field", "message": "Uploaded file is not a readable image."}
    ]

    scan_log = get_scan_log(1)
    assert scan_log.status == "preprocess_failed"
    assert scan_log.files[0].original_content == b"not-an-image"
    assert scan_log.files[0].processed_content is None


def test_low_quality_document_partition_returns_validation_failure(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = register_and_login(client, "proposal-low-quality@example.com")
    install_validation_failure_stubs(
        monkeypatch,
        validation_message="Image 1 document 1: the document is unreadable, blurred, or cropped.",
    )

    response = client.post(
        "/api/ai/scan-documents",
        headers={"Authorization": f"Bearer {token}"},
        files=[("files", ("blurred-receipt.png", b"fake-image", "image/png"))],
    )

    assert response.status_code == 422
    payload = response.json()
    assert payload["message"] == "Document validation failed."
    assert "unreadable, blurred, or cropped" in payload["errors"][0]["message"]

    scan_log = get_scan_log(1)
    assert scan_log.status == "validation_failed"
    assert scan_log.extraction_payload is None


def test_non_medical_document_partition_returns_validation_failure(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = register_and_login(client, "proposal-non-medical@example.com")
    install_validation_failure_stubs(
        monkeypatch,
        validation_message="Image 1 document 1: unsupported or non-medical document type.",
    )

    response = client.post(
        "/api/ai/scan-documents",
        headers={"Authorization": f"Bearer {token}"},
        files=[("files", ("restaurant-receipt.png", b"fake-image", "image/png"))],
    )

    assert response.status_code == 422
    payload = response.json()
    assert payload["message"] == "Document validation failed."
    assert "non-medical document type" in payload["errors"][0]["message"]

    scan_log = get_scan_log(1)
    assert scan_log.status == "validation_failed"
    assert scan_log.extraction_payload is None


def test_scan_then_claim_submit_persists_claim_and_scan_relation(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = register_and_login(client, "proposal-linked-claim@example.com")
    install_successful_scan_stubs(monkeypatch)

    original_bytes = b"stable-claim-image"
    scan_response = client.post(
        "/api/ai/scan-documents",
        headers={"Authorization": f"Bearer {token}"},
        files=[("files", ("receipt.png", original_bytes, "image/png"))],
    )

    assert scan_response.status_code == 200
    scan_payload = scan_response.json()
    scan_log_id = scan_payload["scan_log_id"]

    claim_response = client.post(
        "/api/claims",
        headers={"Authorization": f"Bearer {token}"},
        data={
            "nama_pasien": "Claim User",
            "tanggal_pelayanan": "10-08-2024",
            "fasilitas_kesehatan": "RS Sehat Sentosa",
            "jenis_manfaat": "rawat_jalan",
            "nominal_klaim": "125000",
            "keterangan_penyakit": "Influenza",
            "kronologi": "Demam dua hari lalu menjalani pemeriksaan rawat jalan.",
            "source_scan_log_id": str(scan_log_id),
        },
        files=[("files", ("receipt.png", original_bytes, "image/png"))],
    )

    assert claim_response.status_code == 201
    claim_payload = claim_response.json()
    assert claim_payload["source_scan_log_id"] == scan_log_id
    assert claim_payload["file_count"] == 1
    assert claim_payload["files"][0]["file_data_url"].startswith("data:image/png;base64,")
    assert claim_payload["source_scan_extraction_payload"]["patient_info"]["name"] == "Claim User"

    claim = get_claim(claim_payload["id"])
    assert claim.nama_pasien == "Claim User"
    assert claim.source_scan_log_id == scan_log_id


@pytest.mark.postgres
def test_postgresql_database_url_persists_scan_log_and_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_url = os.getenv("TEST_POSTGRES_DATABASE_URL")
    if not database_url:
        pytest.skip("TEST_POSTGRES_DATABASE_URL is not configured.")

    parsed_url = make_url(database_url)
    database_name = parsed_url.database or ""
    if "test" not in database_name.lower():
        pytest.skip("Refusing to run PostgreSQL integration test outside a test database.")

    monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.setenv("APP_ENV", "test")
    get_settings.cache_clear()
    clear_database_caches()
    install_successful_scan_stubs(monkeypatch)

    app = create_app()
    with TestClient(app) as postgres_client:
        token = register_and_login(
            postgres_client,
            f"proposal-postgres-{uuid4().hex}@example.com",
        )
        original_bytes = f"postgres-image-{uuid4().hex}".encode("utf-8")

        scan_response = postgres_client.post(
            "/api/ai/scan-documents",
            headers={"Authorization": f"Bearer {token}"},
            files=[("files", ("receipt.png", original_bytes, "image/png"))],
        )
        assert scan_response.status_code == 200
        scan_log_id = scan_response.json()["scan_log_id"]

        claim_response = postgres_client.post(
            "/api/claims",
            headers={"Authorization": f"Bearer {token}"},
            data={
                "nama_pasien": "Claim User",
                "tanggal_pelayanan": "10-08-2024",
                "fasilitas_kesehatan": "RS Sehat Sentosa",
                "jenis_manfaat": "rawat_jalan",
                "nominal_klaim": "125000",
                "keterangan_penyakit": "Influenza",
                "kronologi": "Demam dua hari lalu menjalani pemeriksaan rawat jalan.",
                "source_scan_log_id": str(scan_log_id),
            },
            files=[("files", ("receipt.png", original_bytes, "image/png"))],
        )
        assert claim_response.status_code == 201
        claim_id = claim_response.json()["id"]

        assert get_engine().url.drivername.startswith("postgresql")
        claim = get_claim(claim_id)
        assert claim.source_scan_log_id == scan_log_id
        assert claim.nama_pasien == "Claim User"

    clear_database_caches()
    get_settings.cache_clear()
