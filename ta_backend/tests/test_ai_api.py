import asyncio
import base64
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from langchain_core.language_models._utils import _normalize_messages
from langchain_core.messages import AIMessage
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.api.routes import ai as ai_route
from app.api.routes.ai import processor
from app.core.config import Settings, get_settings
from app.db.base import Base
from app.db.session import clear_database_caches, get_session_factory
from app.main import create_app
from app.models.ai_model import AIModel
from app.models.scan_log import AIApiCallLog
from app.repositories.scan_log_repository import ScanLogRepository
from app.schemas.ai import AIMedicalScanResponse, DocumentValidationResponse
from app.services import ai_service as ai_service_module
from app.services.ai_ensemble import MedicalExtractionFusionEngine, parse_indonesian_amount_words
from app.services.ai_service import (
    AIModelSpec,
    DocumentImageInput,
    DocumentValidationProcessor,
    MedicalDocumentProcessor,
    matches_any_patient_name_candidate,
)
from app.services.image_preprocessing import ProcessedDocumentImage


def get_message_block(content: Any, index: int) -> dict[str, Any]:
    assert isinstance(content, list)
    block = content[index]
    assert isinstance(block, dict)
    return cast(dict[str, Any], block)


def get_http_exception_detail_dict(exc: HTTPException) -> dict[str, Any]:
    assert isinstance(exc.detail, dict)
    return cast(dict[str, Any], exc.detail)

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
        "name": "Budi Saputra",
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
    "medicines": [
        {
            "name": "Paracetamol",
            "quantity": 10,
            "unit": "TABLET",
            "dosage_instruction": "3 x 1",
            "unit_price": 1000,
            "total_price": 10000,
        }
    ],
    "other_charges": [
        {
            "category": "CONSULTATION",
            "description": "Konsultasi dokter",
            "amount": 115000,
        }
    ],
    "financial_recap": {
        "total_medicines": 10000,
        "total_other_charges": 115000,
        "grand_total": 125000,
    },
}


def build_settings(google_api_key: str = "test-google-key") -> Settings:
    return Settings(
        app_name="python-backend",
        app_env="test",
        openai_api_key="",
        google_api_key=google_api_key,
        jwt_secret="test-secret",
        access_token_expire_minutes=60,
        database_url="sqlite:///:memory:",
    )


def build_session_factory_with_ai_models(models: list[dict[str, object]]) -> sessionmaker:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)
    with session_factory() as db:
        for model in models:
            db.add(
                AIModel(
                    display_name=cast(str, model["display_name"]),
                    model_name=cast(str, model["model_name"]),
                    provider=cast(str, model["provider"]),
                    priority=cast(int, model.get("priority", 100)),
                    is_active=cast(bool, model["is_active"]),
                )
            )
        db.commit()
    return session_factory


VALIDATION_RESULT = {
    "packet_patient_name": None,
    "patient_name_match_status": "not_checked",
    "patient_name_match_reason": None,
    "review_required": False,
    "review_reasons": [],
    "images": [
        {
            "image_index": 0,
            "filename": "scan.jpg",
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


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    database_path = tmp_path / "ai-test.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{database_path.as_posix()}")
    monkeypatch.setenv("APP_ENV", "test")
    get_settings.cache_clear()
    clear_database_caches()

    app = create_app()
    with TestClient(app) as test_client:
        yield test_client

    clear_database_caches()
    get_settings.cache_clear()


def register_and_login(client: TestClient, *, email: str, full_name: str = "AI User") -> str:
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


def get_scan_log_by_id(scan_log_id: int):
    session_factory = get_session_factory()
    with session_factory() as db:
        return ScanLogRepository(db).get_by_id(scan_log_id)


def test_scan_documents_returns_validated_ai_payload(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = register_and_login(client, email="scan@example.com", full_name="Siti Aminah")
    family_member_response = client.post(
        "/api/family-members",
        headers={"Authorization": f"Bearer {token}"},
        json={"name": "Budi Saputra", "relation": "Child"},
    )
    assert family_member_response.status_code == 201

    def fake_preprocess_document_image(filename: str, content: bytes) -> ProcessedDocumentImage:
        assert filename == "scan.png"
        assert content == b"abc"
        return ProcessedDocumentImage(
            filename="scan.jpg",
            mime_type="image/jpeg",
            content=b"processed-image",
        )

    async def fake_validation_process(images: list[DocumentImageInput], **_: object) -> DocumentValidationResponse:
        assert len(images) == 1
        assert images[0].filename == "scan.jpg"
        return DocumentValidationResponse.model_validate(VALIDATION_RESULT)

    async def fake_process(
        images: list[DocumentImageInput],
        patient_name_candidates: list[str] | None = None,
        validation_hints: dict[str, object] | None = None,
    ) -> AIMedicalScanResponse:
        assert len(images) == 1
        assert images[0].filename == "scan.jpg"
        assert images[0].mime_type == "image/jpeg"
        assert images[0].base64_data == base64.b64encode(b"processed-image").decode("utf-8")
        assert patient_name_candidates == ["Siti Aminah", "Budi Saputra"]
        assert validation_hints is not None
        return AIMedicalScanResponse.model_validate(SCAN_RESULT)

    monkeypatch.setattr(ai_route, "preprocess_document_image", fake_preprocess_document_image)
    monkeypatch.setattr(ai_route.validation_processor, "process", fake_validation_process)
    monkeypatch.setattr(processor, "process", fake_process)

    response = client.post(
        "/api/ai/scan-documents",
        headers={"Authorization": f"Bearer {token}"},
        files=[("files", ("scan.png", b"abc", "image/png"))],
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["scan_log_id"] > 0
    assert payload["summary"] == SCAN_RESULT["summary"]
    assert payload["patient_info"] == SCAN_RESULT["patient_info"]
    assert payload["medical_context"] == SCAN_RESULT["medical_context"]
    assert payload["medicines"] == SCAN_RESULT["medicines"]
    assert payload["other_charges"] == SCAN_RESULT["other_charges"]
    assert payload["financial_recap"] == SCAN_RESULT["financial_recap"]
    assert "overall_confidence" not in payload
    assert "field_confidence" not in payload
    assert "review_required" not in payload
    assert "review_reasons" not in payload
    assert "calibration_version" not in payload

    scan_log = get_scan_log_by_id(payload["scan_log_id"])
    assert scan_log is not None
    assert scan_log.status == "succeeded"
    assert scan_log.validation_payload == VALIDATION_RESULT
    assert scan_log.extraction_payload == SCAN_RESULT
    assert scan_log.primary_model_provider == "google_genai"
    assert scan_log.primary_model_name == "gemini-3.1-flash-lite-preview"
    assert scan_log.ai_execution_mode == "single_model"
    assert len(scan_log.ai_runs) == 0
    assert scan_log.extraction_meta_payload is not None
    assert scan_log.extraction_meta_payload["medicine_postprocess"]["enabled"] is True
    assert len(scan_log.files) == 1
    assert scan_log.files[0].original_content == b"abc"
    assert scan_log.files[0].processed_content == b"processed-image"
    assert scan_log.files[0].original_sha256_hash == hashlib.sha256(b"abc").hexdigest()
    assert scan_log.files[0].processed_sha256_hash == hashlib.sha256(b"processed-image").hexdigest()


def test_scan_documents_sanitizes_nan_numbers_in_ai_payload(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = register_and_login(client, email="scan-nan-payload@example.com", full_name="Budi Saputra")

    def fake_preprocess_document_image(filename: str, content: bytes) -> ProcessedDocumentImage:
        assert filename == "scan.png"
        assert content == b"abc"
        return ProcessedDocumentImage(
            filename="scan.jpg",
            mime_type="image/jpeg",
            content=b"processed-image",
        )

    async def fake_validation_process(images: list[DocumentImageInput], **_: object) -> DocumentValidationResponse:
        assert len(images) == 1
        return DocumentValidationResponse.model_validate(VALIDATION_RESULT)

    async def fake_process(
        images: list[DocumentImageInput],
        patient_name_candidates: list[str] | None = None,
        validation_hints: dict[str, object] | None = None,
    ) -> AIMedicalScanResponse:
        del images, patient_name_candidates, validation_hints
        return AIMedicalScanResponse.model_validate(
            {
                **SCAN_RESULT,
                "summary": {
                    **SCAN_RESULT["summary"],
                    "total_amount": float("nan"),
                },
            }
        )

    monkeypatch.setattr(ai_route, "preprocess_document_image", fake_preprocess_document_image)
    monkeypatch.setattr(ai_route.validation_processor, "process", fake_validation_process)
    monkeypatch.setattr(processor, "process", fake_process)

    response = client.post(
        "/api/ai/scan-documents",
        headers={"Authorization": f"Bearer {token}"},
        files=[("files", ("scan.png", b"abc", "image/png"))],
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["total_amount"] is None


def test_scan_documents_rejects_validation_patient_name_outside_family_candidates_before_extraction(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = register_and_login(client, email="scan-name-mismatch@example.com", full_name="Siti Aminah")
    family_member_response = client.post(
        "/api/family-members",
        headers={"Authorization": f"Bearer {token}"},
        json={"name": "Budi Saputra", "relation": "Child"},
    )
    assert family_member_response.status_code == 201

    def fake_preprocess_document_image(filename: str, content: bytes) -> ProcessedDocumentImage:
        assert filename == "scan.png"
        assert content == b"abc"
        return ProcessedDocumentImage(
            filename="scan.jpg",
            mime_type="image/jpeg",
            content=b"processed-image",
        )

    async def fake_validation_process(images: list[DocumentImageInput], **_: object) -> DocumentValidationResponse:
        assert len(images) == 1
        return DocumentValidationResponse.model_validate(
            {
                **VALIDATION_RESULT,
                "packet_patient_name": "Andi Saputra",
                "patient_name_match_status": "mismatched",
                "patient_name_match_reason": "Document patient name 'Andi Saputra' does not match the user or any saved family member.",
            }
        )

    async def fake_process(
        images: list[DocumentImageInput],
        patient_name_candidates: list[str] | None = None,
        validation_hints: dict[str, object] | None = None,
    ) -> AIMedicalScanResponse:
        raise AssertionError("Extraction should not run when validation already found a patient-name mismatch.")

    monkeypatch.setattr(ai_route, "preprocess_document_image", fake_preprocess_document_image)
    monkeypatch.setattr(ai_route.validation_processor, "process", fake_validation_process)
    monkeypatch.setattr(processor, "process", fake_process)

    response = client.post(
        "/api/ai/scan-documents",
        headers={"Authorization": f"Bearer {token}"},
        files=[("files", ("scan.png", b"abc", "image/png"))],
    )

    assert response.status_code == 422
    payload = response.json()
    assert payload["message"] == "Patient name verification failed."
    assert payload["errors"] == [
        {
            "field": "non_field",
            "message": "Extracted patient name does not match the user or any saved family member. Please review your family members or rescan the document.",
        }
    ]

    scan_log = get_scan_log_by_id(1)
    assert scan_log is not None
    assert scan_log.status == "patient_name_mismatch"
    assert scan_log.validation_payload == {
        **VALIDATION_RESULT,
        "packet_patient_name": "Andi Saputra",
        "patient_name_match_status": "mismatched",
        "patient_name_match_reason": "Document patient name 'Andi Saputra' does not match the user or any saved family member.",
    }
    assert scan_log.extraction_payload is None
    assert scan_log.error_payload is not None
    assert scan_log.error_payload["detail"]["message"] == "Patient name verification failed."


def test_scan_documents_rejects_extraction_patient_name_on_fallback_when_validation_name_is_not_found(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = register_and_login(client, email="scan-name-fallback@example.com", full_name="Siti Aminah")
    family_member_response = client.post(
        "/api/family-members",
        headers={"Authorization": f"Bearer {token}"},
        json={"name": "Budi Saputra", "relation": "Child"},
    )
    assert family_member_response.status_code == 201

    def fake_preprocess_document_image(filename: str, content: bytes) -> ProcessedDocumentImage:
        assert filename == "scan.png"
        assert content == b"abc"
        return ProcessedDocumentImage(
            filename="scan.jpg",
            mime_type="image/jpeg",
            content=b"processed-image",
        )

    async def fake_validation_process(images: list[DocumentImageInput], **_: object) -> DocumentValidationResponse:
        assert len(images) == 1
        return DocumentValidationResponse.model_validate(
            {
                **VALIDATION_RESULT,
                "patient_name_match_status": "not_found",
                "patient_name_match_reason": "Patient name is not visible clearly enough in the uploaded documents.",
                "review_required": True,
                "review_reasons": ["Patient name is not visible clearly enough in the uploaded documents."],
            }
        )

    async def fake_process(
        images: list[DocumentImageInput],
        patient_name_candidates: list[str] | None = None,
        validation_hints: dict[str, object] | None = None,
    ) -> AIMedicalScanResponse:
        del validation_hints
        assert patient_name_candidates == ["Siti Aminah", "Budi Saputra"]
        mismatched_result = {
            **SCAN_RESULT,
            "patient_info": {
                **SCAN_RESULT["patient_info"],
                "name": "Andi Saputra",
            },
        }
        return AIMedicalScanResponse.model_validate(mismatched_result)

    monkeypatch.setattr(ai_route, "preprocess_document_image", fake_preprocess_document_image)
    monkeypatch.setattr(ai_route.validation_processor, "process", fake_validation_process)
    monkeypatch.setattr(processor, "process", fake_process)

    response = client.post(
        "/api/ai/scan-documents",
        headers={"Authorization": f"Bearer {token}"},
        files=[("files", ("scan.png", b"abc", "image/png"))],
    )

    assert response.status_code == 422
    payload = response.json()
    assert payload["message"] == "Patient name verification failed."
    assert payload["errors"] == [
        {
            "field": "non_field",
            "message": "Extracted patient name does not match the user or any saved family member. Please review your family members or rescan the document.",
        }
    ]

    scan_log = get_scan_log_by_id(1)
    assert scan_log is not None
    assert scan_log.status == "patient_name_mismatch"
    assert scan_log.validation_payload is not None
    assert scan_log.validation_payload["patient_name_match_status"] == "not_found"
    assert scan_log.extraction_payload is not None
    assert scan_log.extraction_payload["patient_info"]["name"] == "Andi Saputra"


def test_scan_documents_returns_validation_failure_before_extraction(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = register_and_login(client, email="scan-validation-failure@example.com", full_name="Siti Aminah")

    def fake_preprocess_document_image(filename: str, content: bytes) -> ProcessedDocumentImage:
        assert filename == "scan.png"
        assert content == b"abc"
        return ProcessedDocumentImage(
            filename="scan.jpg",
            mime_type="image/jpeg",
            content=b"processed-image",
        )

    async def fake_validation_process(images: list[DocumentImageInput], **_: object) -> DocumentValidationResponse:
        assert len(images) == 1
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Document validation failed.",
                "errors": [
                    {
                        "field": "non_field",
                        "message": "Image 1 document 1: a visible facility stamp or doctor signature is required.",
                    }
                ],
            },
        )

    async def fake_process(
        images: list[DocumentImageInput],
        patient_name_candidates: list[str] | None = None,
        validation_hints: dict[str, object] | None = None,
    ) -> AIMedicalScanResponse:
        raise AssertionError("Extraction should not run when validation fails.")

    monkeypatch.setattr(ai_route, "preprocess_document_image", fake_preprocess_document_image)
    monkeypatch.setattr(ai_route.validation_processor, "process", fake_validation_process)
    monkeypatch.setattr(processor, "process", fake_process)

    response = client.post(
        "/api/ai/scan-documents",
        headers={"Authorization": f"Bearer {token}"},
        files=[("files", ("scan.png", b"abc", "image/png"))],
    )

    assert response.status_code == 422
    payload = response.json()
    assert payload["message"] == "Document validation failed."
    assert payload["errors"] == [
        {"field": "non_field", "message": "Image 1 document 1: a visible facility stamp or doctor signature is required."}
    ]

    scan_log = get_scan_log_by_id(1)
    assert scan_log is not None
    assert scan_log.status == "validation_failed"
    assert scan_log.validation_payload is None
    assert scan_log.extraction_payload is None
    assert scan_log.error_payload is not None
    assert scan_log.error_payload["detail"]["message"] == "Document validation failed."


def test_scan_documents_accepts_multi_document_validation_result_for_single_image(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = register_and_login(client, email="scan-multi-doc@example.com", full_name="Siti Aminah")
    family_member_response = client.post(
        "/api/family-members",
        headers={"Authorization": f"Bearer {token}"},
        json={"name": "Budi Saputra", "relation": "Child"},
    )
    assert family_member_response.status_code == 201

    def fake_preprocess_document_image(filename: str, content: bytes) -> ProcessedDocumentImage:
        assert filename == "scan.png"
        assert content == b"abc"
        return ProcessedDocumentImage(
            filename="scan.jpg",
            mime_type="image/jpeg",
            content=b"processed-image",
        )

    async def fake_validation_process(images: list[DocumentImageInput], **_: object) -> DocumentValidationResponse:
        assert len(images) == 1
        return DocumentValidationResponse.model_validate(
            {
                "images": [
                    {
                        "image_index": 0,
                        "filename": "scan.jpg",
                        "document_count_detected": 2,
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
                            },
                            {
                                "document_index": 1,
                                "document_type": "prescription",
                                "is_medical_document": True,
                                "readability_ok": True,
                                "facility_identity_visible": True,
                                "stamp_detected": False,
                                "signature_detected": True,
                                "decision": "valid",
                                "reasons": ["Doctor signature is visible."],
                            },
                        ],
                    }
                ]
            }
        )

    async def fake_process(
        images: list[DocumentImageInput],
        patient_name_candidates: list[str] | None = None,
        validation_hints: dict[str, object] | None = None,
    ) -> AIMedicalScanResponse:
        assert len(images) == 1
        assert validation_hints == {
            "ordered_documents": [
                {
                    "image_index": 0,
                    "document_index": 0,
                    "document_type": "receipt",
                    "decision": "valid",
                },
                {
                    "image_index": 0,
                    "document_index": 1,
                    "document_type": "prescription",
                    "decision": "valid",
                },
            ],
            "document_type_counts": {
                "receipt": 1,
                "prescription": 1,
            },
            "image_document_counts": [
                {
                    "image_index": 0,
                    "filename": "scan.jpg",
                    "document_count_detected": 2,
                    "validated_document_count": 2,
                }
            ],
            "total_detected_documents": 2,
        }
        return AIMedicalScanResponse.model_validate(SCAN_RESULT)

    monkeypatch.setattr(ai_route, "preprocess_document_image", fake_preprocess_document_image)
    monkeypatch.setattr(ai_route.validation_processor, "process", fake_validation_process)
    monkeypatch.setattr(processor, "process", fake_process)

    response = client.post(
        "/api/ai/scan-documents",
        headers={"Authorization": f"Bearer {token}"},
        files=[("files", ("scan.png", b"abc", "image/png"))],
    )

    assert response.status_code == 200


def test_scan_documents_merges_validation_review_flags_into_response(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = register_and_login(client, email="scan-validation-review@example.com", full_name="Siti Aminah")
    family_member_response = client.post(
        "/api/family-members",
        headers={"Authorization": f"Bearer {token}"},
        json={"name": "Budi Saputra", "relation": "Child"},
    )
    assert family_member_response.status_code == 201

    def fake_preprocess_document_image(filename: str, content: bytes) -> ProcessedDocumentImage:
        assert filename == "scan.png"
        assert content == b"abc"
        return ProcessedDocumentImage(
            filename="scan.jpg",
            mime_type="image/jpeg",
            content=b"processed-image",
        )

    async def fake_validation_process(images: list[DocumentImageInput], **_: object) -> DocumentValidationResponse:
        assert len(images) == 1
        return DocumentValidationResponse.model_validate(
            {
                "review_required": True,
                "review_reasons": [
                    "Image 1: multiple documents were present but could not be validated clearly.",
                ],
                "images": [
                    {
                        "image_index": 0,
                        "filename": "scan.jpg",
                        "document_count_detected": 2,
                        "documents": [
                            {
                                "document_index": 0,
                                "document_type": "receipt",
                                "is_medical_document": True,
                                "readability_ok": True,
                                "facility_identity_visible": True,
                                "stamp_detected": False,
                                "signature_detected": False,
                                "decision": "valid",
                                "reasons": ["Facility identity is visible."],
                            }
                        ],
                    }
                ],
            }
        )

    async def fake_process(
        images: list[DocumentImageInput],
        patient_name_candidates: list[str] | None = None,
        validation_hints: dict[str, object] | None = None,
    ) -> AIMedicalScanResponse:
        del images, patient_name_candidates, validation_hints
        return AIMedicalScanResponse.model_validate(SCAN_RESULT)

    monkeypatch.setattr(ai_route, "preprocess_document_image", fake_preprocess_document_image)
    monkeypatch.setattr(ai_route.validation_processor, "process", fake_validation_process)
    monkeypatch.setattr(processor, "process", fake_process)

    response = client.post(
        "/api/ai/scan-documents",
        headers={"Authorization": f"Bearer {token}"},
        files=[("files", ("scan.png", b"abc", "image/png"))],
    )

    assert response.status_code == 200
    payload = response.json()
    assert "overall_confidence" not in payload
    assert "field_confidence" not in payload
    assert "review_required" not in payload
    assert "review_reasons" not in payload
    assert "calibration_version" not in payload

    scan_log = get_scan_log_by_id(payload["scan_log_id"])
    assert scan_log is not None
    assert scan_log.validation_payload is not None
    assert scan_log.validation_payload["review_required"] is True
    assert scan_log.validation_payload["review_reasons"] == [
        "Image 1: multiple documents were present but could not be validated clearly."
    ]

    assert response.status_code == 200
    payload = response.json()
    assert payload["scan_log_id"] > 0
    assert payload["summary"] == SCAN_RESULT["summary"]
    assert payload["patient_info"] == SCAN_RESULT["patient_info"]
    assert payload["medical_context"] == SCAN_RESULT["medical_context"]
    assert payload["medicines"] == SCAN_RESULT["medicines"]
    assert payload["other_charges"] == SCAN_RESULT["other_charges"]
    assert payload["financial_recap"] == SCAN_RESULT["financial_recap"]

    scan_log = get_scan_log_by_id(payload["scan_log_id"])
    assert scan_log is not None
    assert scan_log.status == "succeeded"
    assert len(scan_log.files) == 1


def test_scan_documents_rejects_invalid_image_upload(client: TestClient) -> None:
    token = register_and_login(client, email="invalid-image@example.com", full_name="Siti Aminah")

    response = client.post(
        "/api/ai/scan-documents",
        headers={"Authorization": f"Bearer {token}"},
        files=[("files", ("broken.png", b"not-an-image", "image/png"))],
    )

    assert response.status_code == 422
    payload = response.json()
    assert payload["message"] == "Input validation failed."
    assert payload["errors"] == [{"field": "non_field", "message": "Uploaded file is not a readable image."}]

    scan_log = get_scan_log_by_id(1)
    assert scan_log is not None
    assert scan_log.status == "preprocess_failed"
    assert len(scan_log.files) == 1
    assert scan_log.files[0].original_content == b"not-an-image"
    assert scan_log.files[0].processed_content is None


def test_build_messages_includes_rendered_prompt_and_base64_images() -> None:
    medical_processor = MedicalDocumentProcessor(settings=build_settings())
    prompt_bundle = medical_processor._load_prompt_bundle()

    messages = medical_processor._build_messages(
        images=[
            DocumentImageInput(
                filename="scan.png",
                mime_type="image/png",
                base64_data="YWJj",
            )
        ],
        patient_name_candidates=["Siti Aminah", "Budi Saputra"],
        prompt_bundle=prompt_bundle,
    )

    text_block = get_message_block(messages[1].content, 0)
    image_block = get_message_block(messages[1].content, 1)
    image_url = cast(dict[str, Any], image_block["image_url"])

    assert "Return only valid JSON" in str(messages[0].content)
    assert "Analyze the 1 uploaded medical document image(s)." in cast(str, text_block["text"])
    assert '["Siti Aminah", "Budi Saputra"]' in cast(str, text_block["text"])
    assert image_block["type"] == "image_url"
    assert cast(str, image_url["url"]).startswith("data:image/png;base64,")
    assert image_url["url"] == "data:image/png;base64,YWJj"


def test_build_messages_remain_image_url_after_langchain_normalization() -> None:
    medical_processor = MedicalDocumentProcessor(settings=build_settings())
    prompt_bundle = medical_processor._load_prompt_bundle()
    messages = medical_processor._build_messages(
        images=[
            DocumentImageInput(
                filename="scan.png",
                mime_type="image/png",
                base64_data="YWJj",
            )
        ],
        patient_name_candidates=["Siti Aminah"],
        prompt_bundle=prompt_bundle,
    )

    normalized_messages = _normalize_messages(messages)
    image_block = get_message_block(normalized_messages[1].content, 1)
    image_url = cast(dict[str, Any], image_block["image_url"])

    assert image_block["type"] == "image_url"
    assert image_url["url"] == "data:image/png;base64,YWJj"


def test_langchain_normalization_does_not_rewrite_built_image_to_base64_block() -> None:
    medical_processor = MedicalDocumentProcessor(settings=build_settings())
    prompt_bundle = medical_processor._load_prompt_bundle()
    messages = medical_processor._build_messages(
        images=[
            DocumentImageInput(
                filename="scan.png",
                mime_type="image/png",
                base64_data="YWJj",
            )
        ],
        patient_name_candidates=["Siti Aminah"],
        prompt_bundle=prompt_bundle,
    )

    normalized_messages = _normalize_messages(messages)
    image_block = get_message_block(normalized_messages[1].content, 1)
    image_url = cast(dict[str, Any], image_block["image_url"])

    assert image_block["type"] == "image_url"
    assert "base64" not in image_block
    assert image_url["url"] == "data:image/png;base64,YWJj"


def test_process_returns_validated_result_from_model_json(monkeypatch: pytest.MonkeyPatch) -> None:
    medical_processor = MedicalDocumentProcessor(settings=build_settings())

    class FakeModel:
        async def ainvoke(self, messages: list[object]) -> AIMessage:
            assert len(messages) == 2
            return AIMessage(content=f"```json\n{json.dumps(SCAN_RESULT)}\n```")

    monkeypatch.setattr(
        medical_processor,
        "_create_chat_model",
        lambda model_spec: FakeModel(),
    )
    monkeypatch.setattr(
        medical_processor,
        "_get_enabled_model_specs",
        lambda: [AIModelSpec(provider="google_genai", model="gemini-1.5-flash")],
    )

    result = asyncio.run(
        medical_processor.process(
            [
                DocumentImageInput(
                    filename="scan.png",
                    mime_type="image/png",
                    base64_data="YWJj",
                )
            ]
        )
    )

    assert result.result.model_dump() == SCAN_RESULT
    assert result.ai_execution_mode == "ensemble"
    assert result.overall_confidence is not None


def test_process_rejects_non_json_model_output(monkeypatch: pytest.MonkeyPatch) -> None:
    medical_processor = MedicalDocumentProcessor(settings=build_settings())

    class FakeModel:
        async def ainvoke(self, messages: list[object]) -> AIMessage:
            assert len(messages) == 2
            return AIMessage(content="This is not JSON")

    monkeypatch.setattr(
        medical_processor,
        "_create_chat_model",
        lambda model_spec: FakeModel(),
    )
    monkeypatch.setattr(
        medical_processor,
        "_get_enabled_model_specs",
        lambda: [AIModelSpec(provider="google_genai", model="gemini-1.5-flash")],
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            medical_processor.process(
                [
                    DocumentImageInput(
                        filename="scan.png",
                        mime_type="image/png",
                        base64_data="YWJj",
                    )
                ]
            )
        )

    assert exc_info.value.status_code == 502
    assert exc_info.value.detail == "AI ensemble failed to produce enough valid outputs."


def test_process_rejects_invalid_json_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    medical_processor = MedicalDocumentProcessor(settings=build_settings())

    class FakeModel:
        async def ainvoke(self, messages: list[object]) -> AIMessage:
            assert len(messages) == 2
            return AIMessage(content=json.dumps({"summary": {}}))

    monkeypatch.setattr(
        medical_processor,
        "_create_chat_model",
        lambda model_spec: FakeModel(),
    )
    monkeypatch.setattr(
        medical_processor,
        "_get_enabled_model_specs",
        lambda: [AIModelSpec(provider="google_genai", model="gemini-1.5-flash")],
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            medical_processor.process(
                [
                    DocumentImageInput(
                        filename="scan.png",
                        mime_type="image/png",
                        base64_data="YWJj",
                    )
                ]
            )
        )

    assert exc_info.value.status_code == 502
    assert exc_info.value.detail == "AI ensemble failed to produce enough valid outputs."


def test_create_chat_model_requires_google_api_key() -> None:
    medical_processor = MedicalDocumentProcessor(settings=build_settings(google_api_key=""))

    with pytest.raises(HTTPException) as exc_info:
        medical_processor._create_chat_model(
            AIModelSpec(provider="google_genai", model="gemini-1.5-flash")
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "GOOGLE_API_KEY is not configured."


def test_get_enabled_model_specs_requires_at_least_one_active_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    medical_processor = MedicalDocumentProcessor(settings=build_settings())
    empty_session_factory = build_session_factory_with_ai_models([])

    monkeypatch.setattr("app.services.ai_service.get_session_factory", lambda: empty_session_factory)

    with pytest.raises(HTTPException) as exc_info:
        medical_processor._get_enabled_model_specs()

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "No active AI model is configured."


def test_process_uses_first_active_database_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    medical_processor = MedicalDocumentProcessor(settings=build_settings())
    session_factory = build_session_factory_with_ai_models(
        [
            {
                "display_name": "Model A",
                "model_name": "model-a",
                "provider": "google_genai",
                "priority": 20,
                "is_active": True,
            },
            {
                "display_name": "Model B",
                "model_name": "model-b",
                "provider": "google_genai",
                "priority": 5,
                "is_active": True,
            },
        ]
    )
    selected_models: list[str] = []

    class FakeModel:
        async def ainvoke(self, messages: list[object]) -> AIMessage:
            assert len(messages) == 2
            return AIMessage(content=f"```json\n{json.dumps(SCAN_RESULT)}\n```")

    monkeypatch.setattr("app.services.ai_service.get_session_factory", lambda: session_factory)
    monkeypatch.setattr(
        medical_processor,
        "_create_chat_model",
        lambda model_spec: selected_models.append(model_spec.model) or FakeModel(),
    )

    result = asyncio.run(
        medical_processor.process(
            [
                DocumentImageInput(
                    filename="scan.png",
                    mime_type="image/png",
                    base64_data="YWJj",
                )
            ]
        )
    )

    assert result.result.model_dump() == SCAN_RESULT
    assert selected_models == ["model-b", "model-b", "model-a", "model-a"]


def test_ainvoke_with_tracking_persists_usage_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    medical_processor = MedicalDocumentProcessor(settings=build_settings())
    session_factory = build_session_factory_with_ai_models([])

    class FakeResponse:
        def __init__(self) -> None:
            self.content = json.dumps(SCAN_RESULT)
            self.usage_metadata = {
                "input_tokens": 123,
                "output_tokens": 45,
                "total_tokens": 168,
            }

    class FakeModel:
        async def ainvoke(self, messages: list[object]) -> FakeResponse:
            assert len(messages) == 2
            return FakeResponse()

    monkeypatch.setattr("app.services.ai_service.get_session_factory", lambda: session_factory)

    response, tracking_payload = asyncio.run(
        medical_processor._ainvoke_with_tracking(
            FakeModel(),
            [object(), object()],
            provider="google_genai",
            model_name="gemini-test",
            call_kind="scan_extraction",
            success_context={"request_context_payload": {"image_count": 1}},
        )
    )

    assert response.content == json.dumps(SCAN_RESULT)
    assert tracking_payload["prompt_tokens"] == 123
    assert tracking_payload["completion_tokens"] == 45
    assert tracking_payload["total_tokens"] == 168

    with session_factory() as db:
        rows = db.execute(select(AIApiCallLog)).scalars().all()
        assert len(rows) == 1
        assert rows[0].provider == "google_genai"
        assert rows[0].call_kind == "scan_extraction"
        assert rows[0].prompt_tokens == 123
        assert rows[0].completion_tokens == 45
        assert rows[0].total_tokens == 168


def test_ainvoke_with_tracking_persists_failure_without_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    medical_processor = MedicalDocumentProcessor(settings=build_settings())
    session_factory = build_session_factory_with_ai_models([])

    class FakeModel:
        async def ainvoke(self, messages: list[object]) -> AIMessage:
            assert len(messages) == 1
            raise RuntimeError("provider boom")

    monkeypatch.setattr("app.services.ai_service.get_session_factory", lambda: session_factory)

    with pytest.raises(RuntimeError, match="provider boom"):
        asyncio.run(
            medical_processor._ainvoke_with_tracking(
            FakeModel(),
            [object()],
            provider="google_genai",
            model_name="gemini-test",
            call_kind="field_judge",
            success_context={},
        )
        )

    with session_factory() as db:
        rows = db.execute(select(AIApiCallLog)).scalars().all()
        assert len(rows) == 1
        assert rows[0].success is False
        assert rows[0].error_message == "provider boom"
        assert rows[0].prompt_tokens is None


def test_process_uses_judge_for_service_date_conflict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    medical_processor = MedicalDocumentProcessor(settings=build_settings())

    class FakeModel:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        async def ainvoke(self, messages: list[object]) -> AIMessage:
            payload = {
                **SCAN_RESULT,
                "summary": {
                    **SCAN_RESULT["summary"],
                    "service_date": "2024-08-10" if self.model_name == "model-a" else "2024-08-11",
                },
            }
            return AIMessage(content=json.dumps(payload))

    async def fake_judge_field_conflict(
        field_path: str,
        images: list[object],
        candidates: list[dict[str, object]],
    ) -> tuple[str | None, str | None]:
        assert field_path == "summary.service_date"
        del images
        selected = next(candidate for candidate in candidates if candidate["raw_value"] == "2024-08-11")
        return cast(str, selected["normalized_key"]), "Consultation note date is better supported."

    monkeypatch.setattr(
        medical_processor,
        "_create_chat_model",
        lambda model_spec: FakeModel(model_spec.model),
    )
    monkeypatch.setattr(
        medical_processor,
        "_get_enabled_model_specs",
        lambda: [
            AIModelSpec(provider="google_genai", model="model-a"),
            AIModelSpec(provider="google_genai", model="model-b"),
        ],
    )
    monkeypatch.setattr(medical_processor, "_judge_field_conflict", fake_judge_field_conflict)

    result = asyncio.run(
        medical_processor.process(
            [
                DocumentImageInput(
                    filename="scan.png",
                    mime_type="image/png",
                    base64_data="YWJj",
                )
            ]
        )
    )

    assert result.result.summary.service_date == "2024-08-11"
    assert result.field_confidence["summary.service_date"]["judge_used"] is True
    assert result.extraction_meta_payload["judge_invocations"][0]["field_path"] == "summary.service_date"


def test_process_does_not_invoke_judge_for_equivalent_service_date_formats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    medical_processor = MedicalDocumentProcessor(settings=build_settings())

    class FakeModel:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        async def ainvoke(self, messages: list[object]) -> AIMessage:
            del messages
            payload = {
                **SCAN_RESULT,
                "summary": {
                    **SCAN_RESULT["summary"],
                    "service_date": "2024-08-10" if self.model_name == "model-a" else "10/08/2024",
                },
            }
            return AIMessage(content=json.dumps(payload))

    async def fail_if_called(
        field_path: str,
        images: list[object],
        candidates: list[dict[str, object]],
    ) -> tuple[str | None, str | None]:
        del field_path, images, candidates
        raise AssertionError("Judge should not be used when service dates normalize to the same key.")

    monkeypatch.setattr(
        medical_processor,
        "_create_chat_model",
        lambda model_spec: FakeModel(model_spec.model),
    )
    monkeypatch.setattr(
        medical_processor,
        "_get_enabled_model_specs",
        lambda: [
            AIModelSpec(provider="google_genai", model="model-a"),
            AIModelSpec(provider="google_genai", model="model-b"),
        ],
    )
    monkeypatch.setattr(medical_processor, "_judge_field_conflict", fail_if_called)

    result = asyncio.run(
        medical_processor.process(
            [
                DocumentImageInput(
                    filename="scan.png",
                    mime_type="image/png",
                    base64_data="YWJj",
                )
            ]
        )
    )

    assert result.result.summary.service_date == "2024-08-10"
    assert result.field_confidence["summary.service_date"]["judge_used"] is False
    assert result.extraction_meta_payload["judge_invocations"] == []


def test_process_keeps_transaction_date_null_when_not_explicitly_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    medical_processor = MedicalDocumentProcessor(settings=build_settings())

    class FakeModel:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        async def ainvoke(self, messages: list[object]) -> AIMessage:
            del messages
            payload = {
                **SCAN_RESULT,
                "summary": {
                    **SCAN_RESULT["summary"],
                    "service_date": "2024-08-10",
                    "transaction_date": None,
                },
            }
            return AIMessage(content=json.dumps(payload))

    monkeypatch.setattr(
        medical_processor,
        "_create_chat_model",
        lambda model_spec: FakeModel(model_spec.model),
    )
    monkeypatch.setattr(
        medical_processor,
        "_get_enabled_model_specs",
        lambda: [
            AIModelSpec(provider="google_genai", model="model-a"),
            AIModelSpec(provider="google_genai", model="model-b"),
        ],
    )

    result = asyncio.run(
        medical_processor.process(
            [
                DocumentImageInput(
                    filename="scan.png",
                    mime_type="image/png",
                    base64_data="YWJj",
                )
            ]
        )
    )

    assert result.result.summary.service_date == "2024-08-10"
    assert result.result.summary.transaction_date is None


def test_process_keeps_supported_explicit_gender_when_only_null_alternative_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    medical_processor = MedicalDocumentProcessor(settings=build_settings())

    class FakeModel:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        async def ainvoke(self, messages: list[object]) -> AIMessage:
            del messages
            payload = {
                **SCAN_RESULT,
                "patient_info": {
                    **SCAN_RESULT["patient_info"],
                    "gender": "Pria" if self.model_name == "model-a" else None,
                },
            }
            return AIMessage(content=json.dumps(payload))

    monkeypatch.setattr(
        medical_processor,
        "_create_chat_model",
        lambda model_spec: FakeModel(model_spec.model),
    )
    monkeypatch.setattr(
        medical_processor,
        "_get_enabled_model_specs",
        lambda: [
            AIModelSpec(provider="google_genai", model="model-a"),
            AIModelSpec(provider="google_genai", model="model-b"),
        ],
    )

    result = asyncio.run(
        medical_processor.process(
            [
                DocumentImageInput(
                    filename="scan.png",
                    mime_type="image/png",
                    base64_data="YWJj",
                )
            ]
        )
    )

    assert result.result.patient_info.gender == "MALE"
    assert "prefer null over inferred gender when model support is ambiguous" not in result.field_confidence["patient_info.gender"]["rule_hits"]


def test_process_prefers_null_gender_when_competing_labels_are_ambiguous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    medical_processor = MedicalDocumentProcessor(settings=build_settings())

    class FakeModel:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        async def ainvoke(self, messages: list[object]) -> AIMessage:
            del messages
            if self.model_name == "model-a":
                gender_value = "Pria"
            elif self.model_name == "model-b":
                gender_value = "Wanita"
            else:
                gender_value = None
            payload = {
                **SCAN_RESULT,
                "patient_info": {
                    **SCAN_RESULT["patient_info"],
                    "gender": gender_value,
                },
            }
            return AIMessage(content=json.dumps(payload))

    monkeypatch.setattr(
        medical_processor,
        "_create_chat_model",
        lambda model_spec: FakeModel(model_spec.model),
    )
    monkeypatch.setattr(
        medical_processor,
        "_get_enabled_model_specs",
        lambda: [
            AIModelSpec(provider="google_genai", model="model-a"),
            AIModelSpec(provider="google_genai", model="model-b"),
            AIModelSpec(provider="google_genai", model="model-c"),
        ],
    )

    result = asyncio.run(
        medical_processor.process(
            [
                DocumentImageInput(
                    filename="scan.png",
                    mime_type="image/png",
                    base64_data="YWJj",
                )
            ]
        )
    )

    assert result.result.patient_info.gender is None
    assert "prefer null over inferred gender when model support is ambiguous" in result.field_confidence["patient_info.gender"]["rule_hits"]


def test_process_keeps_explicit_gender_when_null_support_is_below_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    medical_processor = MedicalDocumentProcessor(settings=build_settings())

    # Keep the test deterministic with one run per model and 5 producers:
    # male=2, female=2, null=1 => null support ratio = 0.2 (< 0.25 threshold).
    monkeypatch.setattr(ai_service_module, "AI_EXTRACTION_RUNS_PER_MODEL", 1)
    monkeypatch.setattr(ai_service_module, "AI_EXTRACTION_MAX_PRODUCERS", 5)

    class FakeModel:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        async def ainvoke(self, messages: list[object]) -> AIMessage:
            del messages
            if self.model_name in {"model-a", "model-c"}:
                gender_value = "Pria"
            elif self.model_name in {"model-b", "model-d"}:
                gender_value = "Wanita"
            else:
                gender_value = None
            payload = {
                **SCAN_RESULT,
                "patient_info": {
                    **SCAN_RESULT["patient_info"],
                    "gender": gender_value,
                },
            }
            return AIMessage(content=json.dumps(payload))

    monkeypatch.setattr(
        medical_processor,
        "_create_chat_model",
        lambda model_spec: FakeModel(model_spec.model),
    )
    monkeypatch.setattr(
        medical_processor,
        "_get_enabled_model_specs",
        lambda: [
            AIModelSpec(provider="google_genai", model="model-a"),
            AIModelSpec(provider="google_genai", model="model-b"),
            AIModelSpec(provider="google_genai", model="model-c"),
            AIModelSpec(provider="google_genai", model="model-d"),
            AIModelSpec(provider="google_genai", model="model-e"),
        ],
    )

    result = asyncio.run(
        medical_processor.process(
            [
                DocumentImageInput(
                    filename="scan.png",
                    mime_type="image/png",
                    base64_data="YWJj",
                )
            ]
        )
    )

    assert result.result.patient_info.gender in {"MALE", "FEMALE"}
    assert "prefer null over inferred gender when model support is ambiguous" not in result.field_confidence["patient_info.gender"]["rule_hits"]


def test_reconciliation_deferral_guard_does_not_trigger_below_delta_threshold() -> None:
    async def noop_judge(
        field_path: str,
        images: list[Any],
        candidates: list[dict[str, Any]],
    ) -> tuple[str | None, str | None]:
        del field_path, images, candidates
        return None, None

    engine = MedicalExtractionFusionEngine(judge_field=noop_judge)
    result = AIMedicalScanResponse.model_validate(SCAN_RESULT)

    supported_targets = {
        "summary.total_amount": [
            {"value": 125000, "support_count": 5, "producer_count": 0},
            {"value": 168000, "support_count": 1, "producer_count": 0},
            {"value": 200000, "support_count": 1, "producer_count": 0},
        ],
        "financial_recap.grand_total": [
            {"value": 125000, "support_count": 4, "producer_count": 0},
        ],
    }

    below_threshold_subset = {
        "grand_total": 168000,  # relative delta = 0.344 < 0.35
        "canonical_medicine_source": {"support_count": 1, "producer_count": 0},
        "dropped_group_indices": [0],
    }
    assert engine._should_defer_coherent_financial_override(
        result=result,
        best_subset=below_threshold_subset,
        supported_targets=supported_targets,
    ) is False

    above_threshold_subset = {
        "grand_total": 200000,  # relative delta = 0.6 > 0.35
        "canonical_medicine_source": {"support_count": 1, "producer_count": 0},
        "dropped_group_indices": [0],
    }
    assert engine._should_defer_coherent_financial_override(
        result=result,
        best_subset=above_threshold_subset,
        supported_targets=supported_targets,
    ) is True


def test_reconciliation_deferral_guard_support_threshold_boundary() -> None:
    async def noop_judge(
        field_path: str,
        images: list[Any],
        candidates: list[dict[str, Any]],
    ) -> tuple[str | None, str | None]:
        del field_path, images, candidates
        return None, None

    engine = MedicalExtractionFusionEngine(judge_field=noop_judge)
    result = AIMedicalScanResponse.model_validate(SCAN_RESULT)
    supported_targets = {
        "summary.total_amount": [
            {"value": 125000, "support_count": 5, "producer_count": 0},
            {"value": 200000, "support_count": 1, "producer_count": 0},
        ],
        "financial_recap.grand_total": [
            {"value": 125000, "support_count": 4, "producer_count": 0},
        ],
    }

    at_threshold_subset = {
        "grand_total": 200000,
        "canonical_medicine_source": {"support_count": 2, "producer_count": 1},  # support score = 2.2
        "dropped_group_indices": [0],
    }
    assert engine._should_defer_coherent_financial_override(
        result=result,
        best_subset=at_threshold_subset,
        supported_targets=supported_targets,
    ) is True

    above_threshold_subset = {
        "grand_total": 200000,
        "canonical_medicine_source": {"support_count": 2, "producer_count": 2},  # support score = 2.4
        "dropped_group_indices": [0],
    }
    assert engine._should_defer_coherent_financial_override(
        result=result,
        best_subset=above_threshold_subset,
        supported_targets=supported_targets,
    ) is False


def test_reconciliation_deferral_guard_drop_count_threshold_boundary() -> None:
    async def noop_judge(
        field_path: str,
        images: list[Any],
        candidates: list[dict[str, Any]],
    ) -> tuple[str | None, str | None]:
        del field_path, images, candidates
        return None, None

    engine = MedicalExtractionFusionEngine(judge_field=noop_judge)
    result = AIMedicalScanResponse.model_validate(SCAN_RESULT)
    supported_targets = {
        "summary.total_amount": [
            {"value": 125000, "support_count": 5, "producer_count": 0},
            {"value": 200000, "support_count": 1, "producer_count": 0},
        ],
        "financial_recap.grand_total": [
            {"value": 125000, "support_count": 4, "producer_count": 0},
        ],
    }

    below_drop_threshold_subset = {
        "grand_total": 200000,
        "canonical_medicine_source": {"support_count": 1, "producer_count": 0},
        "dropped_group_indices": [],
    }
    assert engine._should_defer_coherent_financial_override(
        result=result,
        best_subset=below_drop_threshold_subset,
        supported_targets=supported_targets,
    ) is False

    at_drop_threshold_subset = {
        "grand_total": 200000,
        "canonical_medicine_source": {"support_count": 1, "producer_count": 0},
        "dropped_group_indices": [0],
    }
    assert engine._should_defer_coherent_financial_override(
        result=result,
        best_subset=at_drop_threshold_subset,
        supported_targets=supported_targets,
    ) is True


def test_parse_indonesian_amount_words_parses_standard_terbilang_phrases() -> None:
    value_1, confidence_1 = parse_indonesian_amount_words("tiga ratus ribu rupiah")
    assert value_1 == 300000
    assert confidence_1 > 0.7

    value_2, confidence_2 = parse_indonesian_amount_words("seratus sembilan puluh lima ribu")
    assert value_2 == 195000
    assert confidence_2 > 0.7


def test_parse_indonesian_amount_words_handles_noise_and_numeric_fallback() -> None:
    value, confidence = parse_indonesian_amount_words("untuk pembayaran: terbilang Rp 195.000,-")
    assert value == 195000
    assert confidence > 0.5


def test_extract_terbilang_consistency_prefers_terbilang_when_numeric_is_weak_and_conflicting() -> None:
    async def noop_judge(
        field_path: str,
        images: list[Any],
        candidates: list[dict[str, Any]],
    ) -> tuple[str | None, str | None]:
        del field_path, images, candidates
        return None, None

    engine = MedicalExtractionFusionEngine(judge_field=noop_judge)
    result = AIMedicalScanResponse.model_validate(SCAN_RESULT)
    selected_groups = [
        {
            "score": 0.2,
            "support_count": 1,
            "producer_count": 0,
            "path_prefix": "other_charges[item_0]",
            "representative_value": {
                "category": "Kwitansi",
                "description": "terbilang tiga ratus ribu rupiah",
                "amount": 210000,
            },
        }
    ]

    consistency = engine._extract_terbilang_consistency(
        result=result,
        selected_groups=selected_groups,
    )

    assert consistency["detected"] is True
    assert consistency["conflict"] is True
    assert consistency["selected_source"] == "terbilang"
    assert consistency["selected_value"] == 300000
    assert "terbilang_overrode_numeric" in consistency["notes"]


def test_process_normalizes_explicit_gender_labels_to_canonical_enum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    medical_processor = MedicalDocumentProcessor(settings=build_settings())

    class FakeModel:
        async def ainvoke(self, messages: list[object]) -> AIMessage:
            del messages
            payload = {
                **SCAN_RESULT,
                "patient_info": {
                    **SCAN_RESULT["patient_info"],
                    "gender": "Pria",
                },
            }
            return AIMessage(content=json.dumps(payload))

    monkeypatch.setattr(
        medical_processor,
        "_create_chat_model",
        lambda model_spec: FakeModel(),
    )
    monkeypatch.setattr(
        medical_processor,
        "_get_enabled_model_specs",
        lambda: [AIModelSpec(provider="google_genai", model="model-a")],
    )

    result = asyncio.run(
        medical_processor.process(
            [
                DocumentImageInput(
                    filename="scan.png",
                    mime_type="image/png",
                    base64_data="YWJj",
                )
            ]
        )
    )

    assert result.result.patient_info.gender == "MALE"
    assert "normalized local gender label to canonical enum" in result.field_confidence["patient_info.gender"]["rule_hits"]
    assert result.extraction_meta_payload["marked_choice_inferences"] == [
        {
            "field_path": "patient_info.gender",
            "value": "MALE",
            "reason": "preserved explicit gender selection from a marked exclusive-choice form field",
        }
    ]


def test_process_keeps_consensus_lists_coherent_and_avoids_inflated_totals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    medical_processor = MedicalDocumentProcessor(settings=build_settings())

    google_outputs = [
        {
            "summary": {
                "detected_documents": ["Formulir Rawat Jalan", "Kwitansi Klinik Utama Makmuri", "Salinan Resep", "Kwitansi Apotek Pojodo"],
                "medical_facility_name": "Klinik Utama Makmuri",
                "service_date": "28/08/2025",
                "transaction_date": None,
                "currency": "IDR",
                "total_amount": 356900,
            },
            "patient_info": {
                "name": "Nick Frederic Kristian",
                "date_of_birth": None,
                "age": 13,
                "gender": "Pria",
                "address": None,
            },
            "medical_context": {
                "anamnesis": "panas, batuk, pilek, tenggorokan",
                "diagnoses": [{"description": "pharyngitis + tonsillitis", "doctor_name": "dr. Makmuri MS, Sp.A(K)"}],
                "therapy_note": "Poliver, Isprinol, obat batuk pilek",
            },
            "medicines": [{"name": "Meiactin tab 500mg", "quantity": 1, "unit": None, "dosage_instruction": None, "unit_price": None, "total_price": None}],
            "other_charges": [
                {"category": "Consultation", "description": "Biaya dokter Klinik Utama Makmuri", "amount": 278000},
                {"category": "Medicine", "description": "Obat-obatan Apotek Pojodo", "amount": 78900},
            ],
            "financial_recap": {"total_medicines": 78900, "total_other_charges": 278000, "grand_total": 356900},
        },
        {
            "summary": {
                "detected_documents": ["Formulir Rawat Jalan", "Kwitansi Klinik Utama Makmuri", "Salinan Resep Apotek Pojodo", "Kwitansi Apotek Pojodo"],
                "medical_facility_name": "Klinik Utama Makmuri",
                "service_date": "28/08/2025",
                "transaction_date": None,
                "currency": "IDR",
                "total_amount": 356900,
            },
            "patient_info": {
                "name": "Nick Frederic Kristian",
                "date_of_birth": None,
                "age": 13,
                "gender": "Pria",
                "address": None,
            },
            "medical_context": {
                "anamnesis": "panas, batuk, pilek, tenggorokan",
                "diagnoses": [{"description": "pharyngitis + asma", "doctor_name": "dr. Makmuri MS, Sp.A(K)"}],
                "therapy_note": "Poline, sirup batuk pilek, diet",
            },
            "medicines": [{"name": "Meiactin tab 500 mg", "quantity": 1, "unit": None, "dosage_instruction": None, "unit_price": None, "total_price": 78900}],
            "other_charges": [{"category": "Consultation", "description": "Biaya konsultasi dokter di Klinik Utama Makmuri", "amount": 278000}],
            "financial_recap": {"total_medicines": 78900, "total_other_charges": 278000, "grand_total": 356900},
        },
    ]
    ollama_outputs = [
        {
            "summary": {
                "detected_documents": ["Formulir Rawat Jalan", "Kwitansi", "Pharmacy Receipt"],
                "medical_facility_name": "Klinik Makmuri",
                "service_date": "28/08/2025",
                "transaction_date": None,
                "currency": "IDR",
                "total_amount": 356900,
            },
            "patient_info": {
                "name": "Nick Frederic Kristian",
                "date_of_birth": None,
                "age": 13,
                "gender": "Pria",
                "address": None,
            },
            "medical_context": {
                "anamnesis": "fanos, batuk pilek lama",
                "diagnoses": [{"description": "pharyngitis + asthma", "doctor_name": "dr. Makmuri MS, Sp.A(K)"}],
                "therapy_note": "Policare, Zithromax",
            },
            "medicines": [],
            "other_charges": [
                {"category": "Medical Consultation/Medicine", "description": "Kwitansi No. 2225/25", "amount": 278000},
                {"category": "Medicine", "description": "Apotek Bogor", "amount": 78900},
            ],
            "financial_recap": {"total_medicines": None, "total_other_charges": 356900, "grand_total": 356900},
        },
        {
            "summary": {
                "detected_documents": ["Formulir Rawat Jalan", "Kwitansi", "Pharmacy Receipt"],
                "medical_facility_name": "Klinik Utama Makmuri",
                "service_date": "28/08/2025",
                "transaction_date": None,
                "currency": "IDR",
                "total_amount": 358900,
            },
            "patient_info": {
                "name": "Nick Frederic Kristian",
                "date_of_birth": None,
                "age": 13,
                "gender": "Pria",
                "address": None,
            },
            "medical_context": {
                "anamnesis": "fanos, batuk pilek lama",
                "diagnoses": [{"description": "pharyngitis + asthma", "doctor_name": "dr. Makmuri MS, Sp.A(K)"}],
                "therapy_note": "Polirese, Zyrtec",
            },
            "medicines": [{"name": "Obat-obatan", "quantity": None, "unit": None, "dosage_instruction": None, "unit_price": None, "total_price": 78900}],
            "other_charges": [{"category": "Consultation/Treatment", "description": "Resep", "amount": 279000}],
            "financial_recap": {"total_medicines": 78900, "total_other_charges": 279000, "grand_total": 357900},
        },
    ]
    call_counts = {"model-google": 0, "model-ollama": 0}

    class FakeModel:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        async def ainvoke(self, messages: list[object]) -> AIMessage:
            del messages
            if self.model_name == "model-openrouter":
                raise HTTPException(status_code=502, detail="Upstream model failure.")

            outputs = google_outputs if self.model_name == "model-google" else ollama_outputs
            call_index = call_counts[self.model_name]
            call_counts[self.model_name] += 1
            return AIMessage(content=json.dumps(outputs[call_index]))

    async def conservative_judge(
        field_path: str,
        images: list[object],
        candidates: list[dict[str, object]],
    ) -> tuple[str | None, str | None]:
        del images
        return cast(str, candidates[0]["normalized_key"]), "Preserve the leading candidate."

    monkeypatch.setattr(
        medical_processor,
        "_create_chat_model",
        lambda model_spec: FakeModel(model_spec.model),
    )
    monkeypatch.setattr(
        medical_processor,
        "_get_enabled_model_specs",
        lambda: [
            AIModelSpec(provider="google_genai", model="model-google"),
            AIModelSpec(provider="openrouter", model="model-openrouter"),
            AIModelSpec(provider="ollama", model="model-ollama"),
        ],
    )
    monkeypatch.setattr(medical_processor, "_judge_field_conflict", conservative_judge)

    result = asyncio.run(
        medical_processor.process(
            [
                DocumentImageInput(
                    filename="scan.png",
                    mime_type="image/png",
                    base64_data="YWJj",
                )
            ]
        )
    )

    assert result.result.summary.total_amount == 356900
    assert result.result.financial_recap.total_medicines == 78900
    assert result.result.financial_recap.total_other_charges == 278000
    assert result.result.financial_recap.grand_total == 356900
    assert len(result.result.medical_context.diagnoses) == 1
    assert result.result.medical_context.diagnoses[0].description in {"pharyngitis + asma", "pharyngitis + asthma"}
    assert len(result.result.medicines) == 1
    assert result.result.medicines[0].total_price == 78900
    assert len(result.result.other_charges) == 1
    assert result.result.other_charges[0].amount == 278000
    assert result.field_confidence["medical_context.anamnesis"]["judge_used"] is True
    assert "judge_resolved" in result.field_confidence["medical_context.anamnesis"]["review_flags"]
    assert result.extraction_meta_payload["constraint_checks"]["financial_recap.grand_total"]["used_derived_value"] is False
    assert "financial_recap.grand_total did not match derived financial totals." not in result.review_reasons
    assert result.extraction_meta_payload["reconciliation_version"] == "financial-reconciliation-v2"
    assert "list_group_meta" in result.extraction_meta_payload
    assert "field_source_summary" not in result.extraction_meta_payload
    selected_total_medicines_candidate = next(
        candidate
        for candidate in result.field_confidence["financial_recap.total_medicines"]["candidates"]
        if candidate["selected"]
    )
    assert selected_total_medicines_candidate["value"] == 78900


def test_process_flags_line_item_total_medicines_mismatch_against_row_sum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    medical_processor = MedicalDocumentProcessor(settings=build_settings())

    class FakeModel:
        async def ainvoke(self, messages: list[object]) -> AIMessage:
            del messages
            payload = {
                **SCAN_RESULT,
                "medicines": [
                    {
                        **SCAN_RESULT["medicines"][0],
                        "total_price": 12000,
                    }
                ],
                "financial_recap": {
                    **SCAN_RESULT["financial_recap"],
                    "total_medicines": 10000,
                    "grand_total": 125000,
                },
            }
            return AIMessage(content=json.dumps(payload))

    monkeypatch.setattr(
        medical_processor,
        "_create_chat_model",
        lambda model_spec: FakeModel(),
    )
    monkeypatch.setattr(
        medical_processor,
        "_get_enabled_model_specs",
        lambda: [AIModelSpec(provider="google_genai", model="model-a")],
    )

    result = asyncio.run(
        medical_processor.process(
            [
                DocumentImageInput(
                    filename="scan.png",
                    mime_type="image/png",
                    base64_data="YWJj",
                )
            ]
        )
    )

    line_item_check = result.extraction_meta_payload["constraint_checks"]["line_item.total_medicines"]
    assert line_item_check["derived_value"] == 12000
    assert line_item_check["original_value"] == 10000
    assert line_item_check["final_value"] == 10000
    assert line_item_check["passed"] is False
    assert line_item_check["used_derived_value"] is False
    assert "line_item.total_medicines did not match derived financial totals." in result.review_reasons


def test_process_marks_backfilled_financial_totals_as_review_worthy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    medical_processor = MedicalDocumentProcessor(settings=build_settings())

    class FakeModel:
        async def ainvoke(self, messages: list[object]) -> AIMessage:
            del messages
            payload = {
                **SCAN_RESULT,
                "medicines": [
                    {
                        **SCAN_RESULT["medicines"][0],
                        "unit_price": None,
                        "total_price": None,
                    }
                ],
                "financial_recap": {
                    **SCAN_RESULT["financial_recap"],
                    "grand_total": None,
                },
            }
            return AIMessage(content=json.dumps(payload))

    monkeypatch.setattr(
        medical_processor,
        "_create_chat_model",
        lambda model_spec: FakeModel(),
    )
    monkeypatch.setattr(
        medical_processor,
        "_get_enabled_model_specs",
        lambda: [AIModelSpec(provider="google_genai", model="model-a")],
    )

    result = asyncio.run(
        medical_processor.process(
            [
                DocumentImageInput(
                    filename="scan.png",
                    mime_type="image/png",
                    base64_data="YWJj",
                )
            ]
        )
    )

    grand_total_check = result.extraction_meta_payload["constraint_checks"]["financial_recap.grand_total"]
    assert grand_total_check["used_derived_value"] is False
    assert result.result.financial_recap.grand_total is not None
    assert result.result.summary.total_amount is not None
    assert result.result.financial_recap.grand_total == result.result.summary.total_amount
    assert result.review_required is True
    assert len(result.review_reasons) > 0


def test_process_prefers_coherent_packet_ledger_over_hallucinated_medicine_prices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    medical_processor = MedicalDocumentProcessor(settings=build_settings())
    outputs = {
        "model-lite": [
            {
                **SCAN_RESULT,
                "summary": {
                    "detected_documents": ["DR MAHE receipt", "Apotek Podojodo receipt", "Salinan Resep", "Formulir Rawat Jalan"],
                    "medical_facility_name": "DR MAHE",
                    "service_date": "2025-08-31",
                    "transaction_date": None,
                    "currency": "IDR",
                    "total_amount": 595000,
                },
                "patient_info": {
                    "name": "Nathan Benedict Kristian",
                    "date_of_birth": None,
                    "age": 6,
                    "gender": "Pria",
                    "address": None,
                },
                "medical_context": {
                    "anamnesis": "Batuk ngilu demam hags",
                    "diagnoses": [{"description": "perhesis", "doctor_name": "dr. Mahendra, Sp.A(K), Ph.D."}],
                    "therapy_note": "fibramax, puyer bapd, prover",
                },
                "medicines": [
                    {"name": "Micropect", "quantity": None, "unit": None, "dosage_instruction": "3 ddl pnr", "unit_price": None, "total_price": None},
                    {"name": "Lasal", "quantity": None, "unit": None, "dosage_instruction": None, "unit_price": None, "total_price": None},
                    {"name": "Tricore", "quantity": None, "unit": None, "dosage_instruction": None, "unit_price": None, "total_price": None},
                    {"name": "Equal", "quantity": None, "unit": None, "dosage_instruction": None, "unit_price": None, "total_price": None},
                ],
                "other_charges": [
                    {"category": "Medical Consultation", "description": "Pemeriksaan & Pengobatan Anak", "amount": 195000},
                    {"category": "Pharmacy", "description": "Apotek Podojodo", "amount": 102900},
                    {"category": "Medical Consultation", "description": "jasa dokter", "amount": 300000},
                ],
                "financial_recap": {"total_medicines": 0, "total_other_charges": 597900, "grand_total": 595000},
            },
            {
                **SCAN_RESULT,
                "summary": {
                    "detected_documents": ["DR MAHE receipt", "Apotek Podojodo receipt", "Salinan Resep", "Formulir Rawat Jalan"],
                    "medical_facility_name": "DR MAHE",
                    "service_date": "2025-08-31",
                    "transaction_date": None,
                    "currency": "IDR",
                    "total_amount": 687000,
                },
                "patient_info": {
                    "name": "Nathan Benedict Kristian",
                    "date_of_birth": None,
                    "age": 6,
                    "gender": "Pria",
                    "address": None,
                },
                "medical_context": {
                    "anamnesis": "Batuk ngilu demam hags",
                    "diagnoses": [{"description": "perhesis", "doctor_name": "dr. Mahendra, Sp.A(K), Ph.D."}],
                    "therapy_note": "Ibramax, puyer bapd, prover",
                },
                "medicines": [
                    {"name": "Micropect", "quantity": None, "unit": None, "dosage_instruction": "3 ddl pmv", "unit_price": None, "total_price": None},
                    {"name": "Lasal", "quantity": None, "unit": None, "dosage_instruction": None, "unit_price": None, "total_price": None},
                    {"name": "Tricone", "quantity": None, "unit": None, "dosage_instruction": None, "unit_price": None, "total_price": None},
                    {"name": "Equal", "quantity": None, "unit": None, "dosage_instruction": None, "unit_price": None, "total_price": None},
                ],
                "other_charges": [
                    {"category": "Medical Consultation", "description": "Pemeriksaan & Pengobatan Anak", "amount": 195000},
                    {"category": "Pharmacy", "description": "Obat-obatan", "amount": 102000},
                    {"category": "Medical Consultation", "description": "jasa dokter", "amount": 390000},
                ],
                "financial_recap": {"total_medicines": 102000, "total_other_charges": 585000, "grand_total": 687000},
            },
        ],
        "model-flash": [
            {
                **SCAN_RESULT,
                "summary": {
                    "detected_documents": ["Kwitansi DR MAHE", "Struk dr. Mahendra", "Kwitansi Apotek Podojodo", "Salinan Resep", "Formulir Rawat Jalan"],
                    "medical_facility_name": "dr. Mahendra Tri Arif S., Sp.A(K)., Ph.D.",
                    "service_date": "2025-08-31",
                    "transaction_date": None,
                    "currency": "IDR",
                    "total_amount": 597900,
                },
                "patient_info": {
                    "name": "Nathan Benedict Kristian",
                    "date_of_birth": None,
                    "age": 6,
                    "gender": "Pria",
                    "address": None,
                },
                "medical_context": {
                    "anamnesis": "Batuk ngikil demam hng",
                    "diagnoses": [{"description": "Pertusis", "doctor_name": "dr. Mahendra Tri Arif S., Sp.A(K)., Ph.D."}],
                    "therapy_note": "Zibramax, puyer batuk, profen",
                },
                "medicines": [
                    {"name": "Zibramax", "quantity": 1, "unit": None, "dosage_instruction": None, "unit_price": 160000, "total_price": 160000},
                    {"name": "Profen Suspens", "quantity": 1, "unit": None, "dosage_instruction": None, "unit_price": 35000, "total_price": 35000},
                    {"name": "Mucospect", "quantity": None, "unit": "mg", "dosage_instruction": "3 dd pulv", "unit_price": None, "total_price": None},
                    {"name": "Lasal", "quantity": None, "unit": "mg", "dosage_instruction": "3 dd pulv", "unit_price": None, "total_price": None},
                    {"name": "Triam", "quantity": None, "unit": "mg", "dosage_instruction": "3 dd pulv", "unit_price": None, "total_price": None},
                    {"name": "Equal", "quantity": None, "unit": "sachet", "dosage_instruction": "3 dd pulv", "unit_price": None, "total_price": None},
                    {"name": "Racikan Puyer (Mucospect, Lasal, Triam, Equal)", "quantity": 15, "unit": "pulv", "dosage_instruction": "3 dd pulv", "unit_price": None, "total_price": 102900},
                ],
                "other_charges": [
                    {"category": "Consultation", "description": "Pemeriksaan / Jasa Dokter", "amount": 300000},
                ],
                "financial_recap": {"total_medicines": 297900, "total_other_charges": 300000, "grand_total": 597900},
            },
            {
                **SCAN_RESULT,
                "summary": {
                    "detected_documents": ["Kwitansi DR MAHE", "Struk dr. Mahendra", "Kwitansi Apotek Podojodo", "Salinan Resep", "Formulir Rawat Jalan"],
                    "medical_facility_name": "dr. Mahendra Tri Arif S., Sp.A(K)., Ph.D.",
                    "service_date": "2025-08-31",
                    "transaction_date": None,
                    "currency": "IDR",
                    "total_amount": 597900,
                },
                "patient_info": {
                    "name": "Nathan Benedict Kristian",
                    "date_of_birth": None,
                    "age": 6,
                    "gender": "Pria",
                    "address": None,
                },
                "medical_context": {
                    "anamnesis": "Batuk ngikil demam hngs",
                    "diagnoses": [{"description": "Pertusis", "doctor_name": "dr. Mahendra Tri Arif S., Sp.A(K)., Ph.D."}],
                    "therapy_note": "Zibramax, puyer batuk, prover",
                },
                "medicines": [
                    {"name": "Zibramax", "quantity": 1, "unit": "x", "dosage_instruction": None, "unit_price": 160000, "total_price": 160000},
                    {"name": "Profen (scusens)", "quantity": 1, "unit": "x", "dosage_instruction": None, "unit_price": 35000, "total_price": 35000},
                    {"name": "Mucospect 7.5mg", "quantity": 15, "unit": "pulv", "dosage_instruction": "3 dd p1v", "unit_price": None, "total_price": None},
                    {"name": "Lasal 1.25mg", "quantity": 15, "unit": "pulv", "dosage_instruction": "3 dd p1v", "unit_price": None, "total_price": None},
                    {"name": "Trilac 3mg", "quantity": 15, "unit": "pulv", "dosage_instruction": "3 dd p1v", "unit_price": None, "total_price": None},
                    {"name": "Equal 1/3 sachet", "quantity": 15, "unit": "pulv", "dosage_instruction": "3 dd p1v", "unit_price": None, "total_price": None},
                    {"name": "Obat Racikan (Puyer)", "quantity": 1, "unit": "R/", "dosage_instruction": None, "unit_price": 102900, "total_price": 102900},
                ],
                "other_charges": [
                    {"category": "Consultation", "description": "Pemeriksaan / Jasa Dokter", "amount": 300000},
                ],
                "financial_recap": {"total_medicines": 297900, "total_other_charges": 300000, "grand_total": 597900},
            },
        ],
        "model-ollama": [
            {
                **SCAN_RESULT,
                "summary": {
                    "detected_documents": ["receipt", "prescription", "medical_form"],
                    "medical_facility_name": "dr. Mahendra Tri Arif S., Sp.A(K), Ph.D.",
                    "service_date": "2025-08-01",
                    "transaction_date": None,
                    "currency": "IDR",
                    "total_amount": 1145000,
                },
                "patient_info": {
                    "name": "Nathan Benedict Kristian",
                    "date_of_birth": None,
                    "age": 6,
                    "gender": None,
                    "address": None,
                },
                "medical_context": {
                    "anamnesis": "Batuk ngilili demam hanya",
                    "diagnoses": [{"description": "perburuan", "doctor_name": "dr. Mahendra Tri Arif S., Sp.A(K), Ph.D."}],
                    "therapy_note": "Abramax puyer bapd, prover",
                },
                "medicines": [
                    {"name": "Microcet", "quantity": None, "unit": None, "dosage_instruction": "1,25mg", "unit_price": None, "total_price": None},
                    {"name": "Tritone", "quantity": None, "unit": None, "dosage_instruction": "3mg", "unit_price": None, "total_price": None},
                ],
                "other_charges": [
                    {"category": "Konsultasi", "description": "Pemeriksaan & Pengobatan Anak", "amount": 100000},
                    {"category": "Konsultasi", "description": "Jasa dokter", "amount": 900000},
                    {"category": "Obat", "description": "Apotek Podojodo", "amount": 145000},
                ],
                "financial_recap": {"total_medicines": None, "total_other_charges": 1145000, "grand_total": 1145000},
            },
            {
                **SCAN_RESULT,
                "summary": {
                    "detected_documents": ["receipt", "prescription", "medical_form"],
                    "medical_facility_name": "dr. Mahendra Tri Arif S., Sp.A(K), Ph.D.",
                    "service_date": "2025-08-01",
                    "transaction_date": None,
                    "currency": "IDR",
                    "total_amount": 1145000,
                },
                "patient_info": {
                    "name": "Nathan Benedict Kristian",
                    "date_of_birth": None,
                    "age": 6,
                    "gender": None,
                    "address": None,
                },
                "medical_context": {
                    "anamnesis": "Batuk ngilili demam hanya",
                    "diagnoses": [{"description": "perburuan", "doctor_name": "dr. Mahendra Tri Arif S., Sp.A(K), Ph.D."}],
                    "therapy_note": "Abramax puyer bap d. prover",
                },
                "medicines": [
                    {"name": "Microcet", "quantity": None, "unit": None, "dosage_instruction": "1,25mg", "unit_price": None, "total_price": None},
                    {"name": "Tritone", "quantity": None, "unit": None, "dosage_instruction": "3mg", "unit_price": None, "total_price": None},
                ],
                "other_charges": [
                    {"category": "Consultation", "description": "Pemeriksaan & Pengobatan Anak", "amount": 100000},
                    {"category": "Consultation", "description": "Jasa dokter", "amount": 900000},
                    {"category": "Consultation", "description": "Pemeriksaan & Pengobatan Anak", "amount": 145000},
                    {"category": "Pharmacy", "description": "Apotek Podojodo", "amount": 100000},
                ],
                "financial_recap": {"total_medicines": 0, "total_other_charges": 1145000, "grand_total": 1145000},
            },
        ],
    }
    call_counts = {model_name: 0 for model_name in outputs}

    class FakeModel:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        async def ainvoke(self, messages: list[object]) -> AIMessage:
            del messages
            call_index = call_counts[self.model_name]
            call_counts[self.model_name] += 1
            return AIMessage(content=json.dumps(outputs[self.model_name][call_index]))

    monkeypatch.setattr(
        medical_processor,
        "_create_chat_model",
        lambda model_spec: FakeModel(model_spec.model),
    )
    monkeypatch.setattr(
        medical_processor,
        "_get_enabled_model_specs",
        lambda: [
            AIModelSpec(provider="google_genai", model="model-lite"),
            AIModelSpec(provider="google_genai", model="model-flash"),
            AIModelSpec(provider="ollama", model="model-ollama"),
        ],
    )
    monkeypatch.setattr(
        medical_processor,
        "_judge_field_conflict",
        lambda field_path, images, candidates: asyncio.sleep(0, result=(cast(str, candidates[0]["normalized_key"]), "Preserve leading candidate.")),
    )

    result = asyncio.run(
        medical_processor.process(
            [
                DocumentImageInput(
                    filename="scan.png",
                    mime_type="image/png",
                    base64_data="YWJj",
                )
            ],
            validation_hints={
                "ordered_documents": [
                    {"image_index": 0, "document_index": 0, "document_type": "receipt", "decision": "valid"},
                    {"image_index": 0, "document_index": 1, "document_type": "receipt", "decision": "valid"},
                    {"image_index": 0, "document_index": 2, "document_type": "receipt", "decision": "valid"},
                    {"image_index": 0, "document_index": 3, "document_type": "prescription", "decision": "valid"},
                    {"image_index": 1, "document_index": 0, "document_type": "medical_form", "decision": "valid"},
                    {"image_index": 2, "document_index": 0, "document_type": "receipt", "decision": "valid"},
                ]
            },
        )
    )

    assert result.result.summary.total_amount == 597900
    assert result.result.financial_recap.total_medicines == 102900
    assert result.result.financial_recap.total_other_charges == 495000
    assert result.result.financial_recap.grand_total == 597900
    assert len(result.result.other_charges) == 2
    assert sorted(charge.amount for charge in result.result.other_charges if charge.amount is not None) == [195000, 300000]

    ledger_source = result.extraction_meta_payload["derived_financials"]["canonical_medicine_ledger_source"]
    assert ledger_source["amount"] == 102900
    assert ledger_source["source_type"] == "other_charge"
    assert any(
        entry["field_path"].endswith(".total_price") and entry["amount"] in {160000, 35000} and entry["counted"] is False
        for entry in result.extraction_meta_payload["derived_financials"]["price_counted_in_ledger"]
    )
    assert any(
        suppression["amount"] == 900000
        for suppression in result.extraction_meta_payload["derived_financials"]["financial_outlier_suppressions"]
    )


def test_process_preserves_strong_consensus_when_coherent_reconciliation_is_weaker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    medical_processor = MedicalDocumentProcessor(settings=build_settings())
    outputs = {
        "model-primary": [
            {
                **SCAN_RESULT,
                "summary": {
                    "detected_documents": ["Kwitansi DR MAHE", "Kwitansi Apotek Podojodo", "Salinan Resep", "Formulir Rawat Jalan"],
                    "medical_facility_name": "dr. Mahendra Tri Arif S., Sp.A(K)., Ph.D.",
                    "service_date": "2025-08-31",
                    "transaction_date": None,
                    "currency": "IDR",
                    "total_amount": 597900,
                },
                "patient_info": {
                    "name": "Nathan Benedict Kristian",
                    "date_of_birth": None,
                    "age": 6,
                    "gender": "Pria",
                    "address": None,
                },
                "medical_context": {
                    "anamnesis": "Batuk ngikil demam hng",
                    "diagnoses": [{"description": "Pertusis", "doctor_name": "dr. Mahendra Tri Arif S., Sp.A(K)., Ph.D."}],
                    "therapy_note": "Zibramax, puyer batuk, profen",
                },
                "medicines": [
                    {"name": "Zibramax", "quantity": 1, "unit": None, "dosage_instruction": None, "unit_price": 160000, "total_price": 160000},
                    {"name": "Profen", "quantity": 1, "unit": None, "dosage_instruction": None, "unit_price": 35000, "total_price": 35000},
                    {"name": "Lasal", "quantity": None, "unit": "mg", "dosage_instruction": "3 dd pulv", "unit_price": None, "total_price": None},
                ],
                "other_charges": [
                    {"category": "Consultation", "description": "Pemeriksaan / Jasa Dokter", "amount": 300000},
                ],
                "financial_recap": {"total_medicines": 297900, "total_other_charges": 300000, "grand_total": 597900},
            },
            {
                **SCAN_RESULT,
                "summary": {
                    "detected_documents": ["Kwitansi DR MAHE", "Kwitansi Apotek Podojodo", "Salinan Resep", "Formulir Rawat Jalan"],
                    "medical_facility_name": "dr. Mahendra Tri Arif S., Sp.A(K)., Ph.D.",
                    "service_date": "2025-08-31",
                    "transaction_date": None,
                    "currency": "IDR",
                    "total_amount": 597900,
                },
                "patient_info": {
                    "name": "Nathan Benedict Kristian",
                    "date_of_birth": None,
                    "age": 6,
                    "gender": "Pria",
                    "address": None,
                },
                "medical_context": {
                    "anamnesis": "Batuk ngikil demam hngs",
                    "diagnoses": [{"description": "Pertusis", "doctor_name": "dr. Mahendra Tri Arif S., Sp.A(K)., Ph.D."}],
                    "therapy_note": "Zibramax, puyer batuk, profen",
                },
                "medicines": [
                    {"name": "Zibramax", "quantity": 1, "unit": None, "dosage_instruction": None, "unit_price": 160000, "total_price": 160000},
                    {"name": "Profen", "quantity": 1, "unit": None, "dosage_instruction": None, "unit_price": 35000, "total_price": 35000},
                    {"name": "Lasal", "quantity": None, "unit": "mg", "dosage_instruction": "3 dd pulv", "unit_price": None, "total_price": None},
                ],
                "other_charges": [
                    {"category": "Consultation", "description": "Pemeriksaan / Jasa Dokter", "amount": 300000},
                ],
                "financial_recap": {"total_medicines": 297900, "total_other_charges": 300000, "grand_total": 597900},
            },
        ],
        "model-weaker": [
            {
                **SCAN_RESULT,
                "summary": {
                    "detected_documents": ["receipt", "prescription", "medical_form"],
                    "medical_facility_name": "DR MAHE",
                    "service_date": "2025-08-01",
                    "transaction_date": None,
                    "currency": "IDR",
                    "total_amount": 1145000,
                },
                "patient_info": {
                    "name": "Nathan Benedict Kristian",
                    "date_of_birth": None,
                    "age": 6,
                    "gender": "Pria",
                    "address": None,
                },
                "medical_context": {
                    "anamnesis": "Batuk ngilili demam hanya",
                    "diagnoses": [{"description": "perburukan", "doctor_name": "dr. Mahendra Tri Arif S., Sp.A(K), Ph.D."}],
                    "therapy_note": "Abramax puyer bap d. prover",
                },
                "medicines": [
                    {"name": "Microcet", "quantity": None, "unit": None, "dosage_instruction": "1,25mg", "unit_price": None, "total_price": None},
                    {"name": "Tritone", "quantity": None, "unit": None, "dosage_instruction": "3mg", "unit_price": None, "total_price": None},
                ],
                "other_charges": [
                    {"category": "Consultation", "description": "Jasa dokter", "amount": 900000},
                    {"category": "Consultation", "description": "Pemeriksaan & Pengobatan Anak", "amount": 145000},
                    {"category": "Pharmacy", "description": "Obat", "amount": 100000},
                    {"category": "Medical Service", "description": "Dr. Mahendra Tri Arif, Sp.A (K), PhD", "amount": 100000},
                    {"category": "Jasa Medis", "description": "Jasa Medis", "amount": 150000},
                ],
                "financial_recap": {"total_medicines": None, "total_other_charges": 1145000, "grand_total": 1145000},
            },
            {
                **SCAN_RESULT,
                "summary": {
                    "detected_documents": ["receipt", "prescription", "medical_form"],
                    "medical_facility_name": "DR MAHE",
                    "service_date": "2025-08-01",
                    "transaction_date": None,
                    "currency": "IDR",
                    "total_amount": 1145000,
                },
                "patient_info": {
                    "name": "Nathan Benedict Kristian",
                    "date_of_birth": None,
                    "age": 6,
                    "gender": None,
                    "address": None,
                },
                "medical_context": {
                    "anamnesis": "Batuk ngilili demam hanya",
                    "diagnoses": [{"description": "perburukan", "doctor_name": "dr. Mahendra Tri Arif S., Sp.A(K), Ph.D."}],
                    "therapy_note": "Abramax puyer bap d. prover",
                },
                "medicines": [
                    {"name": "Microcet", "quantity": None, "unit": None, "dosage_instruction": "1,25mg", "unit_price": None, "total_price": None},
                    {"name": "Tritone", "quantity": None, "unit": None, "dosage_instruction": "3mg", "unit_price": None, "total_price": None},
                ],
                "other_charges": [
                    {"category": "Consultation", "description": "Jasa dokter", "amount": 900000},
                    {"category": "Consultation", "description": "Pemeriksaan & Pengobatan Anak", "amount": 145000},
                    {"category": "Pharmacy", "description": "Obat", "amount": 100000},
                    {"category": "Medical Service", "description": "Dr. Mahendra Tri Arif, Sp.A (K), PhD", "amount": 100000},
                    {"category": "Jasa Medis", "description": "Jasa Medis", "amount": 150000},
                ],
                "financial_recap": {"total_medicines": 0, "total_other_charges": 1145000, "grand_total": 1145000},
            },
        ],
    }
    call_counts = {model_name: 0 for model_name in outputs}

    class FakeModel:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        async def ainvoke(self, messages: list[object]) -> AIMessage:
            del messages
            call_index = call_counts[self.model_name]
            call_counts[self.model_name] += 1
            return AIMessage(content=json.dumps(outputs[self.model_name][call_index]))

    monkeypatch.setattr(
        medical_processor,
        "_create_chat_model",
        lambda model_spec: FakeModel(model_spec.model),
    )
    monkeypatch.setattr(
        medical_processor,
        "_get_enabled_model_specs",
        lambda: [
            AIModelSpec(provider="google_genai", model="model-primary"),
            AIModelSpec(provider="ollama", model="model-weaker"),
        ],
    )
    monkeypatch.setattr(
        medical_processor,
        "_judge_field_conflict",
        lambda field_path, images, candidates: asyncio.sleep(0, result=(cast(str, candidates[0]["normalized_key"]), "Preserve leading candidate.")),
    )

    result = asyncio.run(
        medical_processor.process(
            [
                DocumentImageInput(
                    filename="scan.png",
                    mime_type="image/png",
                    base64_data="YWJj",
                )
            ],
            validation_hints={
                "ordered_documents": [
                    {"image_index": 0, "document_index": 0, "document_type": "receipt", "decision": "valid"},
                    {"image_index": 0, "document_index": 1, "document_type": "receipt", "decision": "valid"},
                    {"image_index": 0, "document_index": 2, "document_type": "receipt", "decision": "valid"},
                    {"image_index": 0, "document_index": 3, "document_type": "prescription", "decision": "valid"},
                    {"image_index": 1, "document_index": 0, "document_type": "medical_form", "decision": "valid"},
                ]
            },
        )
    )

    assert result.result.summary.total_amount == 597900
    assert result.result.financial_recap.total_medicines == 297900
    assert result.result.financial_recap.total_other_charges == 300000
    assert result.result.financial_recap.grand_total == 597900
    assert any(
        "materially weaker than the existing consensus totals" in reason
        for reason in result.review_reasons
    )
    assert not any(
        corrected.get("field_path") == "summary.total_amount" and corrected.get("corrected_value") == 1145000
        for corrected in result.extraction_meta_payload["corrected_fields"]
    )



def test_process_uses_judge_for_low_confidence_dosage_conflict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    medical_processor = MedicalDocumentProcessor(settings=build_settings())

    class FakeModel:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        async def ainvoke(self, messages: list[object]) -> AIMessage:
            del messages
            payload = {
                **SCAN_RESULT,
                "medicines": [
                    {
                        "name": "Mucopret",
                        "quantity": None,
                        "unit": None,
                        "dosage_instruction": "3 dd pulv" if self.model_name == "model-a" else "3 ddl pnv",
                        "unit_price": None,
                        "total_price": None,
                    }
                ],
            }
            return AIMessage(content=json.dumps(payload))

    async def dosage_judge(
        field_path: str,
        images: list[object],
        candidates: list[dict[str, object]],
    ) -> tuple[str | None, str | None]:
        del images
        assert field_path == "medicines.dosage_instruction"
        selected = next(candidate for candidate in candidates if candidate["raw_value"] == "3 dd pulv")
        return cast(str, selected["normalized_key"]), "Prescription dosage is clearer in the primary image."

    monkeypatch.setattr(
        medical_processor,
        "_create_chat_model",
        lambda model_spec: FakeModel(model_spec.model),
    )
    monkeypatch.setattr(
        medical_processor,
        "_get_enabled_model_specs",
        lambda: [
            AIModelSpec(provider="google_genai", model="model-a"),
            AIModelSpec(provider="google_genai", model="model-b"),
        ],
    )
    monkeypatch.setattr(medical_processor, "_judge_field_conflict", dosage_judge)

    result = asyncio.run(
        medical_processor.process(
            [
                DocumentImageInput(
                    filename="scan.png",
                    mime_type="image/png",
                    base64_data="YWJj",
                )
            ]
        )
    )

    dosage_field_key = next(key for key in result.field_confidence if key.endswith(".dosage_instruction"))
    assert result.field_confidence[dosage_field_key]["judge_used"] is True
    assert result.result.medicines[0].dosage_instruction == "3 dd pulv"


def test_process_preserves_primary_supported_packet_document_when_validation_hints_confirm_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    medical_processor = MedicalDocumentProcessor(settings=build_settings())
    payloads = [
        {
            **SCAN_RESULT,
            "summary": {
                **SCAN_RESULT["summary"],
                "detected_documents": [
                    "Kwitansi DR MAHE",
                    "Struk Apotek Podojodo",
                    "Salinan Resep",
                    "Formulir Rawat Jalan Mandiri Inhealth",
                    "Kwitansi DR MAHE (Jasa Dokter)",
                ],
            },
        },
        {
            **SCAN_RESULT,
            "summary": {
                **SCAN_RESULT["summary"],
                "detected_documents": [
                    "Kwitansi DR MAHE",
                    "Salinan Resep",
                    "Formulir Rawat Jalan Mandiri Inhealth",
                    "Kwitansi DR MAHE (Jasa Dokter)",
                ],
            },
        },
    ]

    class FakeModel:
        async def ainvoke(self, messages: list[object]) -> AIMessage:
            del messages
            return AIMessage(content=json.dumps(payloads.pop(0)))

    async def inventory_judge(
        field_path: str,
        images: list[object],
        candidates: list[dict[str, object]],
    ) -> tuple[str | None, str | None]:
        del images
        assert field_path == "summary.detected_documents"
        selected = next(candidate for candidate in candidates if candidate["normalized_key"] == "expanded")
        return cast(str, selected["normalized_key"]), "Validation-backed packet inventory preserves the extra receipt."

    monkeypatch.setattr(
        medical_processor,
        "_create_chat_model",
        lambda model_spec: FakeModel(),
    )
    monkeypatch.setattr(
        medical_processor,
        "_get_enabled_model_specs",
        lambda: [AIModelSpec(provider="google_genai", model="model-a")],
    )
    monkeypatch.setattr(medical_processor, "_judge_field_conflict", inventory_judge)

    validation_hints = {
        "ordered_documents": [
            {"image_index": 0, "document_index": 0, "document_type": "receipt", "decision": "valid"},
            {"image_index": 0, "document_index": 1, "document_type": "receipt", "decision": "valid"},
            {"image_index": 0, "document_index": 2, "document_type": "prescription", "decision": "valid"},
            {"image_index": 1, "document_index": 0, "document_type": "medical_form", "decision": "valid"},
            {"image_index": 2, "document_index": 0, "document_type": "receipt", "decision": "valid"},
        ],
        "document_type_counts": {"receipt": 3, "prescription": 1, "medical_form": 1},
        "image_document_counts": [],
        "total_detected_documents": 5,
    }

    result = asyncio.run(
        medical_processor.process(
            [
                DocumentImageInput(
                    filename="scan.png",
                    mime_type="image/png",
                    base64_data="YWJj",
                )
            ],
            validation_hints=validation_hints,
        )
    )

    assert "Struk Apotek Podojodo" in result.result.summary.detected_documents
    assert len(result.result.summary.detected_documents) == 5
    assert result.extraction_meta_payload["packet_inventory_hints"] == validation_hints


def test_process_preserves_sparse_primary_medicines_beyond_stable_group_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    medical_processor = MedicalDocumentProcessor(settings=build_settings())
    payloads = [
        {
            **SCAN_RESULT,
            "medicines": [
                {"name": "Mucopret", "quantity": None, "unit": None, "dosage_instruction": None, "unit_price": None, "total_price": None},
                {"name": "Equal", "quantity": None, "unit": None, "dosage_instruction": None, "unit_price": None, "total_price": None},
                {"name": "Lasal", "quantity": None, "unit": None, "dosage_instruction": None, "unit_price": None, "total_price": None},
                {"name": "Tritone", "quantity": None, "unit": None, "dosage_instruction": None, "unit_price": None, "total_price": None},
            ],
        },
        {
            **SCAN_RESULT,
            "medicines": [
                {"name": "Mucopret", "quantity": None, "unit": None, "dosage_instruction": None, "unit_price": None, "total_price": None},
            ],
        },
    ]

    class FakeModel:
        async def ainvoke(self, messages: list[object]) -> AIMessage:
            del messages
            return AIMessage(content=json.dumps(payloads.pop(0)))

    monkeypatch.setattr(
        medical_processor,
        "_create_chat_model",
        lambda model_spec: FakeModel(),
    )
    monkeypatch.setattr(
        medical_processor,
        "_get_enabled_model_specs",
        lambda: [AIModelSpec(provider="google_genai", model="model-a")],
    )

    result = asyncio.run(
        medical_processor.process(
            [
                DocumentImageInput(
                    filename="scan.png",
                    mime_type="image/png",
                    base64_data="YWJj",
                )
            ],
            validation_hints={
                "ordered_documents": [
                    {"image_index": 0, "document_index": 0, "document_type": "prescription", "decision": "valid"},
                    {"image_index": 0, "document_index": 1, "document_type": "receipt", "decision": "valid"},
                ]
            },
        )
    )

    assert [medicine.name for medicine in result.result.medicines] == ["Mucopret", "Equal", "Lasal", "Tritone"]
    assert result.extraction_meta_payload["list_group_meta"]["medicines"]["target_count"] == 4


def test_process_avoids_duplicate_receipt_padding_in_collage_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    medical_processor = MedicalDocumentProcessor(settings=build_settings())
    payloads = [
        {
            **SCAN_RESULT,
            "summary": {
                **SCAN_RESULT["summary"],
                "detected_documents": [
                    "dr. Mahendra Kwitansi",
                    "Apotek Podojodo Receipt",
                    "Apotek Podojodo Receipt",
                    "Apotek Podojodo Receipt",
                    "Salinan Resep",
                    "Formulir Rawat Jalan",
                ],
            },
        },
        {
            **SCAN_RESULT,
            "summary": {
                **SCAN_RESULT["summary"],
                "detected_documents": [
                    "dr. Mahendra Kwitansi",
                    "Apotek Podojodo Receipt",
                    "Salinan Resep",
                    "Mandiri Inhealth Formulir Rawat Jalan",
                ],
            },
        },
    ]

    class FakeModel:
        async def ainvoke(self, messages: list[object]) -> AIMessage:
            del messages
            return AIMessage(content=json.dumps(payloads.pop(0)))

    monkeypatch.setattr(
        medical_processor,
        "_create_chat_model",
        lambda model_spec: FakeModel(),
    )
    monkeypatch.setattr(
        medical_processor,
        "_get_enabled_model_specs",
        lambda: [AIModelSpec(provider="google_genai", model="model-a")],
    )

    result = asyncio.run(
        medical_processor.process(
            [
                DocumentImageInput(
                    filename="scan.png",
                    mime_type="image/png",
                    base64_data="YWJj",
                )
            ],
            validation_hints={
                "ordered_documents": [
                    {"image_index": 0, "document_index": 0, "document_type": "medical_form", "decision": "valid"},
                    {"image_index": 0, "document_index": 1, "document_type": "receipt", "decision": "valid"},
                    {"image_index": 0, "document_index": 2, "document_type": "receipt", "decision": "valid"},
                    {"image_index": 0, "document_index": 3, "document_type": "receipt", "decision": "valid"},
                    {"image_index": 1, "document_index": 0, "document_type": "prescription", "decision": "valid"},
                    {"image_index": 2, "document_index": 0, "document_type": "medical_form", "decision": "valid"},
                ]
            },
        )
    )

    assert result.result.summary.detected_documents.count("Apotek Podojodo Receipt") == 1
    assert "Salinan Resep" in result.result.summary.detected_documents
    assert "Formulir Rawat Jalan" in result.result.summary.detected_documents


def test_process_prefers_split_handwritten_ingredients_over_grouped_summary_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    medical_processor = MedicalDocumentProcessor(settings=build_settings())
    payloads = [
        {
            **SCAN_RESULT,
            "medicines": [
                {"name": "Mucopret", "quantity": None, "unit": None, "dosage_instruction": "3 dd pulv", "unit_price": None, "total_price": None},
                {"name": "Lasal", "quantity": None, "unit": None, "dosage_instruction": "3 dd pulv", "unit_price": None, "total_price": None},
                {"name": "Tritone", "quantity": None, "unit": None, "dosage_instruction": "3 dd pulv", "unit_price": None, "total_price": None},
                {"name": "Equal", "quantity": None, "unit": None, "dosage_instruction": "3 dd pulv", "unit_price": None, "total_price": None},
            ],
        },
        {
            **SCAN_RESULT,
            "medicines": [
                {"name": "Mucoprect, Lasal, Tritone, Equal", "quantity": 15, "unit": "pulv", "dosage_instruction": "3 dd pulv", "unit_price": None, "total_price": 102900},
                {"name": "Obat-obatan (Apotek Podojodo)", "quantity": None, "unit": None, "dosage_instruction": None, "unit_price": None, "total_price": 102900},
            ],
        },
    ]

    class FakeModel:
        async def ainvoke(self, messages: list[object]) -> AIMessage:
            del messages
            return AIMessage(content=json.dumps(payloads.pop(0)))

    monkeypatch.setattr(
        medical_processor,
        "_create_chat_model",
        lambda model_spec: FakeModel(),
    )
    monkeypatch.setattr(
        medical_processor,
        "_get_enabled_model_specs",
        lambda: [AIModelSpec(provider="google_genai", model="model-a")],
    )

    result = asyncio.run(
        medical_processor.process(
            [
                DocumentImageInput(
                    filename="scan.png",
                    mime_type="image/png",
                    base64_data="YWJj",
                )
            ],
            validation_hints={
                "ordered_documents": [
                    {"image_index": 0, "document_index": 0, "document_type": "prescription", "decision": "valid"},
                    {"image_index": 0, "document_index": 1, "document_type": "receipt", "decision": "valid"},
                ]
            },
        )
    )

    assert {medicine.name for medicine in result.result.medicines} == {"Mucopret", "Lasal", "Tritone", "Equal"}
    assert "preferred split ingredient extraction over grouped compound" in {
        group["excluded_reason"]
        for group in result.extraction_meta_payload["list_group_meta"]["medicines"]["groups"]
        if group["excluded_reason"] is not None
    }


def test_process_nullifies_ambiguous_handwritten_semantic_field_when_judge_cannot_resolve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    medical_processor = MedicalDocumentProcessor(settings=build_settings())
    payloads = [
        {
            **SCAN_RESULT,
            "medical_context": {
                **SCAN_RESULT["medical_context"],
                "anamnesis": "Batuk ngikil demam hags",
            },
        },
        {
            **SCAN_RESULT,
            "medical_context": {
                **SCAN_RESULT["medical_context"],
                "anamnesis": "Batuk ngiluh demam hari 3",
            },
        },
    ]

    class FakeModel:
        async def ainvoke(self, messages: list[object]) -> AIMessage:
            del messages
            return AIMessage(content=json.dumps(payloads.pop(0)))

    async def unresolved_judge(
        field_path: str,
        images: list[object],
        candidates: list[dict[str, object]],
    ) -> tuple[str | None, str | None]:
        del field_path, images, candidates
        return None, "Handwriting is too ambiguous."

    monkeypatch.setattr(
        medical_processor,
        "_create_chat_model",
        lambda model_spec: FakeModel(),
    )
    monkeypatch.setattr(
        medical_processor,
        "_get_enabled_model_specs",
        lambda: [AIModelSpec(provider="google_genai", model="model-a")],
    )
    monkeypatch.setattr(medical_processor, "_judge_field_conflict", unresolved_judge)

    result = asyncio.run(
        medical_processor.process(
            [
                DocumentImageInput(
                    filename="scan.png",
                    mime_type="image/png",
                    base64_data="YWJj",
                )
            ]
        )
    )

    assert result.result.medical_context.anamnesis is None
    anamnesis_meta = result.field_confidence["medical_context.anamnesis"]
    assert anamnesis_meta["confidence_penalty"] > 0
    assert "handwritten_uncertainty_penalty" in anamnesis_meta["review_flags"]
    assert any(
        "handwritten uncertainty confidence penalty" in hit
        for hit in anamnesis_meta["rule_hits"]
    )
    assert result.extraction_meta_payload["nullified_fields"] == [
        {
            "field_path": "medical_context.anamnesis",
            "original_value": "Batuk ngikil demam hags",
            "reason": "ambiguous handwritten field remained unresolved after adjudication",
        }
    ]


def test_process_does_not_apply_handwritten_penalty_for_strong_semantic_consensus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    medical_processor = MedicalDocumentProcessor(settings=build_settings())

    class FakeModel:
        async def ainvoke(self, messages: list[object]) -> AIMessage:
            del messages
            payload = {
                **SCAN_RESULT,
                "medical_context": {
                    **SCAN_RESULT["medical_context"],
                    "anamnesis": "Batuk ngikil demam",
                },
            }
            return AIMessage(content=json.dumps(payload))

    monkeypatch.setattr(
        medical_processor,
        "_create_chat_model",
        lambda model_spec: FakeModel(),
    )
    monkeypatch.setattr(
        medical_processor,
        "_get_enabled_model_specs",
        lambda: [
            AIModelSpec(provider="google_genai", model="model-a"),
            AIModelSpec(provider="google_genai", model="model-b"),
            AIModelSpec(provider="google_genai", model="model-c"),
        ],
    )

    result = asyncio.run(
        medical_processor.process(
            [
                DocumentImageInput(
                    filename="scan.png",
                    mime_type="image/png",
                    base64_data="YWJj",
                )
            ]
        )
    )

    anamnesis_meta = result.field_confidence["medical_context.anamnesis"]
    assert anamnesis_meta["confidence_penalty"] == 0.0
    assert "handwritten_uncertainty_penalty" not in anamnesis_meta["review_flags"]


def test_process_uses_judge_for_ambiguous_patient_name_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    medical_processor = MedicalDocumentProcessor(settings=build_settings())

    class FakeModel:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        async def ainvoke(self, messages: list[object]) -> AIMessage:
            del messages
            patient_name = "Nathan Bendecit Kristian" if self.model_name == "model-a" else "Nathan Benedict Kristian"
            payload = {
                **SCAN_RESULT,
                "patient_info": {
                    **SCAN_RESULT["patient_info"],
                    "name": patient_name,
                },
            }
            return AIMessage(content=json.dumps(payload))

    async def patient_name_judge(
        field_path: str,
        images: list[object],
        candidates: list[dict[str, object]],
    ) -> tuple[str | None, str | None]:
        del images
        assert field_path == "patient_info.name"
        selected = next(candidate for candidate in candidates if candidate["raw_value"] == "Nathan Benedict Kristian")
        return cast(str, selected["normalized_key"]), "Name spelling is clearer on the medical form."

    monkeypatch.setattr(
        medical_processor,
        "_create_chat_model",
        lambda model_spec: FakeModel(model_spec.model),
    )
    monkeypatch.setattr(
        medical_processor,
        "_get_enabled_model_specs",
        lambda: [
            AIModelSpec(provider="google_genai", model="model-a"),
            AIModelSpec(provider="google_genai", model="model-b"),
        ],
    )
    monkeypatch.setattr(medical_processor, "_judge_field_conflict", patient_name_judge)

    result = asyncio.run(
        medical_processor.process(
            [
                DocumentImageInput(
                    filename="scan.png",
                    mime_type="image/png",
                    base64_data="YWJj",
                )
            ]
        )
    )

    assert result.result.patient_info.name == "Nathan Benedict Kristian"
    assert result.field_confidence["patient_info.name"]["judge_used"] is True
    assert any(
        "ambiguous patient-name candidates" in hit
        for hit in result.field_confidence["patient_info.name"]["rule_hits"]
    )


def test_process_skips_patient_name_judge_when_consensus_is_strong(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    medical_processor = MedicalDocumentProcessor(settings=build_settings())

    class FakeModel:
        async def ainvoke(self, messages: list[object]) -> AIMessage:
            del messages
            payload = {
                **SCAN_RESULT,
                "patient_info": {
                    **SCAN_RESULT["patient_info"],
                    "name": "Nathan Benedict Kristian",
                },
            }
            return AIMessage(content=json.dumps(payload))

    async def should_not_be_called(
        field_path: str,
        images: list[object],
        candidates: list[dict[str, object]],
    ) -> tuple[str | None, str | None]:
        raise AssertionError(f"Unexpected judge invocation for field: {field_path}")

    monkeypatch.setattr(
        medical_processor,
        "_create_chat_model",
        lambda model_spec: FakeModel(),
    )
    monkeypatch.setattr(
        medical_processor,
        "_get_enabled_model_specs",
        lambda: [
            AIModelSpec(provider="google_genai", model="model-a"),
            AIModelSpec(provider="google_genai", model="model-b"),
            AIModelSpec(provider="google_genai", model="model-c"),
        ],
    )
    monkeypatch.setattr(medical_processor, "_judge_field_conflict", should_not_be_called)

    result = asyncio.run(
        medical_processor.process(
            [
                DocumentImageInput(
                    filename="scan.png",
                    mime_type="image/png",
                    base64_data="YWJj",
                )
            ]
        )
    )

    assert result.result.patient_info.name == "Nathan Benedict Kristian"
    assert result.field_confidence["patient_info.name"]["judge_used"] is False


def test_process_canonicalizes_structured_medical_terms_and_records_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    medical_processor = MedicalDocumentProcessor(settings=build_settings())
    payload = {
        **SCAN_RESULT,
        "medical_context": {
            **SCAN_RESULT["medical_context"],
            "diagnoses": [{"description": "pertesis", "doctor_name": "dr. Budi"}],
        },
    }

    class FakeModel:
        async def ainvoke(self, messages: list[object]) -> AIMessage:
            del messages
            return AIMessage(content=json.dumps(payload))

    monkeypatch.setattr(
        medical_processor,
        "_create_chat_model",
        lambda model_spec: FakeModel(),
    )
    monkeypatch.setattr(
        medical_processor,
        "_get_enabled_model_specs",
        lambda: [AIModelSpec(provider="google_genai", model="model-a")],
    )

    result = asyncio.run(
        medical_processor.process(
            [
                DocumentImageInput(
                    filename="scan.png",
                    mime_type="image/png",
                    base64_data="YWJj",
                )
            ]
        )
    )

    assert result.result.medical_context.diagnoses[0].description == "Pertusis"
    assert result.extraction_meta_payload["normalized_terms"] == [
        {
            "field_path": "diagnoses[pertesis].description",
            "raw_value": "pertesis",
            "canonical_value": "Pertusis",
            "similarity": 0.875,
        }
    ]


def test_process_requires_minimum_successful_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    medical_processor = MedicalDocumentProcessor(settings=build_settings())

    class FailingModel:
        async def ainvoke(self, messages: list[object]) -> AIMessage:
            del messages
            raise HTTPException(status_code=502, detail="Upstream model failure.")

    monkeypatch.setattr(medical_processor, "_create_chat_model", lambda model_spec: FailingModel())
    monkeypatch.setattr(
        medical_processor,
        "_get_enabled_model_specs",
        lambda: [AIModelSpec(provider="google_genai", model="model-a")],
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            medical_processor.process(
                [
                    DocumentImageInput(
                        filename="scan.png",
                        mime_type="image/png",
                        base64_data="YWJj",
                    )
                ]
            )
        )

    assert exc_info.value.status_code == 502
    assert exc_info.value.detail == "AI ensemble failed to produce enough valid outputs."


def test_get_enabled_model_specs_ignores_active_providers_that_are_not_scan_compatible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    medical_processor = MedicalDocumentProcessor(settings=build_settings())
    session_factory = build_session_factory_with_ai_models(
        [
            {
                "display_name": "Groq Model",
                "model_name": "llama-3.3-70b-versatile",
                "provider": "groq",
                "priority": 1,
                "is_active": True,
            },
            {
                "display_name": "Gemini Model",
                "model_name": "gemini-1.5-flash",
                "provider": "google_genai",
                "priority": 10,
                "is_active": True,
            },
        ]
    )

    monkeypatch.setattr("app.services.ai_service.get_session_factory", lambda: session_factory)

    enabled_models = medical_processor._get_enabled_model_specs()

    assert [model.provider for model in enabled_models] == ["google_genai"]
    assert [model.model for model in enabled_models] == ["gemini-1.5-flash"]


def test_get_enabled_model_specs_rejects_when_all_active_models_are_not_scan_compatible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    medical_processor = MedicalDocumentProcessor(settings=build_settings())
    session_factory = build_session_factory_with_ai_models(
        [
            {
                "display_name": "Groq Model",
                "model_name": "llama-3.3-70b-versatile",
                "provider": "groq",
                "priority": 1,
                "is_active": True,
            }
        ]
    )

    monkeypatch.setattr("app.services.ai_service.get_session_factory", lambda: session_factory)

    with pytest.raises(HTTPException) as exc_info:
        medical_processor._get_enabled_model_specs()

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Active AI models are configured, but none currently support this image-based scan pipeline."


def test_create_chat_model_uses_ollama_cloud_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    medical_processor = MedicalDocumentProcessor(
        settings=Settings(
            app_name="python-backend",
            app_env="test",
            openai_api_key="",
            google_api_key="",
            jwt_secret="test-secret",
            access_token_expire_minutes=60,
            database_url="sqlite:///:memory:",
            ollama_api_key="test-ollama-key",
        )
    )

    captured_kwargs: dict[str, object] = {}

    class FakeChatOllama:
        def __init__(self, **kwargs: object) -> None:
            captured_kwargs.update(kwargs)

    class FakeModule:
        ChatOllama = FakeChatOllama

    monkeypatch.setattr("app.services.ai_service.import_module", lambda package_name: FakeModule())

    medical_processor._create_chat_model(AIModelSpec(provider="ollama", model="llama3.2-vision"))

    assert captured_kwargs["model"] == "llama3.2-vision"
    assert captured_kwargs["base_url"] == "https://ollama.com"
    assert captured_kwargs["client_kwargs"] == {
        "headers": {
            "Authorization": "Bearer test-ollama-key",
        }
    }


def test_create_chat_model_disables_openrouter_attribution_fields_for_sdk_compatibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    medical_processor = MedicalDocumentProcessor(
        settings=Settings(
            app_name="python-backend",
            app_env="test",
            openai_api_key="",
            google_api_key="",
            jwt_secret="test-secret",
            access_token_expire_minutes=60,
            database_url="sqlite:///:memory:",
            openrouter_api_key="test-openrouter-key",
        )
    )

    captured_kwargs: dict[str, object] = {}

    class FakeChatOpenRouter:
        def __init__(self, **kwargs: object) -> None:
            captured_kwargs.update(kwargs)

    class FakeModule:
        ChatOpenRouter = FakeChatOpenRouter

    monkeypatch.setattr("app.services.ai_service.import_module", lambda package_name: FakeModule())

    medical_processor._create_chat_model(
        AIModelSpec(provider="openrouter", model="meta-llama/llama-3.2-11b-vision-instruct")
    )

    assert captured_kwargs["model"] == "meta-llama/llama-3.2-11b-vision-instruct"
    assert captured_kwargs["api_key"] == "test-openrouter-key"
    assert captured_kwargs["app_title"] is None
    assert captured_kwargs["app_url"] is None


def test_matches_any_patient_name_candidate_accepts_exact_and_single_typo_match() -> None:
    candidates = ["Siti Aminah", "Budi Saputra"]

    assert matches_any_patient_name_candidate("  siti aminah  ", candidates) is True
    assert matches_any_patient_name_candidate("Budi Saputr", candidates) is True


def test_matches_any_patient_name_candidate_rejects_larger_mismatch() -> None:
    candidates = ["Siti Aminah", "Budi Saputra"]

    assert matches_any_patient_name_candidate("Andi Saputra", candidates) is False
    assert matches_any_patient_name_candidate("Buid Saputra", candidates) is False


def test_matches_any_patient_name_candidate_allows_empty_name() -> None:
    candidates = ["Siti Aminah", "Budi Saputra"]

    assert matches_any_patient_name_candidate("", candidates) is True
    assert matches_any_patient_name_candidate(None, candidates) is True


def test_document_validation_processor_accepts_valid_receipt_with_stamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validation_processor = DocumentValidationProcessor(settings=build_settings())

    class FakeModel:
        async def ainvoke(self, messages: list[object]) -> AIMessage:
            assert len(messages) == 2
            return AIMessage(content=json.dumps(VALIDATION_RESULT))

    monkeypatch.setattr(
        validation_processor,
        "_create_chat_model",
        lambda model_spec: FakeModel(),
    )
    monkeypatch.setattr(
        validation_processor,
        "_get_enabled_model_specs",
        lambda: [AIModelSpec(provider="google_genai", model="gemini-1.5-flash")],
    )

    result = asyncio.run(
        validation_processor.process(
            [
                DocumentImageInput(
                    filename="scan.jpg",
                    mime_type="image/jpeg",
                    base64_data="YWJj",
                )
            ]
        )
    )

    assert result.model_dump() == VALIDATION_RESULT


def test_document_validation_processor_allows_receipt_without_stamp_or_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validation_processor = DocumentValidationProcessor(settings=build_settings())
    receipt_result = {
        "images": [
            {
                "image_index": 0,
                "filename": "scan.jpg",
                "document_count_detected": 1,
                "documents": [
                    {
                        "document_index": 0,
                        "document_type": "receipt",
                        "is_medical_document": True,
                        "readability_ok": True,
                        "facility_identity_visible": True,
                        "stamp_detected": False,
                        "signature_detected": False,
                        "decision": "valid",
                        "reasons": [],
                    }
                ],
            }
        ]
    }

    class FakeModel:
        async def ainvoke(self, messages: list[object]) -> AIMessage:
            assert len(messages) == 2
            return AIMessage(content=json.dumps(receipt_result))

    monkeypatch.setattr(validation_processor, "_create_chat_model", lambda model_spec: FakeModel())
    monkeypatch.setattr(
        validation_processor,
        "_get_enabled_model_specs",
        lambda: [AIModelSpec(provider="google_genai", model="gemini-1.5-flash")],
    )

    result = asyncio.run(
        validation_processor.process(
            [
                DocumentImageInput(
                    filename="scan.jpg",
                    mime_type="image/jpeg",
                    base64_data="YWJj",
                )
            ]
        )
    )

    assert result.review_required is False
    assert result.review_reasons == []
    assert result.model_dump() == {
        "packet_patient_name": None,
        "patient_name_match_status": "not_checked",
        "patient_name_match_reason": None,
        "review_required": False,
        "review_reasons": [],
        **receipt_result,
    }


def test_document_validation_processor_rejects_standalone_prescription_without_stamp_or_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validation_processor = DocumentValidationProcessor(settings=build_settings())
    invalid_result = {
        "images": [
            {
                "image_index": 0,
                "filename": "scan.jpg",
                "document_count_detected": 1,
                "documents": [
                    {
                        "document_index": 0,
                        "document_type": "prescription",
                        "is_medical_document": True,
                        "readability_ok": True,
                        "facility_identity_visible": True,
                        "stamp_detected": False,
                        "signature_detected": False,
                        "decision": "valid",
                        "reasons": [],
                    }
                ],
            }
        ]
    }

    class FakeModel:
        async def ainvoke(self, messages: list[object]) -> AIMessage:
            assert len(messages) == 2
            return AIMessage(content=json.dumps(invalid_result))

    monkeypatch.setattr(validation_processor, "_create_chat_model", lambda model_spec: FakeModel())
    monkeypatch.setattr(
        validation_processor,
        "_get_enabled_model_specs",
        lambda: [AIModelSpec(provider="google_genai", model="gemini-1.5-flash")],
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            validation_processor.process(
                [
                    DocumentImageInput(
                        filename="scan.jpg",
                        mime_type="image/jpeg",
                        base64_data="YWJj",
                    )
                ]
            )
        )

    assert exc_info.value.status_code == 422
    detail = get_http_exception_detail_dict(exc_info.value)
    assert detail["message"] == "Document validation failed."
    assert "visible facility stamp or doctor signature is required" in detail["errors"][0]["message"]


def test_document_validation_processor_allows_lab_result_without_stamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validation_processor = DocumentValidationProcessor(settings=build_settings())
    lab_result = {
        "images": [
            {
                "image_index": 0,
                "filename": "scan.jpg",
                "document_count_detected": 1,
                "documents": [
                    {
                        "document_index": 0,
                        "document_type": "lab_result",
                        "is_medical_document": True,
                        "readability_ok": True,
                        "facility_identity_visible": True,
                        "stamp_detected": False,
                        "signature_detected": False,
                        "decision": "valid",
                        "reasons": ["Facility identity is visible."],
                    }
                ],
            }
        ]
    }

    class FakeModel:
        async def ainvoke(self, messages: list[object]) -> AIMessage:
            assert len(messages) == 2
            return AIMessage(content=json.dumps(lab_result))

    monkeypatch.setattr(validation_processor, "_create_chat_model", lambda model_spec: FakeModel())
    monkeypatch.setattr(
        validation_processor,
        "_get_enabled_model_specs",
        lambda: [AIModelSpec(provider="google_genai", model="gemini-1.5-flash")],
    )

    result = asyncio.run(
        validation_processor.process(
            [
                DocumentImageInput(
                    filename="scan.jpg",
                    mime_type="image/jpeg",
                    base64_data="YWJj",
                )
            ]
        )
    )

    assert result.model_dump() == {
        "packet_patient_name": None,
        "patient_name_match_status": "not_checked",
        "patient_name_match_reason": None,
        "review_required": False,
        "review_reasons": [],
        **lab_result,
    }


def test_document_validation_processor_allows_rotation_only_issue_for_otherwise_legible_document(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validation_processor = DocumentValidationProcessor(settings=build_settings())
    rotated_result = {
        "images": [
            {
                "image_index": 0,
                "filename": "scan.jpg",
                "document_count_detected": 1,
                "documents": [
                    {
                        "document_index": 0,
                        "document_type": "receipt",
                        "is_medical_document": True,
                        "readability_ok": False,
                        "facility_identity_visible": True,
                        "stamp_detected": True,
                        "signature_detected": False,
                        "decision": "invalid",
                        "reasons": ["Document is upside down."],
                    }
                ],
            }
        ]
    }

    class FakeModel:
        async def ainvoke(self, messages: list[object]) -> AIMessage:
            assert len(messages) == 2
            return AIMessage(content=json.dumps(rotated_result))

    monkeypatch.setattr(validation_processor, "_create_chat_model", lambda model_spec: FakeModel())
    monkeypatch.setattr(
        validation_processor,
        "_get_enabled_model_specs",
        lambda: [AIModelSpec(provider="google_genai", model="gemini-1.5-flash")],
    )

    result = asyncio.run(
        validation_processor.process(
            [
                DocumentImageInput(
                    filename="scan.jpg",
                    mime_type="image/jpeg",
                    base64_data="YWJj",
                )
            ]
        )
    )

    assert result.model_dump() == {
        "packet_patient_name": None,
        "patient_name_match_status": "not_checked",
        "patient_name_match_reason": None,
        "review_required": False,
        "review_reasons": [],
        **rotated_result,
    }


def test_document_validation_processor_still_rejects_rotated_and_blurry_document(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validation_processor = DocumentValidationProcessor(settings=build_settings())
    invalid_result = {
        "images": [
            {
                "image_index": 0,
                "filename": "scan.jpg",
                "document_count_detected": 1,
                "documents": [
                    {
                        "document_index": 0,
                        "document_type": "receipt",
                        "is_medical_document": True,
                        "readability_ok": False,
                        "facility_identity_visible": True,
                        "stamp_detected": True,
                        "signature_detected": False,
                        "decision": "invalid",
                        "reasons": ["Document is rotated 90 degrees and blurry."],
                    }
                ],
            }
        ]
    }

    class FakeModel:
        async def ainvoke(self, messages: list[object]) -> AIMessage:
            assert len(messages) == 2
            return AIMessage(content=json.dumps(invalid_result))

    monkeypatch.setattr(validation_processor, "_create_chat_model", lambda model_spec: FakeModel())
    monkeypatch.setattr(
        validation_processor,
        "_get_enabled_model_specs",
        lambda: [AIModelSpec(provider="google_genai", model="gemini-1.5-flash")],
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            validation_processor.process(
                [
                    DocumentImageInput(
                        filename="scan.jpg",
                        mime_type="image/jpeg",
                        base64_data="YWJj",
                    )
                ]
            )
        )

    assert exc_info.value.status_code == 422
    detail = get_http_exception_detail_dict(exc_info.value)
    assert detail["message"] == "Document validation failed."


def test_document_validation_processor_rejects_non_medical_and_uncertain_documents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validation_processor = DocumentValidationProcessor(settings=build_settings())
    invalid_result = {
        "images": [
            {
                "image_index": 0,
                "filename": "scan.jpg",
                "document_count_detected": 2,
                "documents": [
                    {
                        "document_index": 0,
                        "document_type": "non_medical",
                        "is_medical_document": False,
                        "readability_ok": True,
                        "facility_identity_visible": False,
                        "stamp_detected": False,
                        "signature_detected": False,
                        "decision": "invalid",
                        "reasons": ["The image appears to be a non-medical document."],
                    },
                    {
                        "document_index": 1,
                        "document_type": "unknown",
                        "is_medical_document": True,
                        "readability_ok": False,
                        "facility_identity_visible": False,
                        "stamp_detected": False,
                        "signature_detected": False,
                        "decision": "uncertain",
                        "reasons": ["Multiple papers are present but cannot be separated clearly."],
                    },
                ],
            }
        ]
    }

    class FakeModel:
        async def ainvoke(self, messages: list[object]) -> AIMessage:
            assert len(messages) == 2
            return AIMessage(content=json.dumps(invalid_result))

    monkeypatch.setattr(validation_processor, "_create_chat_model", lambda model_spec: FakeModel())
    monkeypatch.setattr(
        validation_processor,
        "_get_enabled_model_specs",
        lambda: [AIModelSpec(provider="google_genai", model="gemini-1.5-flash")],
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            validation_processor.process(
                [
                    DocumentImageInput(
                        filename="scan.jpg",
                        mime_type="image/jpeg",
                        base64_data="YWJj",
                    )
                ]
            )
        )

    assert exc_info.value.status_code == 422
    detail = get_http_exception_detail_dict(exc_info.value)
    assert detail["message"] == "Document validation failed."
    assert "non-medical document" in detail["errors"][0]["message"]


def test_document_validation_processor_soft_reviews_collage_with_missing_signature_on_strict_document(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validation_processor = DocumentValidationProcessor(settings=build_settings())
    collage_result = {
        "images": [
            {
                "image_index": 0,
                "filename": "scan.jpg",
                "document_count_detected": 2,
                "documents": [
                    {
                        "document_index": 0,
                        "document_type": "receipt",
                        "is_medical_document": True,
                        "readability_ok": True,
                        "facility_identity_visible": True,
                        "stamp_detected": False,
                        "signature_detected": False,
                        "decision": "valid",
                        "reasons": ["Facility identity is visible."],
                    },
                    {
                        "document_index": 1,
                        "document_type": "prescription",
                        "is_medical_document": True,
                        "readability_ok": True,
                        "facility_identity_visible": True,
                        "stamp_detected": False,
                        "signature_detected": False,
                        "decision": "valid",
                        "reasons": ["Stamp/signature is hard to verify because multiple papers are present in one image."],
                    },
                ],
            }
        ]
    }

    class FakeModel:
        async def ainvoke(self, messages: list[object]) -> AIMessage:
            assert len(messages) == 2
            return AIMessage(content=json.dumps(collage_result))

    monkeypatch.setattr(validation_processor, "_create_chat_model", lambda model_spec: FakeModel())
    monkeypatch.setattr(
        validation_processor,
        "_get_enabled_model_specs",
        lambda: [AIModelSpec(provider="google_genai", model="gemini-1.5-flash")],
    )

    result = asyncio.run(
        validation_processor.process(
            [
                DocumentImageInput(
                    filename="scan.jpg",
                    mime_type="image/jpeg",
                    base64_data="YWJj",
                )
            ]
        )
    )

    assert result.review_required is True
    assert result.review_reasons == [
        "Image 1 document 2: a visible facility stamp or doctor signature is required."
    ]


def test_document_validation_processor_soft_reviews_collage_separation_ambiguity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validation_processor = DocumentValidationProcessor(settings=build_settings())
    collage_result = {
        "images": [
            {
                "image_index": 0,
                "filename": "scan.jpg",
                "document_count_detected": 2,
                "documents": [
                    {
                        "document_index": 0,
                        "document_type": "receipt",
                        "is_medical_document": True,
                        "readability_ok": True,
                        "facility_identity_visible": True,
                        "stamp_detected": False,
                        "signature_detected": False,
                        "decision": "uncertain",
                        "reasons": ["Multiple papers are present but cannot be separated clearly."],
                    }
                ],
            }
        ]
    }

    class FakeModel:
        async def ainvoke(self, messages: list[object]) -> AIMessage:
            assert len(messages) == 2
            return AIMessage(content=json.dumps(collage_result))

    monkeypatch.setattr(validation_processor, "_create_chat_model", lambda model_spec: FakeModel())
    monkeypatch.setattr(
        validation_processor,
        "_get_enabled_model_specs",
        lambda: [AIModelSpec(provider="google_genai", model="gemini-1.5-flash")],
    )

    result = asyncio.run(
        validation_processor.process(
            [
                DocumentImageInput(
                    filename="scan.jpg",
                    mime_type="image/jpeg",
                    base64_data="YWJj",
                )
            ]
        )
    )

    assert result.review_required is True
    assert result.review_reasons == [
        "Image 1: multiple documents were present but could not be validated clearly.",
        "Image 1 document 1: Multiple papers are present but cannot be separated clearly.",
    ]


def test_document_validation_processor_marks_family_patient_name_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validation_processor = DocumentValidationProcessor(settings=build_settings())
    validation_result = {
        "packet_patient_name": "Andi Saputra",
        "patient_name_match_status": "matched",
        "patient_name_match_reason": None,
        "images": [
            {
                "image_index": 0,
                "filename": "scan.jpg",
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

    class FakeModel:
        async def ainvoke(self, messages: list[object]) -> AIMessage:
            assert len(messages) == 2
            return AIMessage(content=json.dumps(validation_result))

    monkeypatch.setattr(validation_processor, "_create_chat_model", lambda model_spec: FakeModel())
    monkeypatch.setattr(
        validation_processor,
        "_get_enabled_model_specs",
        lambda: [AIModelSpec(provider="google_genai", model="gemini-1.5-flash")],
    )

    result = asyncio.run(
        validation_processor.process(
            [
                DocumentImageInput(
                    filename="scan.jpg",
                    mime_type="image/jpeg",
                    base64_data="YWJj",
                )
            ],
            patient_name_candidates=["Siti Aminah", "Budi Saputra"],
        )
    )

    assert result.packet_patient_name == "Andi Saputra"
    assert result.patient_name_match_status == "mismatched"
    assert result.patient_name_match_reason == "Document patient name 'Andi Saputra' does not match the user or any saved family member."


def test_document_validation_processor_marks_expected_patient_name_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validation_processor = DocumentValidationProcessor(settings=build_settings())
    validation_result = {
        "packet_patient_name": "Andi Saputra",
        "patient_name_match_status": "matched",
        "patient_name_match_reason": None,
        "images": [
            {
                "image_index": 0,
                "filename": "scan.jpg",
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

    class FakeModel:
        async def ainvoke(self, messages: list[object]) -> AIMessage:
            assert len(messages) == 2
            return AIMessage(content=json.dumps(validation_result))

    monkeypatch.setattr(validation_processor, "_create_chat_model", lambda model_spec: FakeModel())
    monkeypatch.setattr(
        validation_processor,
        "_get_enabled_model_specs",
        lambda: [AIModelSpec(provider="google_genai", model="gemini-1.5-flash")],
    )

    result = asyncio.run(
        validation_processor.process(
            [
                DocumentImageInput(
                    filename="scan.jpg",
                    mime_type="image/jpeg",
                    base64_data="YWJj",
                )
            ],
            expected_patient_name="Budi Santoso",
        )
    )

    assert result.patient_name_match_status == "mismatched"
    assert result.patient_name_match_reason == "Document patient name 'Andi Saputra' does not match the selected patient."


def test_document_validation_processor_marks_missing_patient_name_as_review_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validation_processor = DocumentValidationProcessor(settings=build_settings())
    validation_result = {
        "packet_patient_name": None,
        "patient_name_match_status": "not_found",
        "patient_name_match_reason": None,
        "images": [
            {
                "image_index": 0,
                "filename": "scan.jpg",
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

    class FakeModel:
        async def ainvoke(self, messages: list[object]) -> AIMessage:
            assert len(messages) == 2
            return AIMessage(content=json.dumps(validation_result))

    monkeypatch.setattr(validation_processor, "_create_chat_model", lambda model_spec: FakeModel())
    monkeypatch.setattr(
        validation_processor,
        "_get_enabled_model_specs",
        lambda: [AIModelSpec(provider="google_genai", model="gemini-1.5-flash")],
    )

    result = asyncio.run(
        validation_processor.process(
            [
                DocumentImageInput(
                    filename="scan.jpg",
                    mime_type="image/jpeg",
                    base64_data="YWJj",
                )
            ],
            patient_name_candidates=["Siti Aminah"],
        )
    )

    assert result.patient_name_match_status == "not_found"
    assert result.patient_name_match_reason == "Patient name is not visible clearly enough in the uploaded documents."
    assert result.review_required is True
    assert "Patient name is not visible clearly enough in the uploaded documents." in result.review_reasons
