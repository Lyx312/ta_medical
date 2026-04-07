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
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.api.routes import ai as ai_route
from app.api.routes.ai import processor
from app.core.config import Settings, get_settings
from app.db.base import Base
from app.db.session import clear_database_caches, get_session_factory
from app.main import create_app
from app.models.ai_model import AIModel
from app.repositories.scan_log_repository import ScanLogRepository
from app.schemas.ai import AIMedicalScanResponse, DocumentValidationResponse
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

    async def fake_validation_process(images: list[DocumentImageInput]) -> DocumentValidationResponse:
        assert len(images) == 1
        assert images[0].filename == "scan.jpg"
        return DocumentValidationResponse.model_validate(VALIDATION_RESULT)

    async def fake_process(
        images: list[DocumentImageInput],
        patient_name_candidates: list[str] | None = None,
    ) -> AIMedicalScanResponse:
        assert len(images) == 1
        assert images[0].filename == "scan.jpg"
        assert images[0].mime_type == "image/jpeg"
        assert images[0].base64_data == base64.b64encode(b"processed-image").decode("utf-8")
        assert patient_name_candidates == ["Siti Aminah", "Budi Saputra"]
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
    assert payload["field_confidence"] == {}
    assert payload["review_required"] is False
    assert payload["review_reasons"] == []
    assert payload["calibration_version"] == "legacy"

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

    async def fake_validation_process(images: list[DocumentImageInput]) -> DocumentValidationResponse:
        assert len(images) == 1
        return DocumentValidationResponse.model_validate(VALIDATION_RESULT)

    async def fake_process(
        images: list[DocumentImageInput],
        patient_name_candidates: list[str] | None = None,
    ) -> AIMedicalScanResponse:
        del images, patient_name_candidates
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


def test_scan_documents_rejects_extracted_patient_name_outside_family_candidates(
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

    async def fake_validation_process(images: list[DocumentImageInput]) -> DocumentValidationResponse:
        assert len(images) == 1
        return DocumentValidationResponse.model_validate(VALIDATION_RESULT)

    async def fake_process(
        images: list[DocumentImageInput],
        patient_name_candidates: list[str] | None = None,
    ) -> AIMedicalScanResponse:
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
    assert scan_log.validation_payload == VALIDATION_RESULT
    assert scan_log.extraction_payload is not None
    assert scan_log.extraction_payload["patient_info"]["name"] == "Andi Saputra"
    assert scan_log.error_payload is not None
    assert scan_log.error_payload["detail"]["message"] == "Patient name verification failed."


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

    async def fake_validation_process(images: list[DocumentImageInput]) -> DocumentValidationResponse:
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

    async def fake_validation_process(images: list[DocumentImageInput]) -> DocumentValidationResponse:
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
    ) -> AIMedicalScanResponse:
        assert len(images) == 1
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
        if field_path == "medical_context.anamnesis":
            raise AssertionError("Judge should not have been invoked for medical_context.anamnesis")
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
    assert result.field_confidence["medical_context.anamnesis"]["judge_used"] is False
    assert result.extraction_meta_payload["constraint_checks"]["financial_recap.grand_total"]["used_derived_value"] is False
    assert "financial_recap.grand_total did not match derived financial totals." not in result.review_reasons


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


def test_document_validation_processor_rejects_receipt_without_stamp_or_signature(
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

    assert result.model_dump() == lab_result


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

    assert result.model_dump() == rotated_result


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
