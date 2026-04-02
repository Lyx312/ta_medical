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

SCAN_RESULT = {
    "summary": {
        "detected_documents": ["HOSPITAL_INVOICE"],
        "medical_facility_name": "RS Sehat Sentosa",
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
    assert {key: value for key, value in payload.items() if key != "scan_log_id"} == SCAN_RESULT

    scan_log = get_scan_log_by_id(payload["scan_log_id"])
    assert scan_log is not None
    assert scan_log.status == "succeeded"
    assert scan_log.validation_payload == VALIDATION_RESULT
    assert scan_log.extraction_payload == SCAN_RESULT
    assert scan_log.ai_model_provider == "google_genai"
    assert scan_log.ai_model_name == "gemini-3.1-flash-lite-preview"
    assert len(scan_log.files) == 1
    assert scan_log.files[0].original_content == b"abc"
    assert scan_log.files[0].processed_content == b"processed-image"
    assert scan_log.files[0].original_sha256_hash == hashlib.sha256(b"abc").hexdigest()
    assert scan_log.files[0].processed_sha256_hash == hashlib.sha256(b"processed-image").hexdigest()


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
    assert {key: value for key, value in payload.items() if key != "scan_log_id"} == SCAN_RESULT

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

    assert result.model_dump() == SCAN_RESULT


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
    assert exc_info.value.detail == "AI model returned non-JSON response."


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
    assert exc_info.value.detail == "AI model returned invalid response shape."


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
                "is_active": True,
            },
            {
                "display_name": "Model B",
                "model_name": "model-b",
                "provider": "google_genai",
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

    assert result.model_dump() == SCAN_RESULT
    assert selected_models == ["model-a"]


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
    assert exc_info.value.detail["message"] == "Document validation failed."
    assert "visible facility stamp or doctor signature is required" in exc_info.value.detail["errors"][0]["message"]


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
    assert exc_info.value.detail["message"] == "Document validation failed."
    assert "non-medical document" in exc_info.value.detail["errors"][0]["message"]
