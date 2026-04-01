import asyncio
import base64
import json
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from langchain_core.language_models._utils import _normalize_messages
from langchain_core.messages import AIMessage

from app.api.routes import ai as ai_route
from app.api.routes.ai import processor
from app.core.config import Settings, get_settings
from app.db.session import clear_database_caches
from app.main import create_app
from app.schemas.ai import AIMedicalScanResponse
from app.services import ai_service
from app.services.ai_service import AIModelSpec, DocumentImageInput, MedicalDocumentProcessor
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
        "name": "Andi Saputra",
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
    monkeypatch.setattr(processor, "process", fake_process)

    response = client.post(
        "/api/ai/scan-documents",
        headers={"Authorization": f"Bearer {token}"},
        files=[("files", ("scan.png", b"abc", "image/png"))],
    )

    assert response.status_code == 200
    assert response.json() == SCAN_RESULT


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


def test_get_enabled_model_specs_requires_at_least_one_enabled_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    medical_processor = MedicalDocumentProcessor(settings=build_settings())

    monkeypatch.setattr(
        ai_service,
        "ACTIVE_AI_MODELS",
        [AIModelSpec(provider="google_genai", model="gemini-1.5-flash", enabled=False)],
    )

    with pytest.raises(HTTPException) as exc_info:
        medical_processor._get_enabled_model_specs()

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "No enabled AI model is configured."
