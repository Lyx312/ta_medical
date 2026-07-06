import asyncio
import base64
import hashlib
import json
from pathlib import Path
from typing import Any, AsyncIterator, cast

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from langchain_core.language_models._utils import _normalize_messages
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.api.routes import ai as ai_route
from app.api.routes.ai import processor
from app.core.ai_pipeline_logging import (
    ai_pipeline_log,
    clear_ai_pipeline_debug_logging_cache,
    format_ai_pipeline_terminal_line,
    get_ai_pipeline_logger,
)
from app.core.config import Settings, get_settings
from app.db.base import Base
from app.db.session import clear_database_caches, get_session_factory
from app.main import create_app
from app.models.ai_model import AIModel
from app.models.scan_log import AIApiCallLog
from app.repositories.scan_log_repository import ScanLogFileCreate, ScanLogRepository
from app.schemas.ai import AIMedicalScanResponse, DocumentValidationResponse
from app.services import ai_service as ai_service_module
from app.services import claim_medicine_relevance as medicine_relevance_service
from app.services.ai_ensemble import (
    FusionWorkingState,
    MedicalExtractionOutcome,
    MedicalExtractionFusionEngine,
    ProducerInferenceTrace,
    clear_self_consistency_calibration_cache,
    parse_indonesian_amount_words,
)
from app.services.ai_ensemble_helpers import classify_other_charge_kind
from app.services.ai_service import (
    AIModelSpec,
    DocumentImageInput,
    DocumentValidationProcessor,
    MedicalDocumentProcessor,
    matches_any_patient_name_candidate,
)
from app.services.image_preprocessing import ProcessedDocumentImage
from app.services.scan_medicine_prioritization import (
    BatchMedicineInterpretationResponse,
    ScanMedicineImageInterpretationJudge,
    prioritize_scan_medicines,
)
from app.services.scan_progress import ScanProgressEventHub, generate_scan_progress_sse
from app.services import model_invocation_test as model_invocation_test_module


def get_message_block(content: Any, index: int) -> dict[str, Any]:
    assert isinstance(content, list)
    block = content[index]
    assert isinstance(block, dict)
    return cast(dict[str, Any], block)


def get_http_exception_detail_dict(exc: HTTPException) -> dict[str, Any]:
    assert isinstance(exc.detail, dict)
    return cast(dict[str, Any], exc.detail)


def parse_sse_message(raw_message: str) -> tuple[str, dict[str, Any] | None]:
    event_name = ""
    payload: dict[str, Any] | None = None

    for line in raw_message.strip().splitlines():
        if line.startswith("event: "):
            event_name = line.removeprefix("event: ").strip()
            continue
        if line.startswith("data: "):
            payload = json.loads(line.removeprefix("data: ").strip())
            continue

    return event_name, payload


async def next_sse_message(
    generator: AsyncIterator[str],
    timeout: float = 0.5,
) -> tuple[str, dict[str, Any] | None]:
    raw_message = await asyncio.wait_for(anext(generator), timeout=timeout)
    return parse_sse_message(raw_message)


def extract_ai_pipeline_log_lines(raw_output: str) -> list[str]:
    lines: list[str] = []
    for line in raw_output.splitlines():
        normalized = line.strip()
        if not normalized or not normalized.startswith("[AI]"):
            continue
        lines.append(normalized)
    return lines

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
    for provider_env_var in (
        "OPENAI_API_KEY",
        "GOOGLE_API_KEY",
        "OPENROUTER_API_KEY",
        "OLLAMA_API_KEY",
        "GROQ_API_KEY",
        "MISTRAL_API_KEY",
        "DEEPSEEK_API_KEY",
        "COHERE_API_KEY",
        "CEREBRAS_API_KEY",
    ):
        monkeypatch.delenv(provider_env_var, raising=False)
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


def get_user_id_from_token(client: TestClient, token: str) -> int:
    response = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    return int(response.json()["id"])


def test_ai_pipeline_debug_logging_is_disabled_by_default(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("AI_PIPELINE_DEBUG_LOGS", raising=False)
    clear_ai_pipeline_debug_logging_cache()

    ai_pipeline_log(event="sample_event", component="test_component", value=1)

    captured = capsys.readouterr()
    assert extract_ai_pipeline_log_lines(captured.err) == []


def test_ai_pipeline_debug_logging_emits_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("AI_PIPELINE_DEBUG_LOGS", "true")
    clear_ai_pipeline_debug_logging_cache()

    ai_pipeline_log(event="sample_event", component="test_component", value=1)

    captured = capsys.readouterr()
    lines = extract_ai_pipeline_log_lines(captured.err)
    assert lines == ["[AI] sample_event"]
    assert not lines[0].startswith("{")

    monkeypatch.delenv("AI_PIPELINE_DEBUG_LOGS", raising=False)
    clear_ai_pipeline_debug_logging_cache()


def test_ai_pipeline_debug_logging_configures_single_owned_stream_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AI_PIPELINE_DEBUG_LOGS", "true")
    clear_ai_pipeline_debug_logging_cache()

    logger = get_ai_pipeline_logger()
    owned_handlers = [
        handler
        for handler in logger.handlers
        if bool(getattr(handler, "_ai_pipeline_owned_handler", False))
    ]
    assert logger.level == 20
    assert logger.propagate is False
    assert len(owned_handlers) == 1

    logger = get_ai_pipeline_logger()
    owned_handlers = [
        handler
        for handler in logger.handlers
        if bool(getattr(handler, "_ai_pipeline_owned_handler", False))
    ]
    assert len(owned_handlers) == 1

    monkeypatch.delenv("AI_PIPELINE_DEBUG_LOGS", raising=False)
    clear_ai_pipeline_debug_logging_cache()


def test_format_ai_pipeline_terminal_line_renders_compact_known_events() -> None:
    assert format_ai_pipeline_terminal_line(
        "scan_request_started",
        {"component": "ai_scan_route", "scan_log_id": 51, "upload_count": 4},
    ) == "[AI][scan:51] SCAN start files=4"
    assert format_ai_pipeline_terminal_line(
        "validation_inference_completed",
        {
            "component": "document_validation_processor",
            "scan_log_id": 51,
            "model": "gemini-3.1-flash-lite-preview",
            "latency_ms": 14798,
            "total_tokens": 6723,
        },
    ) == "[AI][scan:51] VALIDATE model=gemini-3.1-flash-lite-preview done 14.8s tok=6723"
    assert format_ai_pipeline_terminal_line(
        "ensemble_fusion_completed",
        {
            "component": "medical_extraction_fusion_engine",
            "scan_log_id": 51,
            "judge_invocation_count": 0,
            "judge_attempt_count": 2,
            "judge_applied_count": 0,
            "nullified_field_count": 3,
            "overall_confidence": 0.8965,
            "review_required": True,
        },
    ) == "[AI][scan:51] FUSE done judges=0/2 nulls=3 conf=0.8965 review=yes"
    assert format_ai_pipeline_terminal_line(
        "medicine_prioritization_batch_failed",
        {
            "component": "scan_medicine_prioritization",
            "scan_log_id": 51,
            "error_message": "Batch medicine interpretation payload is invalid.",
        },
    ) == "[AI][scan:51] MEDS batch failed err=Batch medicine interpretation payload is invalid."


def test_format_ai_pipeline_terminal_line_truncates_long_error_text() -> None:
    long_error = (
        "5 validation errors for BatchMedicineInterpretationResponse judgements.1.name "
        "Extra inputs are not permitted For further information visit https://errors.pydantic.dev/2.12/v/extra_forbidden "
        "judgements.2.name Extra inputs are not permitted"
    )
    rendered = format_ai_pipeline_terminal_line(
        "medicine_prompt_failed",
        {
            "component": "scan_medicine_prioritization",
            "scan_log_id": 51,
            "error_message": long_error,
        },
    )
    assert rendered is not None
    assert "For further information visit" not in rendered
    assert len(rendered) < 170


def test_medical_document_processor_emits_extraction_and_ensemble_logs_without_pii(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    medical_processor = MedicalDocumentProcessor(settings=build_settings())

    class FakeModel:
        async def ainvoke(self, messages: list[object]) -> AIMessage:
            del messages
            return AIMessage(content=json.dumps(SCAN_RESULT))

    monkeypatch.setenv("AI_PIPELINE_DEBUG_LOGS", "true")
    clear_ai_pipeline_debug_logging_cache()
    monkeypatch.setattr(medical_processor, "_create_chat_model", lambda model_spec: FakeModel())
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
            validation_hints={},
            ai_call_context={"scan_log_id": 123, "user_id": 456},
        )
    )

    captured = capsys.readouterr()
    lines = extract_ai_pipeline_log_lines(captured.err)
    assert result.result.patient_info.name == "Budi Saputra"
    assert any("EXTRACT start" in line for line in lines)
    assert any("RUN p1r1 model-a start" in line for line in lines)
    assert any("RUN p1r1 model-a done" in line for line in lines)
    assert any("FUSE start" in line for line in lines)
    assert any("FUSE done" in line for line in lines)
    assert any("EXTRACT done" in line for line in lines)
    joined_logs = captured.err
    assert "Budi Saputra" not in joined_logs
    assert "Influenza" not in joined_logs
    assert "Demam dua hari" not in joined_logs
    assert "{" not in joined_logs

    monkeypatch.delenv("AI_PIPELINE_DEBUG_LOGS", raising=False)
    clear_ai_pipeline_debug_logging_cache()


def test_field_judge_emits_logs_without_candidate_values(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    medical_processor = MedicalDocumentProcessor(settings=build_settings())

    class FakeJudgeModel:
        async def ainvoke(self, messages: list[object]) -> AIMessage:
            del messages
            return AIMessage(content=json.dumps({"selected_candidate_index": 0, "reason": "Prefer stronger support"}))

    monkeypatch.setenv("AI_PIPELINE_DEBUG_LOGS", "true")
    clear_ai_pipeline_debug_logging_cache()
    monkeypatch.setattr(medical_processor, "_create_chat_model", lambda model_spec: FakeJudgeModel())
    monkeypatch.setattr(
        medical_processor,
        "_get_enabled_model_specs",
        lambda: [AIModelSpec(provider="google_genai", model="judge-model")],
    )

    selected_key, reason = asyncio.run(
        medical_processor._judge_field_conflict(
            "patient_info.address",
            [
                DocumentImageInput(
                    filename="scan.png",
                    mime_type="image/png",
                    base64_data="YWJj",
                )
            ],
            [
                {"raw_value": "JL. Alpha No.1", "normalized_key": "alpha", "run_count": 2, "producers": {"p1", "p2"}},
                {"raw_value": "JL. Beta No.2", "normalized_key": "beta", "run_count": 1, "producers": {"p3"}},
            ],
            ai_call_context={"scan_log_id": 99, "user_id": 7},
        )
    )

    captured = capsys.readouterr()
    lines = extract_ai_pipeline_log_lines(captured.err)
    assert selected_key == "alpha"
    assert reason == "Prefer stronger support"
    assert not any("field_judge_started" in line for line in lines)
    assert any("JUDGE done field=patient_info.address candidates=2 selected=yes" in line for line in lines)
    joined_logs = captured.err
    assert "JL. Alpha No.1" not in joined_logs
    assert "JL. Beta No.2" not in joined_logs

    monkeypatch.delenv("AI_PIPELINE_DEBUG_LOGS", raising=False)
    clear_ai_pipeline_debug_logging_cache()


def test_prioritize_scan_medicines_emits_medicine_prompt_logs_without_pii(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("AI_PIPELINE_DEBUG_LOGS", "true")
    clear_ai_pipeline_debug_logging_cache()

    class FakeModel:
        async def ainvoke(self, messages: list[object]) -> AIMessage:
            del messages
            return AIMessage(
                content=json.dumps(
                    {
                        "judgements": [
                            {
                                "original_index": 0,
                                "interpreted_name": None,
                                "interpretation_confidence": 0.4,
                                "interpretation_reason": "insufficient evidence",
                                "relevance": "relevant",
                                "relevance_confidence": 0.72,
                                "matched_diagnosis": False,
                                "matched_anamnesis": False,
                            },
                            {
                                "original_index": 1,
                                "interpreted_name": None,
                                "interpretation_confidence": 0.4,
                                "interpretation_reason": "insufficient evidence",
                                "relevance": "uncertain",
                                "relevance_confidence": 0.61,
                                "matched_diagnosis": False,
                                "matched_anamnesis": False,
                            },
                        ]
                    }
                )
            )

    monkeypatch.setattr(ScanMedicineImageInterpretationJudge, "_create_chat_model", lambda self, model_spec: FakeModel())
    monkeypatch.setattr(
        ScanMedicineImageInterpretationJudge,
        "_get_enabled_model_specs",
        lambda self: [AIModelSpec(provider="google_genai", model="medicine-judge-model")],
    )

    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)
    scan_result = AIMedicalScanResponse.model_validate(
        {
            **SCAN_RESULT,
            "medicines": [
                {
                    "name": "Drug Alpha",
                    "quantity": 10,
                    "unit": "TABLET",
                    "dosage_instruction": "3 x 1",
                    "unit_price": 1000,
                    "total_price": 10000,
                },
                {
                    "name": "Drug Beta",
                    "quantity": 10,
                    "unit": "TABLET",
                    "dosage_instruction": "3 x 1",
                    "unit_price": 1000,
                    "total_price": 10000,
                },
            ],
        }
    )
    extraction_meta_payload = {
        "list_group_meta": {
            "medicines": {
                "selected_groups": [
                    {"output_index": 0, "key": "drugalpha", "score": 0.81, "support_count": 2, "producer_count": 1},
                    {"output_index": 1, "key": "drugbeta", "score": 0.79, "support_count": 2, "producer_count": 1},
                ],
                "groups": [],
            }
        }
    }

    with session_factory() as db:
        prioritized_result, log_payload = asyncio.run(
            prioritize_scan_medicines(
                db,
                scan_result,
                extraction_meta_payload,
                images=[
                    DocumentImageInput(
                        filename="scan.png",
                        mime_type="image/png",
                        base64_data="YWJj",
                    )
                ],
                interpretation_mode="suggest_only",
                scan_log_id=321,
                user_id=654,
            )
        )

    captured = capsys.readouterr()
    lines = extract_ai_pipeline_log_lines(captured.err)
    assert [medicine.name for medicine in prioritized_result.medicines] == ["Drug Alpha", "Drug Beta"]
    assert log_payload["enabled"] is True
    assert any("MEDS start count=2 mode=suggest_only" in line for line in lines)
    assert any("MEDS batch ambiguous=2" in line for line in lines)
    assert any("MEDS prompt model=medicine-judge-model start ambiguous=2" in line for line in lines)
    assert any("MEDS prompt done judgements=2" in line for line in lines)
    assert any("MEDS done count=2 ambiguous=2 applied=no batch=yes" in line for line in lines)
    joined_logs = captured.err
    assert "Drug Alpha" not in joined_logs
    assert "Drug Beta" not in joined_logs

    monkeypatch.delenv("AI_PIPELINE_DEBUG_LOGS", raising=False)
    clear_ai_pipeline_debug_logging_cache()


def test_scan_documents_route_emits_stage_logs_without_pii(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    token = register_and_login(client, email="scan-logs@example.com", full_name="Budi Saputra")

    def fake_preprocess_document_image(filename: str, content: bytes) -> ProcessedDocumentImage:
        del filename, content
        return ProcessedDocumentImage(
            filename="scan.jpg",
            mime_type="image/jpeg",
            content=b"processed-image",
        )

    async def fake_validation_process(images: list[DocumentImageInput], **_: object) -> DocumentValidationResponse:
        del images
        return DocumentValidationResponse.model_validate(VALIDATION_RESULT)

    async def fake_process(
        images: list[DocumentImageInput],
        patient_name_candidates: list[str] | None = None,
        validation_hints: dict[str, object] | None = None,
        **_: object,
    ) -> MedicalExtractionOutcome:
        del images, patient_name_candidates, validation_hints
        return MedicalExtractionOutcome(
            result=AIMedicalScanResponse.model_validate(SCAN_RESULT),
            overall_confidence=0.91,
            field_confidence={},
            review_required=False,
            review_reasons=[],
            calibration_version="self_consistency_logistic_v1",
            ai_execution_mode="ensemble",
            primary_model_provider="google_genai",
            primary_model_name="model-a",
            ai_runs=[],
            extraction_meta_payload={"review_required": False},
        )

    async def fake_prioritize_scan_medicines(*args: object, **kwargs: object) -> tuple[AIMedicalScanResponse, dict[str, Any]]:
        del args, kwargs
        return AIMedicalScanResponse.model_validate(SCAN_RESULT), {"enabled": True, "applied": False}

    monkeypatch.setenv("AI_PIPELINE_DEBUG_LOGS", "true")
    clear_ai_pipeline_debug_logging_cache()
    monkeypatch.setattr(ai_route, "preprocess_document_image", fake_preprocess_document_image)
    monkeypatch.setattr(ai_route.validation_processor, "process", fake_validation_process)
    monkeypatch.setattr(ai_route.processor, "process", fake_process)
    monkeypatch.setattr(ai_route, "prioritize_scan_medicines", fake_prioritize_scan_medicines)

    response = client.post(
        "/api/ai/scan-documents",
        headers={"Authorization": f"Bearer {token}"},
        files=[("files", ("scan.png", b"abc", "image/png"))],
    )

    captured = capsys.readouterr()
    lines = extract_ai_pipeline_log_lines(captured.err)
    assert response.status_code == 200
    assert any("SCAN start files=1" in line for line in lines)
    assert any("PREP done images=1" in line for line in lines)
    assert sum(1 for line in lines if "VALIDATE start" in line) == 1
    assert sum(1 for line in lines if "VALIDATE done status=not_checked review=no" in line) == 1
    assert not any("text_extraction_started" in line for line in lines)
    assert not any("text_extraction_completed" in line for line in lines)
    assert not any("medicine_prioritization_started" in line and "ai_scan_route" in line for line in lines)
    assert any("SCAN done mode=ensemble runs=0 review=no" in line for line in lines)
    joined_logs = captured.err
    assert "Budi Saputra" not in joined_logs
    assert "Influenza" not in joined_logs

    monkeypatch.delenv("AI_PIPELINE_DEBUG_LOGS", raising=False)
    clear_ai_pipeline_debug_logging_cache()


def test_invoke_model_route_returns_text_only_response(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_messages: list[list[object]] = []

    class FakeModel:
        async def ainvoke(self, messages: list[object]) -> AIMessage:
            captured_messages.append(messages)
            return AIMessage(
                content='{"answer":"ok"}',
                usage_metadata={"input_tokens": 11, "output_tokens": 5, "total_tokens": 16},
            )

    monkeypatch.setattr(
        model_invocation_test_module.ModelInvocationTestService,
        "_create_chat_model",
        lambda self, model_spec: FakeModel(),
    )

    response = client.post(
        "/api/test/invoke-model",
        data={
            "message": "hello model",
            "provider": "google_genai",
            "model_name": "gemini-1.5-flash",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["provider"] == "google_genai"
    assert payload["model_name"] == "gemini-1.5-flash"
    assert payload["supports_image_input"] is True
    assert payload["image_supplied"] is False
    assert payload["raw_text"] == '{"answer":"ok"}'
    assert payload["parsed_json"] == {"answer": "ok"}
    assert payload["prompt_tokens"] == 11
    assert payload["completion_tokens"] == 5
    assert payload["total_tokens"] == 16
    assert len(captured_messages) == 1
    human_message = captured_messages[0][0]
    assert getattr(human_message, "content") == "hello model"


def test_invoke_model_route_supports_optional_image(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_messages: list[list[object]] = []

    class FakeModel:
        async def ainvoke(self, messages: list[object]) -> AIMessage:
            captured_messages.append(messages)
            return AIMessage(content="not json")

    monkeypatch.setattr(
        model_invocation_test_module.ModelInvocationTestService,
        "_create_chat_model",
        lambda self, model_spec: FakeModel(),
    )

    response = client.post(
        "/api/test/invoke-model",
        data={
            "message": "describe image",
            "provider": "google_genai",
            "model_name": "gemini-1.5-flash",
        },
        files={"image": ("scan.png", b"abc", "image/png")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["image_supplied"] is True
    assert payload["parsed_json"] is None
    assert len(captured_messages) == 1
    human_message = captured_messages[0][0]
    content = getattr(human_message, "content")
    assert isinstance(content, list)
    text_block = get_message_block(content, 0)
    image_block = get_message_block(content, 1)
    assert text_block["type"] == "text"
    assert text_block["text"] == "describe image"
    assert image_block["type"] == "image_url"
    image_url = cast(dict[str, Any], image_block["image_url"])
    assert image_url["url"] == "data:image/png;base64,YWJj"


def test_invoke_model_route_uses_google_async_compat_fallback(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_requests: list[dict[str, Any]] = []
    captured_messages: list[list[object]] = []

    class FakeAsyncClient:
        async def generate_content(self, **kwargs: Any) -> AIMessage:
            captured_requests.append(kwargs)
            return AIMessage(
                content='{"answer":"compat-ok"}',
                usage_metadata={"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
            )

    class FakeModel:
        def __init__(self) -> None:
            self.async_client = FakeAsyncClient()
            self.timeout = 9
            self.default_metadata = [("x-test-header", "1")]

        async def ainvoke(self, messages: list[object]) -> AIMessage:
            raise TypeError(
                "GenerativeServiceAsyncClient.generate_content() got an unexpected "
                "keyword argument 'max_retries'"
            )

        def _prepare_request(self, messages: list[object]) -> dict[str, Any]:
            captured_messages.append(messages)
            return {"messages": messages}

    monkeypatch.setattr(
        model_invocation_test_module.ModelInvocationTestService,
        "_create_chat_model",
        lambda self, model_spec: FakeModel(),
    )

    response = client.post(
        "/api/test/invoke-model",
        data={
            "message": "hello model",
            "provider": "google_genai",
            "model_name": "gemini-1.5-flash",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["raw_text"] == '{"answer":"compat-ok"}'
    assert payload["parsed_json"] == {"answer": "compat-ok"}
    assert payload["total_tokens"] == 10
    assert len(captured_messages) == 1
    assert len(captured_requests) == 1
    assert captured_requests[0]["request"]["messages"] == captured_messages[0]
    assert captured_requests[0]["timeout"] == 9
    assert captured_requests[0]["metadata"] == [("x-test-header", "1")]
    assert "max_retries" not in captured_requests[0]


def test_invoke_model_route_rejects_blank_message(client: TestClient) -> None:
    response = client.post(
        "/api/test/invoke-model",
        data={
            "message": "   ",
            "provider": "google_genai",
            "model_name": "gemini-1.5-flash",
        },
    )

    assert response.status_code == 422
    assert response.json() == {
        "message": "Input validation failed.",
        "errors": [
            {
                "field": "message",
                "message": "Message must not be empty.",
            }
        ],
    }


def test_invoke_model_route_rejects_unknown_provider(client: TestClient) -> None:
    response = client.post(
        "/api/test/invoke-model",
        data={
            "message": "hello",
            "provider": "not_real",
            "model_name": "whatever",
        },
    )

    assert response.status_code == 422
    assert response.json() == {
        "message": "Input validation failed.",
        "errors": [
            {
                "field": "provider",
                "message": "Provider is not registered.",
            }
        ],
    }


def test_invoke_model_route_rejects_image_for_non_image_provider(client: TestClient) -> None:
    response = client.post(
        "/api/test/invoke-model",
        data={
            "message": "hello",
            "provider": "mistralai",
            "model_name": "mistral-small-latest",
        },
        files={"image": ("scan.png", b"abc", "image/png")},
    )

    assert response.status_code == 422
    assert response.json() == {
        "message": "Input validation failed.",
        "errors": [
            {
                "field": "image",
                "message": "This provider does not support image input.",
            }
        ],
    }


def test_invoke_model_route_surfaces_missing_provider_credential(client: TestClient) -> None:
    response = client.post(
        "/api/test/invoke-model",
        data={
            "message": "hello",
            "provider": "google_genai",
            "model_name": "gemini-1.5-flash",
        },
    )

    assert response.status_code == 500
    assert response.json() == {
        "message": "Input validation failed.",
        "errors": [
            {
                "field": "non_field",
                "message": "GOOGLE_API_KEY is not configured.",
            }
        ],
    }


def test_invoke_model_route_returns_null_parsed_json_for_non_json_text(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeModel:
        async def ainvoke(self, messages: list[object]) -> AIMessage:
            del messages
            return AIMessage(content="plain text output")

    monkeypatch.setattr(
        model_invocation_test_module.ModelInvocationTestService,
        "_create_chat_model",
        lambda self, model_spec: FakeModel(),
    )

    response = client.post(
        "/api/test/invoke-model",
        data={
            "message": "hello model",
            "provider": "google_genai",
            "model_name": "gemini-1.5-flash",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["raw_text"] == "plain text output"
    assert payload["parsed_json"] is None


def test_invoke_model_route_returns_clear_provider_runtime_error(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeModel:
        async def ainvoke(self, messages: list[object]) -> AIMessage:
            del messages
            raise RuntimeError("provider boom")

    monkeypatch.setattr(
        model_invocation_test_module.ModelInvocationTestService,
        "_create_chat_model",
        lambda self, model_spec: FakeModel(),
    )

    response = client.post(
        "/api/test/invoke-model",
        data={
            "message": "hello model",
            "provider": "google_genai",
            "model_name": "gemini-1.5-flash",
        },
    )

    assert response.status_code == 502
    assert response.json() == {
        "message": "AI model invocation failed.",
        "errors": [
            {
                "field": "non_field",
                "message": "provider boom",
            }
        ],
    }


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


def test_start_scan_documents_job_returns_initial_progress_snapshot(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = register_and_login(client, email="scan-job@example.com", full_name="Siti Aminah")

    async def fake_run_scan_documents_job(scan_log_id: int, user_id: int) -> None:
        del scan_log_id, user_id
        return None

    monkeypatch.setattr(ai_route, "run_scan_documents_job", fake_run_scan_documents_job)

    response = client.post(
        "/api/ai/scan-documents/jobs",
        headers={"Authorization": f"Bearer {token}"},
        files=[("files", ("scan.png", b"abc", "image/png"))],
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["scan_log_id"] > 0
    assert payload["progress"] == {
        "stage": "queued",
        "title": "Preparing scan",
        "detail": "Upload complete. Starting server-side processing.",
        "percent": 15,
        "status": "pending",
        "image_count": 1,
        "total_runs": None,
        "completed_runs": None,
        "successful_runs": None,
        "medicine_count": None,
    }

    scan_log = get_scan_log_by_id(payload["scan_log_id"])
    assert scan_log is not None
    assert scan_log.progress_payload == payload["progress"]


def test_get_scan_documents_job_status_returns_succeeded_result_for_owner(client: TestClient) -> None:
    token = register_and_login(client, email="scan-job-status@example.com", full_name="Siti Aminah")
    user_id = get_user_id_from_token(client, token)
    session_factory = get_session_factory()

    with session_factory() as db:
        repository = ScanLogRepository(db)
        scan_log = repository.create_pending(
            user_id=user_id,
            file_records=[
                ScanLogFileCreate(
                    original_filename="scan.png",
                    original_mime_type="image/png",
                    original_byte_size=3,
                    original_sha256_hash=hashlib.sha256(b"abc").hexdigest(),
                    original_content=b"abc",
                )
            ],
            progress_payload={
                "stage": "prioritizing_medicines",
                "title": "Finalizing results",
                "detail": "Applying final checks before review.",
                "percent": 97,
                "status": "running",
                "image_count": 1,
                "total_runs": 4,
                "completed_runs": 4,
                "successful_runs": 4,
                "medicine_count": 1,
            },
        )
        repository.mark_succeeded(
            scan_log,
            validation_payload=VALIDATION_RESULT,
            extraction_payload=SCAN_RESULT,
            primary_model_provider="google_genai",
            primary_model_name="gemini-3.1-flash-lite-preview",
            ai_execution_mode="ensemble",
            extraction_meta_payload={"review_required": False},
        )
        scan_log_id = scan_log.id

    response = client.get(
        f"/api/ai/scan-documents/jobs/{scan_log_id}",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["scan_log_id"] == scan_log_id
    assert payload["progress"]["stage"] == "completed"
    assert payload["progress"]["title"] == "Ready to review"
    assert payload["progress"]["detail"] == "Results are ready for you to review."
    assert payload["progress"]["percent"] == 100
    assert payload["progress"]["status"] == "succeeded"
    assert payload["error_message"] is None
    assert payload["result"]["scan_log_id"] == scan_log_id
    assert payload["result"]["summary"] == SCAN_RESULT["summary"]


def test_get_scan_documents_job_status_keeps_non_terminal_finalizing_title(client: TestClient) -> None:
    token = register_and_login(client, email="scan-job-finalizing@example.com", full_name="Siti Aminah")
    user_id = get_user_id_from_token(client, token)
    session_factory = get_session_factory()

    with session_factory() as db:
        repository = ScanLogRepository(db)
        scan_log = repository.create_pending(
            user_id=user_id,
            file_records=[
                ScanLogFileCreate(
                    original_filename="scan.png",
                    original_mime_type="image/png",
                    original_byte_size=3,
                    original_sha256_hash=hashlib.sha256(b"abc").hexdigest(),
                    original_content=b"abc",
                )
            ],
            progress_payload={
                "stage": "prioritizing_medicines",
                "title": "Finalizing results",
                "detail": "Applying final checks before review.",
                "percent": 97,
                "status": "running",
                "image_count": 1,
                "total_runs": 4,
                "completed_runs": 4,
                "successful_runs": 4,
                "medicine_count": 1,
            },
        )
        scan_log_id = scan_log.id

    response = client.get(
        f"/api/ai/scan-documents/jobs/{scan_log_id}",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["progress"]["stage"] == "prioritizing_medicines"
    assert payload["progress"]["title"] == "Finalizing results"
    assert payload["progress"]["detail"] == "Applying final checks before review."
    assert payload["progress"]["title"] != "Ready to review"
    assert payload["result"] is None


def test_get_scan_documents_job_status_requires_owner(client: TestClient) -> None:
    owner_token = register_and_login(client, email="scan-job-owner@example.com", full_name="Owner User")
    other_token = register_and_login(client, email="scan-job-other@example.com", full_name="Other User")
    owner_user_id = get_user_id_from_token(client, owner_token)
    session_factory = get_session_factory()

    with session_factory() as db:
        repository = ScanLogRepository(db)
        scan_log = repository.create_pending(
            user_id=owner_user_id,
            file_records=[
                ScanLogFileCreate(
                    original_filename="scan.png",
                    original_mime_type="image/png",
                    original_byte_size=3,
                    original_sha256_hash=hashlib.sha256(b"abc").hexdigest(),
                    original_content=b"abc",
                )
            ],
        )
        scan_log_id = scan_log.id

    response = client.get(
        f"/api/ai/scan-documents/jobs/{scan_log_id}",
        headers={"Authorization": f"Bearer {other_token}"},
    )

    assert response.status_code == 404


def test_generate_scan_progress_sse_yields_initial_snapshot_immediately() -> None:
    async def scenario() -> None:
        disconnect_flag = False
        event_hub = ScanProgressEventHub()
        generator = generate_scan_progress_sse(
            scan_log_id=123,
            initial_event="progress",
            initial_payload={
                "scan_log_id": 123,
                "progress": {
                    "stage": "queued",
                    "title": "Preparing scan",
                    "detail": "Upload complete. Starting server-side processing.",
                    "percent": 15,
                    "status": "pending",
                    "image_count": 1,
                    "total_runs": None,
                    "completed_runs": None,
                    "successful_runs": None,
                    "medicine_count": None,
                },
                "result": None,
                "error_message": None,
            },
            initial_terminal=False,
            is_disconnected=lambda: asyncio.sleep(0, result=disconnect_flag),
            event_hub=event_hub,
            heartbeat_interval=60,
            poll_interval=0.05,
        )
        try:
            event_name, payload = await next_sse_message(generator)
            assert event_name == "progress"
            assert payload is not None
            assert payload["scan_log_id"] == 123
            assert payload["progress"]["stage"] == "queued"
            assert event_hub.subscriber_count(123) == 1
        finally:
            await generator.aclose()
        assert event_hub.subscriber_count(123) == 0

    asyncio.run(scenario())


def test_scan_documents_job_events_requires_owner(client: TestClient) -> None:
    owner_token = register_and_login(client, email="scan-job-events-owner@example.com", full_name="Owner User")
    other_token = register_and_login(client, email="scan-job-events-other@example.com", full_name="Other User")
    owner_user_id = get_user_id_from_token(client, owner_token)
    session_factory = get_session_factory()

    with session_factory() as db:
        repository = ScanLogRepository(db)
        scan_log = repository.create_pending(
            user_id=owner_user_id,
            file_records=[
                ScanLogFileCreate(
                    original_filename="scan.png",
                    original_mime_type="image/png",
                    original_byte_size=3,
                    original_sha256_hash=hashlib.sha256(b"abc").hexdigest(),
                    original_content=b"abc",
                )
            ],
        )
        scan_log_id = scan_log.id

    response = client.get(
        f"/api/ai/scan-documents/jobs/{scan_log_id}/events",
        headers={"Authorization": f"Bearer {other_token}"},
    )

    assert response.status_code == 404


def test_generate_scan_progress_sse_yields_progress_then_completion() -> None:
    async def scenario() -> None:
        event_hub = ScanProgressEventHub()
        generator = generate_scan_progress_sse(
            scan_log_id=321,
            initial_event="progress",
            initial_payload={
                "scan_log_id": 321,
                "progress": {
                    "stage": "queued",
                    "title": "Preparing scan",
                    "detail": "Upload complete. Starting server-side processing.",
                    "percent": 15,
                    "status": "pending",
                    "image_count": 1,
                    "total_runs": None,
                    "completed_runs": None,
                    "successful_runs": None,
                    "medicine_count": None,
                },
                "result": None,
                "error_message": None,
            },
            initial_terminal=False,
            is_disconnected=lambda: asyncio.sleep(0, result=False),
            event_hub=event_hub,
            heartbeat_interval=60,
            poll_interval=0.05,
        )

        try:
            initial_event_name, _initial_payload = await next_sse_message(generator)
            assert initial_event_name == "progress"

            event_hub.publish(
                321,
                event="progress",
                payload={
                    "scan_log_id": 321,
                    "progress": {
                        "stage": "extracting",
                        "title": "Extracting medical data",
                        "detail": "Running 1 of 4 AI extraction passes.",
                        "percent": 50,
                        "status": "running",
                        "image_count": 1,
                        "total_runs": 4,
                        "completed_runs": 1,
                        "successful_runs": 1,
                        "medicine_count": None,
                    },
                    "result": None,
                    "error_message": None,
                },
            )

            progress_event_name, progress_payload = await next_sse_message(generator)
            assert progress_event_name == "progress"
            assert progress_payload is not None
            assert progress_payload["progress"]["stage"] == "extracting"
            assert progress_payload["progress"]["completed_runs"] == 1

            event_hub.publish(
                321,
                event="completed",
                payload={
                    "scan_log_id": 321,
                    "progress": {
                        "stage": "completed",
                        "title": "Ready to review",
                        "detail": "Results are ready for you to review.",
                        "percent": 100,
                        "status": "succeeded",
                        "image_count": 1,
                        "total_runs": 4,
                        "completed_runs": 4,
                        "successful_runs": 4,
                        "medicine_count": 1,
                    },
                    "result": {
                        "scan_log_id": 321,
                        **SCAN_RESULT,
                    },
                    "error_message": None,
                },
                terminal=True,
            )

            completed_event_name, completed_payload = await next_sse_message(generator)
            assert completed_event_name == "completed"
            assert completed_payload is not None
            assert completed_payload["progress"]["status"] == "succeeded"
            assert completed_payload["result"]["scan_log_id"] == 321

            with pytest.raises(StopAsyncIteration):
                await asyncio.wait_for(anext(generator), timeout=0.5)
        finally:
            await generator.aclose()

        assert event_hub.subscriber_count(321) == 0

    asyncio.run(scenario())


def test_generate_scan_progress_sse_yields_cancelled_and_stops() -> None:
    async def scenario() -> None:
        event_hub = ScanProgressEventHub()
        generator = generate_scan_progress_sse(
            scan_log_id=456,
            initial_event="progress",
            initial_payload={
                "scan_log_id": 456,
                "progress": {
                    "stage": "queued",
                    "title": "Preparing scan",
                    "detail": "Upload complete. Starting server-side processing.",
                    "percent": 15,
                    "status": "pending",
                    "image_count": 1,
                    "total_runs": None,
                    "completed_runs": None,
                    "successful_runs": None,
                    "medicine_count": None,
                },
                "result": None,
                "error_message": None,
            },
            initial_terminal=False,
            is_disconnected=lambda: asyncio.sleep(0, result=False),
            event_hub=event_hub,
            heartbeat_interval=60,
            poll_interval=0.05,
        )

        try:
            _initial_event_name, _initial_payload = await next_sse_message(generator)
            event_hub.publish(
                456,
                event="cancelled",
                payload={
                    "scan_log_id": 456,
                    "progress": {
                        "stage": "failed",
                        "title": "Scan cancelled",
                        "detail": "Scan was cancelled by the user.",
                        "percent": 97,
                        "status": "failed",
                        "image_count": 1,
                        "total_runs": None,
                        "completed_runs": None,
                        "successful_runs": None,
                        "medicine_count": None,
                    },
                    "result": None,
                    "error_message": "Scan was cancelled by the user.",
                },
                terminal=True,
            )

            terminal_event_name, terminal_payload = await next_sse_message(generator)
            assert terminal_event_name == "cancelled"
            assert terminal_payload is not None
            assert terminal_payload["progress"]["status"] == "failed"

            with pytest.raises(StopAsyncIteration):
                await asyncio.wait_for(anext(generator), timeout=0.5)
        finally:
            await generator.aclose()

        assert event_hub.subscriber_count(456) == 0

    asyncio.run(scenario())


def test_generate_scan_progress_sse_exits_on_disconnect() -> None:
    async def scenario() -> None:
        event_hub = ScanProgressEventHub()
        disconnect_flag = False

        async def is_disconnected() -> bool:
            return disconnect_flag

        generator = generate_scan_progress_sse(
            scan_log_id=654,
            initial_event="progress",
            initial_payload={
                "scan_log_id": 654,
                "progress": {
                    "stage": "queued",
                    "title": "Preparing scan",
                    "detail": "Upload complete. Starting server-side processing.",
                    "percent": 15,
                    "status": "pending",
                    "image_count": 1,
                    "total_runs": None,
                    "completed_runs": None,
                    "successful_runs": None,
                    "medicine_count": None,
                },
                "result": None,
                "error_message": None,
            },
            initial_terminal=False,
            is_disconnected=is_disconnected,
            event_hub=event_hub,
            heartbeat_interval=60,
            poll_interval=0.05,
        )

        try:
            _initial_event_name, _initial_payload = await next_sse_message(generator)
            assert event_hub.subscriber_count(654) == 1
            disconnect_flag = True

            with pytest.raises(StopAsyncIteration):
                await asyncio.wait_for(anext(generator), timeout=0.5)
        finally:
            await generator.aclose()

        assert event_hub.subscriber_count(654) == 0

    asyncio.run(scenario())


def test_scan_documents_job_events_returns_terminal_snapshot_for_owner(client: TestClient) -> None:
    token = register_and_login(client, email="scan-job-events-terminal@example.com", full_name="Siti Aminah")
    user_id = get_user_id_from_token(client, token)
    session_factory = get_session_factory()

    with session_factory() as db:
        repository = ScanLogRepository(db)
        scan_log = repository.create_pending(
            user_id=user_id,
            file_records=[
                ScanLogFileCreate(
                    original_filename="scan.png",
                    original_mime_type="image/png",
                    original_byte_size=3,
                    original_sha256_hash=hashlib.sha256(b"abc").hexdigest(),
                    original_content=b"abc",
                )
            ],
            progress_payload={
                "stage": "queued",
                "title": "Preparing scan",
                "detail": "Upload complete. Starting server-side processing.",
                "percent": 15,
                "status": "pending",
                "image_count": 1,
                "total_runs": None,
                "completed_runs": None,
                "successful_runs": None,
                "medicine_count": None,
            },
        )
        cancelled_scan_log = repository.mark_cancelled(scan_log)
        scan_log_id = cancelled_scan_log.id

    response = client.get(
        f"/api/ai/scan-documents/jobs/{scan_log_id}/events",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    event_name, payload = parse_sse_message(response.text)
    assert event_name == "cancelled"
    assert payload is not None
    assert payload["scan_log_id"] == scan_log_id
    assert payload["progress"]["status"] == "failed"


def test_medical_document_processor_reports_progress_during_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    medical_processor = MedicalDocumentProcessor(settings=build_settings())
    progress_events: list[dict[str, Any]] = []

    async def fake_run_single_inference_trace(**kwargs: object) -> ProducerInferenceTrace:
        producer_rank = cast(int, kwargs["producer_rank"])
        run_index = cast(int, kwargs["run_index"])
        await asyncio.sleep(0.01 if run_index == 0 else 0.02)
        return ProducerInferenceTrace(
            producer_id=f"producer-{producer_rank}",
            provider="google_genai",
            model=f"model-{producer_rank}",
            producer_rank=producer_rank,
            run_index=run_index,
            temperature=0.2,
            success=True,
            call_kind="scan_extraction",
            payload=AIMedicalScanResponse.model_validate(SCAN_RESULT),
        )

    async def fake_fuse(self: MedicalExtractionFusionEngine, **kwargs: object) -> MedicalExtractionOutcome:
        del self
        all_traces = cast(list[ProducerInferenceTrace], kwargs["all_traces"])
        return MedicalExtractionOutcome(
            result=AIMedicalScanResponse.model_validate(SCAN_RESULT),
            overall_confidence=0.9,
            field_confidence={},
            review_required=False,
            review_reasons=[],
            calibration_version="test",
            ai_execution_mode="ensemble",
            primary_model_provider="google_genai",
            primary_model_name="gemini-3.1-flash-lite-preview",
            ai_runs=all_traces,
            extraction_meta_payload={"review_required": False},
        )

    monkeypatch.setattr(
        medical_processor,
        "_get_enabled_model_specs",
        lambda: [
            AIModelSpec(provider="google_genai", model="gemini-3.1-flash-lite-preview"),
            AIModelSpec(provider="google_genai", model="gemini-3-flash-preview"),
        ],
    )
    monkeypatch.setattr(medical_processor, "_run_single_inference_trace", fake_run_single_inference_trace)
    monkeypatch.setattr(MedicalExtractionFusionEngine, "fuse", fake_fuse)

    asyncio.run(
        medical_processor.process(
            images=[DocumentImageInput(filename="scan.png", mime_type="image/png", base64_data="YWJj")],
            progress_callback=lambda payload: progress_events.append(payload),
        )
    )

    assert progress_events[0]["event"] == "extraction_started"
    run_events = [event for event in progress_events if event["event"] == "extraction_run_completed"]
    assert [event["completed_runs"] for event in run_events] == [1, 2, 3, 4]
    assert run_events[-1]["successful_runs"] == 4
    assert progress_events[-2]["event"] == "fusion_started"
    assert progress_events[-1]["event"] == "fusion_completed"


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
    rendered_user_prompt = cast(str, text_block["text"])
    rendered_system_prompt = str(messages[0].content)

    assert "Return only valid JSON" in rendered_system_prompt
    assert "Analyze the 1 uploaded medical document image(s) as one claim packet." in rendered_user_prompt
    assert "PRIMARY TASK" in rendered_user_prompt
    assert "FINANCIAL EXTRACTION RULES" in rendered_user_prompt
    assert "AMBIGUITY / ANTI-INFERENCE RULES" in rendered_user_prompt
    assert "Extract summary.total_amount as the grand total for the full uploaded claim packet" in rendered_user_prompt
    assert "Do not duplicate pharmacy or medicine spending across medicines and other_charges." in rendered_user_prompt
    assert "Do not infer gender from the patient's name" in rendered_user_prompt
    assert "If handwritten free text is too ambiguous to support one clear reading, use null instead of guessing." in rendered_user_prompt
    assert "Prefer the consultation or treatment date" in rendered_user_prompt
    assert '["Siti Aminah", "Budi Saputra"]' in rendered_user_prompt
    assert image_block["type"] == "image_url"
    assert cast(str, image_url["url"]).startswith("data:image/png;base64,")
    assert image_url["url"] == "data:image/png;base64,YWJj"


def test_document_validation_prompt_defines_provider_identity_not_insurer_branding() -> None:
    validation_processor = DocumentValidationProcessor(settings=build_settings())
    prompt_bundle = validation_processor._load_prompt_bundle()

    messages = validation_processor._build_messages(
        images=[
            DocumentImageInput(
                filename="scan.jpg",
                mime_type="image/jpeg",
                base64_data="YWJj",
            )
        ],
        patient_name_candidates=["Nathan Benedict Kristian"],
        expected_patient_name="Nathan Benedict Kristian",
        prompt_bundle=prompt_bundle,
    )

    text_block = get_message_block(messages[1].content, 0)
    rendered_user_prompt = cast(str, text_block["text"])

    assert "medical provider identity" in rendered_user_prompt
    assert "hospital, clinic, lab, pharmacy, medical practice, or treating doctor" in rendered_user_prompt
    assert "SIP/license number" in rendered_user_prompt
    assert "official stamp" in rendered_user_prompt
    assert "insurer, payer, or TPA branding alone" in rendered_user_prompt
    assert "mandiri inhealth" in rendered_user_prompt


def test_financial_packet_judge_prompt_emphasizes_terbilang_and_force_pick() -> None:
    medical_processor = MedicalDocumentProcessor(settings=build_settings())
    prompt_bundle = medical_processor._load_prompt_bundle(ai_service_module.FINANCIAL_PACKET_JUDGE_PROMPT_FILE_PATH)

    assert "remaining ledger candidates are close and visually ambiguous" in prompt_bundle.system_template
    assert "Always choose one provided candidate index unless the packet is genuinely unreadable." in prompt_bundle.user_template
    assert "Use terbilang or amount-in-words text as a consistency check" in prompt_bundle.user_template
    assert "Prioritize evidence in this order: visible numeric receipt amounts, terbilang consistency, then packet coherence across pages." in prompt_bundle.user_template
    assert 'Do not over-weight weak handwritten description text such as "untuk pembayaran"' in prompt_bundle.user_template


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

    assert result.result.summary.detected_documents == SCAN_RESULT["summary"]["detected_documents"]
    assert result.result.summary.service_date == SCAN_RESULT["summary"]["service_date"]
    assert result.result.summary.total_amount == SCAN_RESULT["summary"]["total_amount"]
    assert result.result.summary.medical_facility_name == "rumah sakit sehat sentosa"
    assert result.result.patient_info.model_dump() == SCAN_RESULT["patient_info"]
    assert result.result.medical_context.model_dump() == SCAN_RESULT["medical_context"]
    assert result.result.financial_recap.model_dump() == SCAN_RESULT["financial_recap"]
    assert result.ai_execution_mode == "ensemble"
    assert result.overall_confidence is not None
    service_date_meta = result.field_confidence["summary.service_date"]
    assert service_date_meta["vote_count"] == 2
    assert service_date_meta["vote_share"] == 1.0
    assert service_date_meta["producer_count"] == 1
    assert service_date_meta["producer_share"] == 1.0
    assert service_date_meta["margin"] == 1.0
    assert service_date_meta["calibration_version"] == "self_consistency_logistic_v1"
    assert "raw_confidence" not in service_date_meta
    assert "support_count" not in service_date_meta
    assert "producer_agreement" not in service_date_meta
    assert "consensus_features" not in service_date_meta
    assert result.extraction_meta_payload["calibration_artifact_status"] == "bootstrap_unfitted"
    assert result.extraction_meta_payload["calibration_artifact_warning"] is not None


def test_process_fails_closed_when_calibration_artifact_override_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    medical_processor = MedicalDocumentProcessor(settings=build_settings())

    class FakeModel:
        async def ainvoke(self, messages: list[object]) -> AIMessage:
            del messages
            return AIMessage(content=json.dumps(SCAN_RESULT))

    missing_artifact = tmp_path / "missing-artifact.json"
    monkeypatch.setenv("AI_ENSEMBLE_CALIBRATION_ARTIFACT", str(missing_artifact))
    clear_self_consistency_calibration_cache()
    monkeypatch.setattr(medical_processor, "_create_chat_model", lambda model_spec: FakeModel())
    monkeypatch.setattr(
        medical_processor,
        "_get_enabled_model_specs",
        lambda: [AIModelSpec(provider="google_genai", model="gemini-1.5-flash")],
    )

    with pytest.raises(RuntimeError, match="calibration artifact is missing"):
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

    monkeypatch.delenv("AI_ENSEMBLE_CALIBRATION_ARTIFACT", raising=False)
    clear_self_consistency_calibration_cache()


def test_process_marks_fitted_calibration_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    medical_processor = MedicalDocumentProcessor(settings=build_settings())

    class FakeModel:
        async def ainvoke(self, messages: list[object]) -> AIMessage:
            del messages
            return AIMessage(content=json.dumps(SCAN_RESULT))

    fitted_artifact = tmp_path / "fitted-artifact.json"
    fitted_artifact.write_text(
        json.dumps(
            {
                "calibration_version": "self_consistency_logistic_v1",
                "feature_names": [
                    "vote_share",
                    "margin",
                    "producer_share",
                    "judge_used",
                    "is_critical",
                    "is_semantic",
                ],
                "coefficients": [3.8, 1.6, 1.2, -0.35, 0.15, -0.2],
                "intercept": -3.0,
                "training_claim_count": 12,
                "created_at": "2026-05-05T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("AI_ENSEMBLE_CALIBRATION_ARTIFACT", str(fitted_artifact))
    clear_self_consistency_calibration_cache()
    monkeypatch.setattr(medical_processor, "_create_chat_model", lambda model_spec: FakeModel())
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

    assert result.extraction_meta_payload["calibration_artifact_status"] == "fitted"
    assert result.extraction_meta_payload["calibration_artifact_warning"] is None

    monkeypatch.delenv("AI_ENSEMBLE_CALIBRATION_ARTIFACT", raising=False)
    clear_self_consistency_calibration_cache()


def test_process_uses_producer_count_tiebreak_before_judge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    medical_processor = MedicalDocumentProcessor(settings=build_settings())
    call_counts: dict[str, int] = {}

    class FakeModel:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        async def ainvoke(self, messages: list[object]) -> AIMessage:
            del messages
            current = call_counts.get(self.model_name, 0)
            call_counts[self.model_name] = current + 1
            address = "JL. Alpha No.1"
            if self.model_name == "model-a":
                address = "JL. Alpha No.1"
            elif self.model_name == "model-b":
                address = "JL. Beta No.2" if current == 0 else "JL. Gamma No.3"
            elif self.model_name == "model-c":
                address = "JL. Beta No.2" if current == 0 else "JL. Delta No.4"
            payload = {
                **SCAN_RESULT,
                "patient_info": {
                    **SCAN_RESULT["patient_info"],
                    "address": address,
                },
            }
            return AIMessage(content=json.dumps(payload))

    async def should_not_be_called(
        field_path: str,
        images: list[object],
        candidates: list[dict[str, object]],
    ) -> tuple[str | None, str | None]:
        del images, candidates
        raise AssertionError(f"Judge should not run for producer-count tiebreak: {field_path}")

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

    address_meta = result.field_confidence["patient_info.address"]
    assert result.result.patient_info.address == "JL. Beta No.2"
    assert address_meta["judge_used"] is False
    assert address_meta["vote_count"] == 2
    assert address_meta["producer_count"] == 2
    assert address_meta["vote_share"] == pytest.approx(0.3333, rel=1e-6)


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

    assert result.result.model_dump() == {
        **SCAN_RESULT,
        "summary": {
            **SCAN_RESULT["summary"],
            "medical_facility_name": "rumah sakit sehat sentosa",
        },
    }
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

    messages = [
        SystemMessage(content="system prompt"),
        HumanMessage(
            content=[
                {"type": "text", "text": "user prompt"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,YWJj"}},
            ]
        ),
    ]

    response, tracking_payload = asyncio.run(
        medical_processor._ainvoke_with_tracking(
            FakeModel(),
            messages,
            provider="google_genai",
            model_name="gemini-test",
            call_kind="scan_extraction",
            success_context={"request_context_payload": {"image_count": 1}},
            prompt_file_path=Path("test_prompt.yaml"),
        )
    )

    assert response.content == json.dumps(SCAN_RESULT)
    assert tracking_payload["prompt_tokens"] == 123
    assert tracking_payload["completion_tokens"] == 45
    assert tracking_payload["total_tokens"] == 168
    assert isinstance(tracking_payload["ai_api_call_log_id"], int)

    with session_factory() as db:
        rows = db.execute(select(AIApiCallLog)).scalars().all()
        assert len(rows) == 1
        assert rows[0].provider == "google_genai"
        assert rows[0].call_kind == "scan_extraction"
        assert rows[0].prompt_tokens == 123
        assert rows[0].completion_tokens == 45
        assert rows[0].total_tokens == 168
        assert rows[0].prompt_file_path == "test_prompt.yaml"
        assert rows[0].raw_response_text == json.dumps(SCAN_RESULT)
        assert rows[0].response_content_payload == json.dumps(SCAN_RESULT)
        assert isinstance(rows[0].prompt_payload, list)
        assert "YWJj" not in json.dumps(rows[0].prompt_payload)
        assert rows[0].prompt_payload[1]["content"][1]["image_url"]["redacted"] is True


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
        assert rows[0].prompt_payload is not None
        assert rows[0].raw_response_text is None


def test_ainvoke_with_tracking_uses_google_async_compat_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    medical_processor = MedicalDocumentProcessor(settings=build_settings())
    session_factory = build_session_factory_with_ai_models([])
    captured_requests: list[dict[str, Any]] = []

    class FakeAsyncClient:
        async def generate_content(self, **kwargs: Any) -> AIMessage:
            captured_requests.append(kwargs)
            return AIMessage(
                content=json.dumps(SCAN_RESULT),
                usage_metadata={"input_tokens": 19, "output_tokens": 8, "total_tokens": 27},
            )

    class FakeModel:
        def __init__(self) -> None:
            self.async_client = FakeAsyncClient()
            self.timeout = 5
            self.default_metadata = [("x-ai-call", "compat")]

        async def ainvoke(self, messages: list[object]) -> AIMessage:
            assert len(messages) == 1
            raise TypeError(
                "GenerativeServiceAsyncClient.generate_content() got an unexpected "
                "keyword argument 'max_retries'"
            )

        def _prepare_request(self, messages: list[object]) -> dict[str, Any]:
            return {"messages": messages}

    monkeypatch.setattr("app.services.ai_service.get_session_factory", lambda: session_factory)

    response, tracking_payload = asyncio.run(
        medical_processor._ainvoke_with_tracking(
            FakeModel(),
            [object()],
            provider="google_genai",
            model_name="gemini-test",
            call_kind="scan_extraction",
            success_context={"request_context_payload": {"image_count": 1}},
        )
    )

    assert response.content == json.dumps(SCAN_RESULT)
    assert tracking_payload["prompt_tokens"] == 19
    assert tracking_payload["completion_tokens"] == 8
    assert tracking_payload["total_tokens"] == 27
    assert len(captured_requests) == 1
    assert captured_requests[0]["timeout"] == 5
    assert captured_requests[0]["metadata"] == [("x-ai-call", "compat")]
    assert "max_retries" not in captured_requests[0]

    with session_factory() as db:
        rows = db.execute(select(AIApiCallLog)).scalars().all()
        assert len(rows) == 1
        assert rows[0].success is True
        assert rows[0].provider == "google_genai"
        assert rows[0].prompt_tokens == 19
        assert rows[0].completion_tokens == 8
        assert rows[0].total_tokens == 27


def test_run_model_inference_persists_parsed_scan_extraction_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    medical_processor = MedicalDocumentProcessor(settings=build_settings())
    session_factory = build_session_factory_with_ai_models([])

    class FakeModel:
        async def ainvoke(self, messages: list[object]) -> AIMessage:
            assert len(messages) == 2
            return AIMessage(content=json.dumps(SCAN_RESULT))

    monkeypatch.setattr("app.services.ai_service.get_session_factory", lambda: session_factory)
    monkeypatch.setattr(medical_processor, "_create_chat_model", lambda model_spec: FakeModel())

    prompt_bundle = medical_processor._load_prompt_bundle()
    result, tracking_payload = asyncio.run(
        medical_processor._run_model_inference(
            model_spec=AIModelSpec(provider="google_genai", model="gemini-test"),
            images=[DocumentImageInput(filename="scan.png", mime_type="image/png", base64_data="YWJj")],
            patient_name_candidates=["Budi Saputra"],
            validation_hints={},
            prompt_bundle=prompt_bundle,
            ai_call_context={"scan_log_id": 1, "user_id": 2},
        )
    )

    assert result.summary.medical_facility_name == "RS Sehat Sentosa"
    assert isinstance(tracking_payload["ai_api_call_log_id"], int)

    with session_factory() as db:
        row = db.execute(select(AIApiCallLog)).scalar_one()
        assert row.call_kind == "scan_extraction"
        assert row.parsed_response_payload == SCAN_RESULT
        assert row.response_validation_error is None
        assert row.result_context_payload == {"call_kind": "scan_extraction", "schema_valid": True}


def test_document_validation_inference_persists_parsed_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validation_processor = DocumentValidationProcessor(settings=build_settings())
    session_factory = build_session_factory_with_ai_models([])

    class FakeModel:
        async def ainvoke(self, messages: list[object]) -> AIMessage:
            assert len(messages) == 2
            return AIMessage(content=json.dumps(VALIDATION_RESULT))

    monkeypatch.setattr("app.services.ai_service.get_session_factory", lambda: session_factory)
    monkeypatch.setattr(validation_processor, "_create_chat_model", lambda model_spec: FakeModel())

    prompt_bundle = validation_processor._load_prompt_bundle()
    result = asyncio.run(
        validation_processor._run_model_inference(
            model_spec=AIModelSpec(provider="google_genai", model="gemini-test"),
            images=[DocumentImageInput(filename="scan.jpg", mime_type="image/jpeg", base64_data="YWJj")],
            prompt_bundle=prompt_bundle,
            patient_name_candidates=["Budi Saputra"],
            expected_patient_name="Budi Saputra",
            ai_call_context={"scan_log_id": 3, "user_id": 4},
        )
    )

    assert result.images[0].filename == "scan.jpg"

    with session_factory() as db:
        row = db.execute(select(AIApiCallLog)).scalar_one()
        assert row.call_kind == "scan_validation"
        assert row.parsed_response_payload == VALIDATION_RESULT
        assert row.response_validation_error is None
        assert row.result_context_payload == {"call_kind": "scan_validation", "schema_valid": True}


def test_judge_field_conflict_persists_selection_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    medical_processor = MedicalDocumentProcessor(settings=build_settings())
    session_factory = build_session_factory_with_ai_models([])

    class FakeJudgeModel:
        async def ainvoke(self, messages: list[object]) -> AIMessage:
            assert len(messages) == 2
            return AIMessage(content=json.dumps({"selected_candidate_index": 0, "reason": "Prefer stronger support"}))

    monkeypatch.setattr("app.services.ai_service.get_session_factory", lambda: session_factory)
    monkeypatch.setattr(medical_processor, "_create_chat_model", lambda model_spec: FakeJudgeModel())
    monkeypatch.setattr(
        medical_processor,
        "_get_enabled_model_specs",
        lambda: [AIModelSpec(provider="google_genai", model="judge-model")],
    )

    selected_key, reason = asyncio.run(
        medical_processor._judge_field_conflict(
            "patient_info.address",
            [DocumentImageInput(filename="scan.png", mime_type="image/png", base64_data="YWJj")],
            [
                {"raw_value": "JL. Alpha No.1", "normalized_key": "alpha", "run_count": 2, "producers": {"p1", "p2"}},
                {"raw_value": "JL. Beta No.2", "normalized_key": "beta", "run_count": 1, "producers": {"p3"}},
            ],
            ai_call_context={"scan_log_id": 99, "user_id": 7},
        )
    )

    assert selected_key == "alpha"
    assert reason == "Prefer stronger support"

    with session_factory() as db:
        row = db.execute(select(AIApiCallLog)).scalar_one()
        assert row.call_kind == "field_judge"
        assert row.prompt_file_path.endswith("medical_document_field_judge.yaml")
        assert row.request_context_payload == {
            "field_path": "patient_info.address",
            "candidate_count": 2,
            "image_count": 1,
        }
        assert row.parsed_response_payload == {"selected_candidate_index": 0, "reason": "Prefer stronger support"}
        assert row.result_context_payload == {
            "field_path": "patient_info.address",
            "selected_candidate_index": 0,
            "selected_candidate_key": "alpha",
            "selected_candidate_valid": True,
            "abstained": False,
        }
        assert "YWJj" not in json.dumps(row.prompt_payload)


def test_financial_packet_judge_persists_selection_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    medical_processor = MedicalDocumentProcessor(settings=build_settings())
    session_factory = build_session_factory_with_ai_models([])

    class FakeJudgeModel:
        async def ainvoke(self, messages: list[object]) -> AIMessage:
            assert len(messages) == 2
            return AIMessage(content=json.dumps({"selected_candidate_index": 1, "reason": "Receipt-backed total is stronger"}))

    monkeypatch.setattr("app.services.ai_service.get_session_factory", lambda: session_factory)
    monkeypatch.setattr(medical_processor, "_create_chat_model", lambda model_spec: FakeJudgeModel())
    monkeypatch.setattr(
        medical_processor,
        "_get_enabled_model_specs",
        lambda: [AIModelSpec(provider="google_genai", model="judge-model")],
    )

    selected_index, reason = asyncio.run(
        medical_processor._judge_financial_packet_conflict(
            [DocumentImageInput(filename="scan.png", mime_type="image/png", base64_data="YWJj")],
            [{"grand_total": 125000}, {"grand_total": 126000}],
            ai_call_context={"scan_log_id": 101, "user_id": 5},
        )
    )

    assert selected_index == 1
    assert reason == "Receipt-backed total is stronger"

    with session_factory() as db:
        row = db.execute(select(AIApiCallLog)).scalar_one()
        assert row.call_kind == "field_judge"
        assert row.prompt_file_path.endswith("financial_packet_judge.yaml")
        assert row.request_context_payload == {
            "field_path": "financial.packet",
            "candidate_count": 2,
            "image_count": 1,
        }
        assert row.result_context_payload == {
            "field_path": "financial.packet",
            "selected_candidate_index": 1,
            "selected_candidate_valid": True,
            "abstained": False,
        }


def test_medicine_relevance_judge_persists_parsed_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    judge = medicine_relevance_service.MedicineRelevanceLLMJudge()
    session_factory = build_session_factory_with_ai_models([])

    class FakeModel:
        async def ainvoke(self, messages: list[object]) -> AIMessage:
            assert len(messages) == 2
            return AIMessage(
                content=json.dumps(
                    {
                        "relevance": "relevant",
                        "confidence": 0.91,
                        "reason": "Matches diagnosis context.",
                        "matched_diagnosis": True,
                        "matched_anamnesis": False,
                    }
                )
            )

    monkeypatch.setattr("app.services.ai_service.get_session_factory", lambda: session_factory)
    monkeypatch.setattr(judge, "_create_chat_model", lambda model_spec: FakeModel())
    monkeypatch.setattr(
        judge,
        "_get_enabled_model_specs",
        lambda: [AIModelSpec(provider="google_genai", model="judge-model")],
    )

    result = asyncio.run(
        judge.judge(
            medicine_name="Paracetamol",
            diagnosis_text="Influenza",
            anamnesis_text="Demam dua hari",
            evidence_rows=[],
            ai_call_context={"user_id": 7},
        )
    )

    assert result.relevance == "relevant"

    with session_factory() as db:
        row = db.execute(select(AIApiCallLog)).scalar_one()
        assert row.call_kind == "medicine_relevance"
        assert row.parsed_response_payload["relevance"] == "relevant"
        assert row.result_context_payload == {
            "medicine_name": "Paracetamol",
            "relevance": "relevant",
            "confidence": 0.91,
        }


def test_medicine_prioritization_judge_persists_parsed_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    judge = ScanMedicineImageInterpretationJudge()
    session_factory = build_session_factory_with_ai_models([])

    class FakeModel:
        async def ainvoke(self, messages: list[object]) -> AIMessage:
            assert len(messages) == 2
            return AIMessage(
                content=json.dumps(
                    {
                        "judgements": [
                            {
                                "original_index": 0,
                                "interpreted_name": "Paracetamol",
                                "interpretation_confidence": 0.94,
                                "interpretation_reason": "Label is clear.",
                                "relevance": "relevant",
                                "relevance_confidence": 0.73,
                                "matched_diagnosis": True,
                                "matched_anamnesis": False,
                            }
                        ]
                    }
                )
            )

    monkeypatch.setattr("app.services.ai_service.get_session_factory", lambda: session_factory)
    monkeypatch.setattr(judge, "_create_chat_model", lambda model_spec: FakeModel())
    monkeypatch.setattr(
        judge,
        "_get_enabled_model_specs",
        lambda: [AIModelSpec(provider="google_genai", model="judge-model")],
    )

    result = asyncio.run(
        judge.judge_batch(
            images=[DocumentImageInput(filename="scan.png", mime_type="image/png", base64_data="YWJj")],
            diagnosis_text="Influenza",
            anamnesis_text="Demam",
            ambiguous_medicines=[{"name": "Paracetmol"}],
            ranked_group_suggestions=[],
            db_medicine_hints=[],
            ai_call_context={"scan_log_id": 44, "user_id": 3},
        )
    )

    assert isinstance(result, BatchMedicineInterpretationResponse)
    assert len(result.judgements) == 1

    with session_factory() as db:
        row = db.execute(select(AIApiCallLog)).scalar_one()
        assert row.call_kind == "medicine_prioritization"
        assert row.parsed_response_payload["judgements"][0]["interpreted_name"] == "Paracetamol"
        assert row.result_context_payload == {
            "ambiguous_medicine_count": 1,
            "judgement_count": 1,
        }


@pytest.mark.parametrize(
    ("response_payload", "expected_interpreted_name"),
    [
        (
            {
                "instructions": [{"text": "Return exactly one key: judgements."}],
                "judgements": [
                    {
                        "original_index": 0,
                        "interpreted_name": "Paracetamol",
                        "interpretation_confidence": 0.94,
                        "interpretation_reason": "Label is clear.",
                        "relevance": "relevant",
                        "relevance_confidence": 0.73,
                        "matched_diagnosis": True,
                        "matched_anamnesis": False,
                    }
                ],
            },
            "Paracetamol",
        ),
        (
            {
                "judgments": [
                    {
                        "original_index": 0,
                        "interpreted_name": "Paracetamol",
                        "interpretation_confidence": 0.94,
                        "interpretation_reason": "Label is clear.",
                        "relevance": "relevant",
                        "relevance_confidence": 0.73,
                        "matched_diagnosis": True,
                        "matched_anamnesis": False,
                    }
                ],
            },
            "Paracetamol",
        ),
        (
            {
                "judgements": [
                    {
                        "original_index": 0,
                        "name": "CARDIO ASPIRIN 100 MG",
                        "interpretation_confidence": 0.94,
                        "interpretation_reason": "Label is clear.",
                        "relevance": "relevant",
                        "relevance_confidence": 0.73,
                        "matched_diagnosis": True,
                        "matched_anamnesis": False,
                    }
                ],
            },
            "CARDIO ASPIRIN 100 MG",
        ),
    ],
)
def test_medicine_prioritization_judge_normalizes_common_schema_slips(
    monkeypatch: pytest.MonkeyPatch,
    response_payload: dict[str, Any],
    expected_interpreted_name: str,
) -> None:
    judge = ScanMedicineImageInterpretationJudge()
    session_factory = build_session_factory_with_ai_models([])

    class FakeModel:
        async def ainvoke(self, messages: list[object]) -> AIMessage:
            assert len(messages) == 2
            return AIMessage(content=json.dumps(response_payload))

    monkeypatch.setattr("app.services.ai_service.get_session_factory", lambda: session_factory)
    monkeypatch.setattr(judge, "_create_chat_model", lambda model_spec: FakeModel())
    monkeypatch.setattr(
        judge,
        "_get_enabled_model_specs",
        lambda: [AIModelSpec(provider="google_genai", model="judge-model")],
    )

    result = asyncio.run(
        judge.judge_batch(
            images=[DocumentImageInput(filename="scan.png", mime_type="image/png", base64_data="YWJj")],
            diagnosis_text="Influenza",
            anamnesis_text="Demam",
            ambiguous_medicines=[{"extracted_name": "Paracetmol"}],
            ranked_group_suggestions=[],
            db_medicine_hints=[],
            ai_call_context={"scan_log_id": 44, "user_id": 3},
        )
    )

    assert result.judgements[0].interpreted_name == expected_interpreted_name

    with session_factory() as db:
        row = db.execute(select(AIApiCallLog)).scalar_one()
        assert row.parsed_response_payload["judgements"][0]["interpreted_name"] == expected_interpreted_name
        assert "instructions" not in row.parsed_response_payload
        assert "judgments" not in row.parsed_response_payload
        assert "name" not in row.parsed_response_payload["judgements"][0]
        assert row.response_validation_error is None


def test_medicine_prioritization_judge_keeps_strict_validation_after_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    judge = ScanMedicineImageInterpretationJudge()
    session_factory = build_session_factory_with_ai_models([])

    class FakeModel:
        async def ainvoke(self, messages: list[object]) -> AIMessage:
            assert len(messages) == 2
            return AIMessage(
                content=json.dumps(
                    {
                        "instructions": [{"text": "Return exactly one key: judgements."}],
                        "expected_types": {"judgements": "array<object>"},
                    }
                )
            )

    monkeypatch.setattr("app.services.ai_service.get_session_factory", lambda: session_factory)
    monkeypatch.setattr(judge, "_create_chat_model", lambda model_spec: FakeModel())
    monkeypatch.setattr(
        judge,
        "_get_enabled_model_specs",
        lambda: [AIModelSpec(provider="google_genai", model="judge-model")],
    )

    with pytest.raises(RuntimeError, match="Batch medicine interpretation payload is invalid."):
        asyncio.run(
            judge.judge_batch(
                images=[DocumentImageInput(filename="scan.png", mime_type="image/png", base64_data="YWJj")],
                diagnosis_text="Influenza",
                anamnesis_text="Demam",
                ambiguous_medicines=[{"extracted_name": "Paracetmol"}],
                ranked_group_suggestions=[],
                db_medicine_hints=[],
                ai_call_context={"scan_log_id": 44, "user_id": 3},
            )
        )

    with session_factory() as db:
        row = db.execute(select(AIApiCallLog)).scalar_one()
        assert row.response_validation_error == "schema validation failed paths=instructions,expected_types"


def test_medicine_prioritization_judge_logs_validation_failures_without_input_values(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    judge = ScanMedicineImageInterpretationJudge()
    session_factory = build_session_factory_with_ai_models([])

    class FakeModel:
        async def ainvoke(self, messages: list[object]) -> AIMessage:
            assert len(messages) == 2
            return AIMessage(
                content=json.dumps(
                    {
                        "judgements": [
                            {
                                "original_index": 0,
                                "interpreted_name": "Paracetamol",
                                "interpretation_confidence": 0.94,
                                "interpretation_reason": "Label is clear.",
                                "relevance": "relevant",
                                "relevance_confidence": 0.73,
                                "matched_diagnosis": True,
                                "matched_anamnesis": False,
                                "unexpected": "CARDIO ASPIRIN 100 MG",
                            }
                        ]
                    }
                )
            )

    monkeypatch.setenv("AI_PIPELINE_DEBUG_LOGS", "true")
    clear_ai_pipeline_debug_logging_cache()
    monkeypatch.setattr("app.services.ai_service.get_session_factory", lambda: session_factory)
    monkeypatch.setattr(judge, "_create_chat_model", lambda model_spec: FakeModel())
    monkeypatch.setattr(
        judge,
        "_get_enabled_model_specs",
        lambda: [AIModelSpec(provider="google_genai", model="judge-model")],
    )

    with pytest.raises(RuntimeError, match="Batch medicine interpretation payload is invalid."):
        asyncio.run(
            judge.judge_batch(
                images=[DocumentImageInput(filename="scan.png", mime_type="image/png", base64_data="YWJj")],
                diagnosis_text="Influenza",
                anamnesis_text="Demam",
                ambiguous_medicines=[{"extracted_name": "Paracetmol"}],
                ranked_group_suggestions=[],
                db_medicine_hints=[],
                ai_call_context={"scan_log_id": 44, "user_id": 3},
            )
        )

    captured = capsys.readouterr()
    assert "CARDIO ASPIRIN 100 MG" not in captured.err
    assert "input_value" not in captured.err
    assert "schema validation failed paths=judgements.0.unexpected" in captured.err

    with session_factory() as db:
        row = db.execute(select(AIApiCallLog)).scalar_one()
        assert row.response_validation_error == "schema validation failed paths=judgements.0.unexpected"
        assert "CARDIO ASPIRIN 100 MG" not in row.response_validation_error

    monkeypatch.delenv("AI_PIPELINE_DEBUG_LOGS", raising=False)
    clear_ai_pipeline_debug_logging_cache()


def test_model_invocation_test_persists_call_log_without_scan_log_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = model_invocation_test_module.ModelInvocationTestService(settings=build_settings())
    session_factory = build_session_factory_with_ai_models([])

    class FakeModel:
        async def ainvoke(self, messages: list[object]) -> AIMessage:
            assert len(messages) == 1
            return AIMessage(content=json.dumps({"ok": True}))

    monkeypatch.setattr("app.services.ai_service.get_session_factory", lambda: session_factory)
    monkeypatch.setattr(service, "_create_chat_model", lambda model_spec: FakeModel())

    result = asyncio.run(
        service.invoke_model(
            provider="google_genai",
            model_name="gemini-test",
            message="hello",
            image=None,
        )
    )

    assert result.parsed_json == {"ok": True}

    with session_factory() as db:
        row = db.execute(select(AIApiCallLog)).scalar_one()
        assert row.scan_log_id is None
        assert row.call_kind == "model_test"
        assert row.raw_response_text == json.dumps({"ok": True})
        assert row.parsed_response_payload == {"ok": True}
        assert row.result_context_payload == {
            "call_kind": "model_test",
            "image_supplied": False,
            "json_parsed": True,
        }


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
    assert result.extraction_meta_payload["judge_attempt_count"] == 1
    assert result.extraction_meta_payload["judge_applied_count"] == 1
    assert result.extraction_meta_payload["judge_cached_hit_count"] == 0


def test_process_defers_financial_judge_when_reconciliation_can_resolve_conflict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    medical_processor = MedicalDocumentProcessor(settings=build_settings())

    outputs = {
        "model-a": {
            **SCAN_RESULT,
            "summary": {
                **SCAN_RESULT["summary"],
                "total_amount": 999999,
            },
            "financial_recap": {
                "total_medicines": 111111,
                "total_other_charges": 222222,
                "grand_total": 999999,
            },
        },
        "model-b": {
            **SCAN_RESULT,
        },
    }

    class FakeModel:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        async def ainvoke(self, messages: list[object]) -> AIMessage:
            del messages
            return AIMessage(content=json.dumps(outputs[self.model_name]))

    async def fail_for_financial_judge(
        field_path: str,
        images: list[object],
        candidates: list[dict[str, object]],
    ) -> tuple[str | None, str | None]:
        del images, candidates
        if field_path in {
            "summary.total_amount",
            "financial_recap.total_medicines",
            "financial_recap.total_other_charges",
            "financial_recap.grand_total",
        }:
            raise AssertionError("Financial fields should not be judged during the first pass when reconciliation can resolve them.")
        return None, None

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
    monkeypatch.setattr(medical_processor, "_judge_field_conflict", fail_for_financial_judge)

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

    assert result.result.summary.total_amount == SCAN_RESULT["summary"]["total_amount"]
    assert result.result.financial_recap.total_medicines == SCAN_RESULT["financial_recap"]["total_medicines"]
    assert result.result.financial_recap.total_other_charges == SCAN_RESULT["financial_recap"]["total_other_charges"]
    assert result.result.financial_recap.grand_total == SCAN_RESULT["financial_recap"]["grand_total"]


def test_fusion_rejudge_cache_reuses_no_selection_result() -> None:
    judge_call_count = 0

    async def fake_judge(
        field_path: str,
        images: list[object],
        candidates: list[dict[str, object]],
    ) -> tuple[str | None, str | None]:
        nonlocal judge_call_count
        del field_path, images, candidates
        judge_call_count += 1
        return None, "Still ambiguous"

    engine = MedicalExtractionFusionEngine(judge_field=fake_judge)
    state = FusionWorkingState(
        field_confidence={
            "summary.total_amount": {
                "value": 125000,
                "confidence": 0.4,
                "candidates": [
                    {
                        "normalized_key": "125000",
                        "value": 125000,
                        "support_count": 2,
                        "producer_count": 2,
                    },
                    {
                        "normalized_key": "130000",
                        "value": 130000,
                        "support_count": 2,
                        "producer_count": 2,
                    },
                ],
                "judge_used": False,
                "judge_reason": None,
                "review_flags": [],
                "rule_hits": [],
            }
        }
    )
    result = AIMedicalScanResponse.model_validate(SCAN_RESULT)

    first = asyncio.run(
        engine._maybe_rejudge_field_from_candidates(
            result,
            field_path="summary.total_amount",
            images=[
                DocumentImageInput(
                    filename="scan.png",
                    mime_type="image/png",
                    base64_data="YWJj",
                )
            ],
            field_confidence=state.field_confidence,
            state=state,
        )
    )
    second = asyncio.run(
        engine._maybe_rejudge_field_from_candidates(
            result,
            field_path="summary.total_amount",
            images=[
                DocumentImageInput(
                    filename="scan.png",
                    mime_type="image/png",
                    base64_data="YWJj",
                )
            ],
            field_confidence=state.field_confidence,
            state=state,
        )
    )

    assert first is False
    assert second is False
    assert judge_call_count == 1
    assert state.judge_attempt_count == 1
    assert state.judge_cached_hit_count == 1


def test_build_medicine_ledger_candidates_prefers_explicit_medicine_price_source_for_same_amount() -> None:
    async def noop_judge(
        field_path: str,
        images: list[object],
        candidates: list[dict[str, object]],
    ) -> tuple[str | None, str | None]:
        del field_path, images, candidates
        return None, None

    engine = MedicalExtractionFusionEngine(judge_field=noop_judge)
    result = AIMedicalScanResponse.model_validate(
        {
            **SCAN_RESULT,
            "medicines": [
                {
                    **SCAN_RESULT["medicines"][0],
                    "total_price": 10000,
                }
            ],
            "other_charges": [
                {
                    "category": "OTHER",
                    "description": "Obat",
                    "amount": 10000,
                }
            ],
        }
    )
    list_group_meta = {
        "medicines": {
            "selected_groups": [
                {
                    "output_index": 0,
                    "path_prefix": "medicines[paracetamol]",
                    "key": "paracetamol",
                    "score": 1.0,
                    "support_count": 1,
                    "producer_count": 1,
                    "is_grouped_compound": False,
                    "is_package_level_medicine": False,
                }
            ]
        },
        "other_charges": {
            "groups": [],
            "selected_groups": [
                {
                    "bucket": "medicine",
                    "amount": 10000,
                    "key": "MEDICINE:10000:other",
                    "path_prefix": "other_charges[medicine-10000-other]",
                    "score": 1.0,
                    "support_count": 2,
                    "producer_count": 1,
                    "representative_value": {
                        "category": "OTHER",
                        "description": "Obat",
                        "amount": 10000,
                    },
                }
            ],
        },
    }

    candidates = engine._build_medicine_ledger_candidates(
        result,
        selected_other_charge_groups=cast(list[dict[str, Any]], list_group_meta["other_charges"]["selected_groups"]),
        list_group_meta=cast(dict[str, Any], list_group_meta),
    )

    assert len(candidates) == 1
    assert candidates[0]["amount"] == 10000
    assert candidates[0]["source_type"] == "medicine_row"


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


def test_financial_dominance_helper_requires_minimum_strength_gap() -> None:
    async def noop_judge(
        field_path: str,
        images: list[Any],
        candidates: list[dict[str, Any]],
    ) -> tuple[str | None, str | None]:
        del field_path, images, candidates
        return None, None

    engine = MedicalExtractionFusionEngine(judge_field=noop_judge)
    below_threshold_candidates = [
        {
            "evidence_summary": {
                "evidence_strength": 8.0,
                "grand_total_match_score": 2.0,
                "other_total_match_score": 1.5,
            }
        },
        {
            "evidence_summary": {
                "evidence_strength": 6.76,
                "grand_total_match_score": 2.0,
                "other_total_match_score": 1.5,
            }
        },
    ]
    assert engine._has_dominant_financial_candidate(below_threshold_candidates) is False

    at_threshold_candidates = [
        {
            "evidence_summary": {
                "evidence_strength": 8.0,
                "grand_total_match_score": 2.0,
                "other_total_match_score": 1.5,
            }
        },
        {
            "evidence_summary": {
                "evidence_strength": 6.75,
                "grand_total_match_score": 2.0,
                "other_total_match_score": 1.5,
            }
        },
    ]
    assert engine._has_dominant_financial_candidate(at_threshold_candidates) is True


def test_financial_dominance_helper_requires_non_regressing_grand_total_match() -> None:
    async def noop_judge(
        field_path: str,
        images: list[Any],
        candidates: list[dict[str, Any]],
    ) -> tuple[str | None, str | None]:
        del field_path, images, candidates
        return None, None

    engine = MedicalExtractionFusionEngine(judge_field=noop_judge)
    weaker_grand_total_match_candidates = [
        {
            "evidence_summary": {
                "evidence_strength": 8.5,
                "grand_total_match_score": 1.9,
                "other_total_match_score": 1.8,
            }
        },
        {
            "evidence_summary": {
                "evidence_strength": 7.0,
                "grand_total_match_score": 2.1,
                "other_total_match_score": 1.8,
            }
        },
    ]
    assert engine._has_dominant_financial_candidate(weaker_grand_total_match_candidates) is False

    stronger_grand_total_match_candidates = [
        {
            "evidence_summary": {
                "evidence_strength": 8.5,
                "grand_total_match_score": 2.1,
                "other_total_match_score": 1.8,
            }
        },
        {
            "evidence_summary": {
                "evidence_strength": 7.0,
                "grand_total_match_score": 2.1,
                "other_total_match_score": 1.8,
            }
        },
    ]
    assert engine._has_dominant_financial_candidate(stronger_grand_total_match_candidates) is True


def test_financial_dominance_helper_requires_non_regressing_other_total_match() -> None:
    async def noop_judge(
        field_path: str,
        images: list[Any],
        candidates: list[dict[str, Any]],
    ) -> tuple[str | None, str | None]:
        del field_path, images, candidates
        return None, None

    engine = MedicalExtractionFusionEngine(judge_field=noop_judge)
    weaker_other_total_match_candidates = [
        {
            "evidence_summary": {
                "evidence_strength": 8.5,
                "grand_total_match_score": 2.1,
                "other_total_match_score": 1.3,
            }
        },
        {
            "evidence_summary": {
                "evidence_strength": 7.0,
                "grand_total_match_score": 2.1,
                "other_total_match_score": 1.4,
            }
        },
    ]
    assert engine._has_dominant_financial_candidate(weaker_other_total_match_candidates) is False

    stronger_other_total_match_candidates = [
        {
            "evidence_summary": {
                "evidence_strength": 8.5,
                "grand_total_match_score": 2.1,
                "other_total_match_score": 1.4,
            }
        },
        {
            "evidence_summary": {
                "evidence_strength": 7.0,
                "grand_total_match_score": 2.1,
                "other_total_match_score": 1.4,
            }
        },
    ]
    assert engine._has_dominant_financial_candidate(stronger_other_total_match_candidates) is True


def test_financial_dominance_helper_requires_adjudication_for_material_total_difference() -> None:
    async def noop_judge(
        field_path: str,
        images: list[Any],
        candidates: list[dict[str, Any]],
    ) -> tuple[str | None, str | None]:
        del field_path, images, candidates
        return None, None

    engine = MedicalExtractionFusionEngine(judge_field=noop_judge)
    materially_different_candidates = [
        {
            "grand_total": 1145000,
            "medicine_total": 145000,
            "other_total": 1000000,
            "evidence_summary": {
                "evidence_strength": 9.5,
                "grand_total_match_score": 2.5,
                "other_total_match_score": 2.5,
            },
        },
        {
            "grand_total": 597900,
            "medicine_total": 102900,
            "other_total": 495000,
            "evidence_summary": {
                "evidence_strength": 8.0,
                "grand_total_match_score": 2.2,
                "other_total_match_score": 2.0,
                "receipt_backed_packet": True,
            },
        },
    ]

    assert engine._has_dominant_financial_candidate(materially_different_candidates) is False


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

    async def conservative_financial_packet_judge(
        images: list[object],
        candidates: list[dict[str, object]],
    ) -> tuple[int | None, str | None]:
        del images
        selected_index = next(index for index, candidate in enumerate(candidates) if candidate["grand_total"] == 356900)
        return selected_index, "Receipt-backed 356900 ledger is strongest."

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
    monkeypatch.setattr(medical_processor, "_judge_financial_packet_conflict", conservative_financial_packet_judge)

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


def test_other_charge_classifier_treats_pharmacy_payment_terms_as_medicine() -> None:
    assert classify_other_charge_kind("SERVICE", "Pembayaran Resep Apotek Makmuri") == "MEDICINE"
    assert classify_other_charge_kind("OTHER", "Nota Penjualan Resep a/n pasien") == "MEDICINE"
    assert classify_other_charge_kind("SERVICE", "Instalasi Farmasi Klinik") == "MEDICINE"
    assert classify_other_charge_kind("LABORATORY", "Pemeriksaan Laboratorium Kolesterol") == "LABORATORY"
    assert classify_other_charge_kind("CONSULTATION", "Biaya konsultasi dokter") == "CONSULTATION"


def test_process_deduplicates_padma_pharmacy_receipt_from_other_charges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    medical_processor = MedicalDocumentProcessor(settings=build_settings())
    outputs = {
        "model-primary": [
            {
                **SCAN_RESULT,
                "summary": {
                    "detected_documents": ["Formulir Rawat Jalan", "Kwitansi", "Resep", "Kwitansi Apotek"],
                    "medical_facility_name": "Apotek Padma Medika",
                    "service_date": "2026-04-26",
                    "transaction_date": None,
                    "currency": "IDR",
                    "total_amount": 344000,
                },
                "patient_info": {"name": "Bryan Immanuel Wibawa", "date_of_birth": None, "age": 10, "gender": None, "address": None},
                "medical_context": {
                    "anamnesis": "Nyeri telan batuk pilek demam sariawan",
                    "diagnoses": [{"description": "ISPA + stomatitis", "doctor_name": "Dr. Gani Wangunhardjo, Sp.A"}],
                    "therapy_note": "5 macam",
                },
                "medicines": [{"name": "Cataflam", "quantity": 8, "unit": "Cap", "dosage_instruction": None, "unit_price": None, "total_price": None}],
                "other_charges": [
                    {"category": "CONSULTATION", "description": "Biaya Pemeriksaan & Pengobatan", "amount": 200000},
                    {"category": "OTHER", "description": "Resep", "amount": 144000},
                ],
                "financial_recap": {"total_medicines": 144000, "total_other_charges": 200000, "grand_total": 344000},
            },
            {
                **SCAN_RESULT,
                "summary": {
                    "detected_documents": ["Formulir Rawat Jalan", "Kwitansi", "Nota Penjualan", "Resep"],
                    "medical_facility_name": "Apotek Padma Medika",
                    "service_date": "2026-04-26",
                    "transaction_date": "2026-04-26",
                    "currency": "IDR",
                    "total_amount": 344000,
                },
                "patient_info": {"name": "Bryan Immanuel Wibawa", "date_of_birth": None, "age": 10, "gender": None, "address": None},
                "medical_context": {
                    "anamnesis": "Nyeri telan batuk pilek demam sariawan",
                    "diagnoses": [{"description": "ISPA + stomatitis", "doctor_name": "Dr. Gani Wangunhardjo, Sp.A"}],
                    "therapy_note": "4 macam",
                },
                "medicines": [{"name": "Cataflam", "quantity": 8, "unit": "Cap", "dosage_instruction": None, "unit_price": None, "total_price": None}],
                "other_charges": [
                    {"category": "CONSULTATION", "description": "Biaya Pemeriksaan & Pengobatan", "amount": 200000},
                    {"category": "SERVICE", "description": "Resep a/n Bryan Immanuel W", "amount": 144000},
                ],
                "financial_recap": {"total_medicines": 144000, "total_other_charges": 344000, "grand_total": 344000},
            },
        ],
        "model-secondary": [
            {
                **SCAN_RESULT,
                "summary": {
                    "detected_documents": ["Formulir Rawat Jalan", "Kwitansi", "Resep"],
                    "medical_facility_name": "Dr. Gani Wangunhardjo",
                    "service_date": "2026-04-26",
                    "transaction_date": "2026-04-26",
                    "currency": "IDR",
                    "total_amount": 344000,
                },
                "patient_info": {"name": "Bryan Immanuel Wibawa", "date_of_birth": None, "age": 10, "gender": "MALE", "address": None},
                "medical_context": {
                    "anamnesis": "Nyeri telan batuk pilek demam",
                    "diagnoses": [{"description": "ISPA + stomatitis", "doctor_name": "Dr. Gani Wangunhardjo, Sp.A"}],
                    "therapy_note": "5 macam",
                },
                "medicines": [{"name": "Tremenza", "quantity": 15, "unit": "Cap", "dosage_instruction": None, "unit_price": None, "total_price": None}],
                "other_charges": [
                    {"category": "SERVICE", "description": "Biaya Pemeriksaan & Pengobatan", "amount": 200000},
                    {"category": "Pharmacy", "description": "Apotek Padma Medika", "amount": 144000},
                ],
                "financial_recap": {"total_medicines": 144000, "total_other_charges": 200000, "grand_total": 344000},
            },
            {
                **SCAN_RESULT,
                "summary": {
                    "detected_documents": ["Formulir Rawat Jalan", "Kwitansi", "Resep"],
                    "medical_facility_name": "Dr. Gani Wangunhardjo",
                    "service_date": "2026-04-26",
                    "transaction_date": "2026-04-26",
                    "currency": "IDR",
                    "total_amount": 344000,
                },
                "patient_info": {"name": "Bryan Immanuel Wibawa", "date_of_birth": None, "age": 10, "gender": "MALE", "address": None},
                "medical_context": {
                    "anamnesis": "Nyeri telan batuk pilek demam",
                    "diagnoses": [{"description": "ISPA + stomatitis", "doctor_name": "Dr. Gani Wangunhardjo, Sp.A"}],
                    "therapy_note": "5 macam",
                },
                "medicines": [{"name": "Tremenza", "quantity": 15, "unit": "Cap", "dosage_instruction": None, "unit_price": None, "total_price": None}],
                "other_charges": [
                    {"category": "SERVICE", "description": "Biaya Pemeriksaan & Pengobatan", "amount": 200000},
                    {"category": "OTHER", "description": "Pembelian obat Apotek Padma Medika", "amount": 144000},
                ],
                "financial_recap": {"total_medicines": 144000, "total_other_charges": 200000, "grand_total": 344000},
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

    monkeypatch.setattr(medical_processor, "_create_chat_model", lambda model_spec: FakeModel(model_spec.model))
    monkeypatch.setattr(
        medical_processor,
        "_get_enabled_model_specs",
        lambda: [
            AIModelSpec(provider="google_genai", model="model-primary"),
            AIModelSpec(provider="ollama", model="model-secondary"),
        ],
    )
    monkeypatch.setattr(
        medical_processor,
        "_judge_field_conflict",
        lambda field_path, images, candidates: asyncio.sleep(0, result=(cast(str, candidates[0]["normalized_key"]), "Preserve leading candidate.")),
    )
    monkeypatch.setattr(
        medical_processor,
        "_judge_financial_packet_conflict",
        lambda images, candidates: asyncio.sleep(0, result=(0, "Use the coherent pharmacy plus consultation ledger.")),
    )

    result = asyncio.run(
        medical_processor.process(
            [DocumentImageInput(filename="scan.png", mime_type="image/png", base64_data="YWJj")]
        )
    )

    assert result.result.financial_recap.total_medicines == 144000
    assert result.result.financial_recap.total_other_charges == 200000
    assert result.result.financial_recap.grand_total == 344000
    assert result.result.summary.total_amount == 344000
    assert len(result.result.other_charges) == 1
    assert result.result.other_charges[0].amount == 200000


def test_process_reconciles_multi_receipt_packet_and_merges_duplicate_payment_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    medical_processor = MedicalDocumentProcessor(settings=build_settings())
    outputs = {
        "model-primary": [
            {
                **SCAN_RESULT,
                "summary": {
                    "detected_documents": ["Kwitansi Pembayaran Laboratorium", "Kwitansi Dokter", "Salinan Resep", "Nota Penjualan"],
                    "medical_facility_name": "Parahita Diagnostic Center",
                    "service_date": "2023-07-07",
                    "transaction_date": "2023-07-07",
                    "currency": "IDR",
                    "total_amount": 657900,
                },
                "patient_info": {"name": "Jane Doe", "date_of_birth": None, "age": 25, "gender": None, "address": None},
                "medical_context": {
                    "anamnesis": "kontrol",
                    "diagnoses": [{"description": "PJK + HT", "doctor_name": "dr. Samuel Sudanawidjaja"}],
                    "therapy_note": "Platogrix, Thrombo, Rytmonorm",
                },
                "medicines": [{"name": "Platogrix 75 mg", "quantity": 30, "unit": "tablet", "dosage_instruction": None, "unit_price": None, "total_price": None}],
                "other_charges": [
                    {"category": "LABORATORY", "description": "Pemeriksaan Laboratorium: Kolesterol Total, Kolesterol HDL, Glukosa Puasa / BSN, Free T4", "amount": 657900},
                    {"category": "CONSULTATION", "description": "Pemeriksaan & konsultasi", "amount": 350000},
                    {"category": "OTHER", "description": "Nota Penjualan Resep Apotek UBAYA", "amount": 1472600},
                ],
                "financial_recap": {"total_medicines": 1472600, "total_other_charges": 1007900, "grand_total": 657900},
            },
            {
                **SCAN_RESULT,
                "summary": {
                    "detected_documents": ["Kwitansi Pembayaran Laboratorium", "Kwitansi", "Salinan Resep", "Nota Penjualan"],
                    "medical_facility_name": "Parahita Diagnostic Center",
                    "service_date": "2023-07-07",
                    "transaction_date": "2023-07-07",
                    "currency": "IDR",
                    "total_amount": 2130900,
                },
                "patient_info": {"name": "Jane Doe", "date_of_birth": None, "age": 25, "gender": None, "address": None},
                "medical_context": {
                    "anamnesis": "kontrol",
                    "diagnoses": [{"description": "PJK + HT", "doctor_name": "dr. Samuel Sudanawidjaja"}],
                    "therapy_note": "Platogrix, Thrombo, Rytmonorm",
                },
                "medicines": [{"name": "Thrombo aspilet", "quantity": 15, "unit": "tablet", "dosage_instruction": None, "unit_price": None, "total_price": None}],
                "other_charges": [
                    {"category": "LABORATORY", "description": "Kolesterol Total, Kolesterol HDL, Glukosa Puasa / BSN, Free T4", "amount": 657900},
                    {"category": "CONSULTATION", "description": "Pemeriksaan dan konsultasi", "amount": 350000},
                    {"category": "SERVICE", "description": "Pembayaran Resep Apotek UBAYA", "amount": 1472600},
                ],
                "financial_recap": {"total_medicines": 1472600, "total_other_charges": 1007900, "grand_total": 2130900},
            },
        ],
        "model-secondary": [
            {
                **SCAN_RESULT,
                "summary": {
                    "detected_documents": ["Kwitansi", "Resep", "Formulir Rawat Jalan", "Nota Penjualan"],
                    "medical_facility_name": "Parahita Diagnostic Center",
                    "service_date": "2023-07-07",
                    "transaction_date": None,
                    "currency": "IDR",
                    "total_amount": 657900,
                },
                "patient_info": {"name": "Jane Doe", "date_of_birth": None, "age": 25, "gender": None, "address": None},
                "medical_context": {
                    "anamnesis": "kontrol",
                    "diagnoses": [{"description": "PJK + HT", "doctor_name": "dr. Samuel Sudanawidjaja"}],
                    "therapy_note": "obat rutin",
                },
                "medicines": [{"name": "Rytmonorm", "quantity": 30, "unit": "tablet", "dosage_instruction": None, "unit_price": None, "total_price": None}],
                "other_charges": [
                    {"category": "LABORATORY", "description": "Tagihan Pemeriksaan Laboratorium Kolesterol Total HDL Glukosa Puasa Free T4", "amount": 657900},
                    {"category": "LABORATORY", "description": "Kwitansi Pemeriksaan Laboratorium Kolesterol Total HDL Glukosa Puasa Free T4", "amount": 657900},
                    {"category": "CONSULTATION", "description": "Pemeriksaan & konsultasi", "amount": 350000},
                    {"category": "SERVICE", "description": "Resep a/n Jane Doe dari dr. Samuel", "amount": 1472600},
                ],
                "financial_recap": {"total_medicines": 1472600, "total_other_charges": 1665800, "grand_total": 657900},
            },
            {
                **SCAN_RESULT,
                "summary": {
                    "detected_documents": ["Kwitansi", "Resep", "Formulir Rawat Jalan", "Nota Penjualan"],
                    "medical_facility_name": "Parahita Diagnostic Center",
                    "service_date": "2023-07-07",
                    "transaction_date": None,
                    "currency": "IDR",
                    "total_amount": 2130600,
                },
                "patient_info": {"name": "Jane Doe", "date_of_birth": None, "age": 25, "gender": None, "address": None},
                "medical_context": {
                    "anamnesis": "kontrol",
                    "diagnoses": [{"description": "PJK + HT", "doctor_name": "dr. Samuel Sudanawidjaja"}],
                    "therapy_note": "obat rutin",
                },
                "medicines": [{"name": "Concor", "quantity": 30, "unit": "tablet", "dosage_instruction": None, "unit_price": None, "total_price": None}],
                "other_charges": [
                    {"category": "LABORATORY", "description": "Pemeriksaan Laboratorium Kolesterol Total HDL Glukosa Puasa Free T4", "amount": 657900},
                    {"category": "CONSULTATION", "description": "Pemeriksaan dan konsultasi dokter", "amount": 350000},
                    {"category": "OTHER", "description": "Instalasi Farmasi UBAYA Nota Penjualan", "amount": 1472600},
                ],
                "financial_recap": {"total_medicines": 1472600, "total_other_charges": 1007900, "grand_total": 2130600},
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

    monkeypatch.setattr(medical_processor, "_create_chat_model", lambda model_spec: FakeModel(model_spec.model))
    monkeypatch.setattr(
        medical_processor,
        "_get_enabled_model_specs",
        lambda: [
            AIModelSpec(provider="google_genai", model="model-primary"),
            AIModelSpec(provider="ollama", model="model-secondary"),
        ],
    )
    monkeypatch.setattr(
        medical_processor,
        "_judge_field_conflict",
        lambda field_path, images, candidates: asyncio.sleep(0, result=(cast(str, candidates[0]["normalized_key"]), "Preserve leading candidate.")),
    )
    monkeypatch.setattr(
        medical_processor,
        "_judge_financial_packet_conflict",
        lambda images, candidates: asyncio.sleep(
            0,
            result=(
                next(index for index, candidate in enumerate(candidates) if candidate["grand_total"] == 2480500),
                "Full packet ledger counts lab, consultation, and pharmacy once.",
            ),
        ),
    )

    result = asyncio.run(
        medical_processor.process(
            [DocumentImageInput(filename="scan.png", mime_type="image/png", base64_data="YWJj")]
        )
    )

    assert result.result.financial_recap.total_medicines == 1472600
    assert result.result.financial_recap.total_other_charges == 1007900
    assert result.result.financial_recap.grand_total == 2480500
    assert result.result.summary.total_amount == 2480500
    assert sorted(charge.amount for charge in result.result.other_charges if charge.amount is not None) == [350000, 657900]


def test_process_skips_financial_packet_judge_when_heuristic_receipt_backed_ledger_is_dominant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    medical_processor = MedicalDocumentProcessor(settings=build_settings())
    outputs = {
        "model-primary": [
            {
                **SCAN_RESULT,
                "summary": {
                    "detected_documents": ["Formulir Rawat Jalan", "Salinan Resep", "Kwitansi", "Kwitansi"],
                    "medical_facility_name": "Klinik Utama Makmuri",
                    "service_date": "2025-08-28",
                    "transaction_date": "2025-08-31",
                    "currency": "IDR",
                    "total_amount": 356900,
                },
                "patient_info": {
                    "name": "Nick Frederic Kristian",
                    "date_of_birth": None,
                    "age": 13,
                    "gender": "MALE",
                    "address": None,
                },
                "medical_context": {
                    "anamnesis": "demam, batuk pilek lama, nyeri",
                    "diagnoses": [{"description": "pharyngitis + asthma", "doctor_name": "dr. Makmuri MS, Sp.A(K)"}],
                    "therapy_note": "Policare",
                },
                "medicines": [{"name": "Montelukast", "quantity": 10, "unit": "tablet", "dosage_instruction": "S 1 dd 1", "unit_price": None, "total_price": 78900}],
                "other_charges": [{"category": "SERVICE", "description": "Pembayaran Resep dr. Makmuri", "amount": 278000}],
                "financial_recap": {"total_medicines": 78900, "total_other_charges": 278000, "grand_total": 356900},
            },
            {
                **SCAN_RESULT,
                "summary": {
                    "detected_documents": ["Formulir Rawat Jalan", "Salinan Resep", "Kwitansi", "Kwitansi"],
                    "medical_facility_name": "Klinik Utama Makmuri",
                    "service_date": "2025-08-28",
                    "transaction_date": "2025-08-31",
                    "currency": "IDR",
                    "total_amount": 356900,
                },
                "patient_info": {
                    "name": "Nick Frederic Kristian",
                    "date_of_birth": None,
                    "age": 13,
                    "gender": "MALE",
                    "address": None,
                },
                "medical_context": {
                    "anamnesis": "fans, batuk pilek lama, nyeri",
                    "diagnoses": [{"description": "pharyngitis + asthma", "doctor_name": "dr. Makmuri MS, Sp.A(K)"}],
                    "therapy_note": "Policare",
                },
                "medicines": [{"name": "Montelukast tab 500 mg", "quantity": 10, "unit": "tablet", "dosage_instruction": "S 1 dd 1", "unit_price": None, "total_price": 78900}],
                "other_charges": [{"category": "OTHER", "description": "Resep dr. Makmuri", "amount": 278000}],
                "financial_recap": {"total_medicines": 78900, "total_other_charges": 278000, "grand_total": 356900},
            },
        ],
        "model-secondary": [
            {
                **SCAN_RESULT,
                "summary": {
                    "detected_documents": ["Formulir Rawat Jalan", "Resep", "Kwitansi"],
                    "medical_facility_name": "Klinik Utama Makmuri",
                    "service_date": "2025-08-28",
                    "transaction_date": None,
                    "currency": "IDR",
                    "total_amount": 296900,
                },
                "patient_info": {
                    "name": "Nick Frederic Kristian",
                    "date_of_birth": None,
                    "age": 13,
                    "gender": None,
                    "address": None,
                },
                "medical_context": {
                    "anamnesis": "panas, batuk, pilek, tenggorokan",
                    "diagnoses": [{"description": "faringitis + tonsilitis", "doctor_name": "dr. Makmuri MS, Sp.A(K)"}],
                    "therapy_note": "Poliklinik, resep obat sesuai dosis",
                },
                "medicines": [{"name": "Meactrin tab 500 mg", "quantity": 1, "unit": None, "dosage_instruction": None, "unit_price": 78900, "total_price": 78900}],
                "other_charges": [{"category": "CONSULTATION", "description": "Biaya konsultasi dokter", "amount": 218000}],
                "financial_recap": {"total_medicines": 78900, "total_other_charges": 218000, "grand_total": 296900},
            },
            {
                **SCAN_RESULT,
                "summary": {
                    "detected_documents": ["Formulir Rawat Jalan", "Resep", "Kwitansi"],
                    "medical_facility_name": "Klinik Utama Makmuri",
                    "service_date": "2025-08-28",
                    "transaction_date": None,
                    "currency": "IDR",
                    "total_amount": 296900,
                },
                "patient_info": {
                    "name": "Nick Frederic Kristian",
                    "date_of_birth": None,
                    "age": 13,
                    "gender": None,
                    "address": None,
                },
                "medical_context": {
                    "anamnesis": "panas, batuk, pilek, tenggorokan",
                    "diagnoses": [{"description": "faringitis + tonsilitis", "doctor_name": "dr. Makmuri MS, Sp.A(K)"}],
                    "therapy_note": "Poliklinik, resep obat sesuai dosis",
                },
                "medicines": [{"name": "Meactrin tab 500 mg", "quantity": 1, "unit": None, "dosage_instruction": None, "unit_price": 78900, "total_price": 78900}],
                "other_charges": [{"category": "CONSULTATION", "description": "Biaya konsultasi dokter", "amount": 218000}],
                "financial_recap": {"total_medicines": 78900, "total_other_charges": 218000, "grand_total": 296900},
            },
        ],
        "model-noisy": [
            {
                **SCAN_RESULT,
                "summary": {
                    "detected_documents": ["Formulir Rawat Jalan", "Resep", "Kwitansi"],
                    "medical_facility_name": "Klinik Makmuri",
                    "service_date": "2025-08-28",
                    "transaction_date": None,
                    "currency": "IDR",
                    "total_amount": 296900,
                },
                "patient_info": {
                    "name": "Nick Frederic Kristian",
                    "date_of_birth": None,
                    "age": 13,
                    "gender": "MALE",
                    "address": None,
                },
                "medical_context": {
                    "anamnesis": "Panas, batuk pilek sejak 3 hari",
                    "diagnoses": [{"description": "Gangguan pernafasan atas", "doctor_name": "dr. Makmuri MS. Sp.A(K)"}],
                    "therapy_note": "Obat: Obatan",
                },
                "medicines": [{"name": "Paracetamol", "quantity": 1, "unit": "strip", "dosage_instruction": "1 tab x 3x sehari", "unit_price": None, "total_price": 78900}],
                "other_charges": [{"category": "OTHER", "description": "Obat-obatan", "amount": 218000}],
                "financial_recap": {"total_medicines": 78900, "total_other_charges": 218000, "grand_total": 296900},
            },
            {
                **SCAN_RESULT,
                "summary": {
                    "detected_documents": ["Formulir Rawat Jalan", "Resep", "Kwitansi"],
                    "medical_facility_name": "Klinik Makmuri",
                    "service_date": "2025-08-28",
                    "transaction_date": None,
                    "currency": "IDR",
                    "total_amount": 296900,
                },
                "patient_info": {
                    "name": "Nick Frederic Kristian",
                    "date_of_birth": None,
                    "age": 13,
                    "gender": "MALE",
                    "address": None,
                },
                "medical_context": {
                    "anamnesis": "Panas, batuk pilek sejak 3 hari",
                    "diagnoses": [{"description": "Gangguan pernafasan atas", "doctor_name": "dr. Makmuri MS. Sp.A(K)"}],
                    "therapy_note": "Obat: Obatan",
                },
                "medicines": [{"name": "Paracetamol", "quantity": 1, "unit": "strip", "dosage_instruction": "1 tab x 3x sehari", "unit_price": None, "total_price": 78900}],
                "other_charges": [{"category": "OTHER", "description": "Obat-obatan", "amount": 218000}],
                "financial_recap": {"total_medicines": 78900, "total_other_charges": 218000, "grand_total": 296900},
            },
        ],
    }
    call_counts = {model_name: 0 for model_name in outputs}
    financial_judge_calls = 0

    class FakeModel:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        async def ainvoke(self, messages: list[object]) -> AIMessage:
            del messages
            call_index = call_counts[self.model_name]
            call_counts[self.model_name] += 1
            return AIMessage(content=json.dumps(outputs[self.model_name][call_index]))

    async def field_judge(
        field_path: str,
        images: list[object],
        candidates: list[dict[str, object]],
    ) -> tuple[str | None, str | None]:
        del field_path, images, candidates
        return None, "Not needed for this test"

    async def packet_judge(
        images: list[object],
        candidates: list[dict[str, object]],
    ) -> tuple[int | None, str | None]:
        nonlocal financial_judge_calls
        del images, candidates
        financial_judge_calls += 1
        return 0, "Judge should not be called when the heuristic ledger is already dominant."

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
            AIModelSpec(provider="google_genai", model="model-secondary"),
            AIModelSpec(provider="ollama", model="model-noisy"),
        ],
    )
    monkeypatch.setattr(medical_processor, "_judge_field_conflict", field_judge)
    monkeypatch.setattr(medical_processor, "_judge_financial_packet_conflict", packet_judge)

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

    assert financial_judge_calls == 0
    assert result.result.summary.total_amount == 296900
    assert result.result.financial_recap.total_medicines == 78900
    assert result.result.financial_recap.total_other_charges == 218000
    assert result.result.financial_recap.grand_total == 296900
    assert result.extraction_meta_payload["financial_resolution_mode"] == "heuristic"
    assert result.extraction_meta_payload["financial_judge_attempt_count"] == 0
    assert result.extraction_meta_payload["financial_judge_applied_count"] == 0
    assert result.extraction_meta_payload["financial_judge_skipped_reason"] == "dominant_candidate"
    assert result.extraction_meta_payload["derived_financials"]["financial_judge_skipped_reason"] == "dominant_candidate"
    assert any(
        "Skipped packet-level financial adjudication" in note
        for note in result.extraction_meta_payload["derived_financials"]["notes"]
    )
    assert result.extraction_meta_payload["financial_evidence_summary"]["selected_candidate"]["grand_total"] == 296900


def test_process_uses_financial_packet_judge_only_for_close_ledger_tie_breaks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    medical_processor = MedicalDocumentProcessor(settings=build_settings())
    outputs = {
        "model-primary": [
            {
                **SCAN_RESULT,
                "summary": {
                    "detected_documents": ["Formulir Rawat Jalan", "Resep", "Kwitansi"],
                    "medical_facility_name": "Klinik Utama Makmuri",
                    "service_date": "2025-08-28",
                    "transaction_date": None,
                    "currency": "IDR",
                    "total_amount": 296900,
                },
                "patient_info": {
                    "name": "Nick Frederic Kristian",
                    "date_of_birth": None,
                    "age": 13,
                    "gender": None,
                    "address": None,
                },
                "medical_context": {
                    "anamnesis": "panas, batuk, pilek, tenggorokan",
                    "diagnoses": [{"description": "faringitis + tonsilitis", "doctor_name": "dr. Makmuri MS, Sp.A(K)"}],
                    "therapy_note": "Poliklinik, resep obat sesuai dosis",
                },
                "medicines": [{"name": "Meactrin tab 500 mg", "quantity": 1, "unit": None, "dosage_instruction": None, "unit_price": 78900, "total_price": 78900}],
                "other_charges": [{"category": "CONSULTATION", "description": "Biaya konsultasi dokter", "amount": 218000}],
                "financial_recap": {"total_medicines": 78900, "total_other_charges": 218000, "grand_total": 296900},
            },
            {
                **SCAN_RESULT,
                "summary": {
                    "detected_documents": ["Formulir Rawat Jalan", "Resep", "Kwitansi"],
                    "medical_facility_name": "Klinik Utama Makmuri",
                    "service_date": "2025-08-28",
                    "transaction_date": None,
                    "currency": "IDR",
                    "total_amount": 296900,
                },
                "patient_info": {
                    "name": "Nick Frederic Kristian",
                    "date_of_birth": None,
                    "age": 13,
                    "gender": None,
                    "address": None,
                },
                "medical_context": {
                    "anamnesis": "panas, batuk, pilek, tenggorokan",
                    "diagnoses": [{"description": "faringitis + tonsilitis", "doctor_name": "dr. Makmuri MS, Sp.A(K)"}],
                    "therapy_note": "Poliklinik, resep obat sesuai dosis",
                },
                "medicines": [{"name": "Meactrin tab 500 mg", "quantity": 1, "unit": None, "dosage_instruction": None, "unit_price": 78900, "total_price": 78900}],
                "other_charges": [{"category": "CONSULTATION", "description": "Biaya konsultasi dokter", "amount": 218000}],
                "financial_recap": {"total_medicines": 78900, "total_other_charges": 218000, "grand_total": 296900},
            },
        ],
        "model-secondary": [
            {
                **SCAN_RESULT,
                "summary": {
                    "detected_documents": ["Formulir Rawat Jalan", "Salinan Resep", "Kwitansi", "Kwitansi"],
                    "medical_facility_name": "Klinik Utama Makmuri",
                    "service_date": "2025-08-28",
                    "transaction_date": "2025-08-31",
                    "currency": "IDR",
                    "total_amount": 356900,
                },
                "patient_info": {
                    "name": "Nick Frederic Kristian",
                    "date_of_birth": None,
                    "age": 13,
                    "gender": "MALE",
                    "address": None,
                },
                "medical_context": {
                    "anamnesis": "demam, batuk pilek lama, nyeri",
                    "diagnoses": [{"description": "pharyngitis + asthma", "doctor_name": "dr. Makmuri MS, Sp.A(K)"}],
                    "therapy_note": "Policare",
                },
                "medicines": [{"name": "Montelukast", "quantity": 10, "unit": "tablet", "dosage_instruction": "S 1 dd 1", "unit_price": None, "total_price": 78900}],
                "other_charges": [{"category": "SERVICE", "description": "Pembayaran Resep dr. Makmuri", "amount": 278000}],
                "financial_recap": {"total_medicines": 78900, "total_other_charges": 278000, "grand_total": 356900},
            },
            {
                **SCAN_RESULT,
                "summary": {
                    "detected_documents": ["Formulir Rawat Jalan", "Salinan Resep", "Kwitansi", "Kwitansi"],
                    "medical_facility_name": "Klinik Utama Makmuri",
                    "service_date": "2025-08-28",
                    "transaction_date": "2025-08-31",
                    "currency": "IDR",
                    "total_amount": 356900,
                },
                "patient_info": {
                    "name": "Nick Frederic Kristian",
                    "date_of_birth": None,
                    "age": 13,
                    "gender": "MALE",
                    "address": None,
                },
                "medical_context": {
                    "anamnesis": "fans, batuk pilek lama, nyeri",
                    "diagnoses": [{"description": "pharyngitis + asthma", "doctor_name": "dr. Makmuri MS, Sp.A(K)"}],
                    "therapy_note": "Policare",
                },
                "medicines": [{"name": "Montelukast tab 500 mg", "quantity": 10, "unit": "tablet", "dosage_instruction": "S 1 dd 1", "unit_price": None, "total_price": 78900}],
                "other_charges": [{"category": "OTHER", "description": "Resep dr. Makmuri", "amount": 278000}],
                "financial_recap": {"total_medicines": 78900, "total_other_charges": 278000, "grand_total": 356900},
            },
        ],
    }
    call_counts = {model_name: 0 for model_name in outputs}
    financial_judge_calls = 0

    class FakeModel:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        async def ainvoke(self, messages: list[object]) -> AIMessage:
            del messages
            call_index = call_counts[self.model_name]
            call_counts[self.model_name] += 1
            return AIMessage(content=json.dumps(outputs[self.model_name][call_index]))

    async def field_judge(
        field_path: str,
        images: list[object],
        candidates: list[dict[str, object]],
    ) -> tuple[str | None, str | None]:
        del field_path, images, candidates
        return None, "Not needed for this test"

    async def packet_judge(
        images: list[object],
        candidates: list[dict[str, object]],
    ) -> tuple[int | None, str | None]:
        nonlocal financial_judge_calls
        del images
        financial_judge_calls += 1
        selected_index = next(index for index, candidate in enumerate(candidates) if candidate["grand_total"] == 356900)
        return selected_index, "Visible clinic and pharmacy receipts support 356900."

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
            AIModelSpec(provider="google_genai", model="model-secondary"),
        ],
    )
    monkeypatch.setattr(medical_processor, "_judge_field_conflict", field_judge)
    monkeypatch.setattr(medical_processor, "_judge_financial_packet_conflict", packet_judge)
    monkeypatch.setattr(
        MedicalExtractionFusionEngine,
        "_has_dominant_financial_candidate",
        lambda self, plausible_candidates: False,
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

    assert financial_judge_calls == 1
    assert result.result.summary.total_amount == 356900
    assert result.result.financial_recap.total_medicines == 78900
    assert result.result.financial_recap.total_other_charges == 278000
    assert result.result.financial_recap.grand_total == 356900
    assert result.extraction_meta_payload["financial_resolution_mode"] == "adjudicated"
    assert result.extraction_meta_payload["financial_judge_attempt_count"] == 1
    assert result.extraction_meta_payload["financial_judge_applied_count"] == 1
    assert result.extraction_meta_payload["financial_judge_skipped_reason"] is None


def test_process_marks_financial_review_when_packet_judge_abstains_on_ambiguous_ledgers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    medical_processor = MedicalDocumentProcessor(settings=build_settings())
    outputs = {
        "model-primary": [
            {
                **SCAN_RESULT,
                "summary": {
                    "detected_documents": ["Formulir Rawat Jalan", "Resep", "Kwitansi"],
                    "medical_facility_name": "Klinik Utama Makmuri",
                    "service_date": "2025-08-28",
                    "transaction_date": None,
                    "currency": "IDR",
                    "total_amount": 296900,
                },
                "patient_info": {
                    "name": "Nick Frederic Kristian",
                    "date_of_birth": None,
                    "age": 13,
                    "gender": None,
                    "address": None,
                },
                "medical_context": {
                    "anamnesis": "panas, batuk, pilek, tenggorokan",
                    "diagnoses": [{"description": "faringitis + tonsilitis", "doctor_name": "dr. Makmuri MS, Sp.A(K)"}],
                    "therapy_note": "Poliklinik, resep obat sesuai dosis",
                },
                "medicines": [{"name": "Meactrin tab 500 mg", "quantity": 1, "unit": None, "dosage_instruction": None, "unit_price": 78900, "total_price": 78900}],
                "other_charges": [{"category": "CONSULTATION", "description": "Biaya konsultasi dokter", "amount": 218000}],
                "financial_recap": {"total_medicines": 78900, "total_other_charges": 218000, "grand_total": 296900},
            },
            {
                **SCAN_RESULT,
                "summary": {
                    "detected_documents": ["Formulir Rawat Jalan", "Resep", "Kwitansi"],
                    "medical_facility_name": "Klinik Utama Makmuri",
                    "service_date": "2025-08-28",
                    "transaction_date": None,
                    "currency": "IDR",
                    "total_amount": 296900,
                },
                "patient_info": {
                    "name": "Nick Frederic Kristian",
                    "date_of_birth": None,
                    "age": 13,
                    "gender": None,
                    "address": None,
                },
                "medical_context": {
                    "anamnesis": "panas, batuk, pilek, tenggorokan",
                    "diagnoses": [{"description": "faringitis + tonsilitis", "doctor_name": "dr. Makmuri MS, Sp.A(K)"}],
                    "therapy_note": "Poliklinik, resep obat sesuai dosis",
                },
                "medicines": [{"name": "Meactrin tab 500 mg", "quantity": 1, "unit": None, "dosage_instruction": None, "unit_price": 78900, "total_price": 78900}],
                "other_charges": [{"category": "CONSULTATION", "description": "Biaya konsultasi dokter", "amount": 218000}],
                "financial_recap": {"total_medicines": 78900, "total_other_charges": 218000, "grand_total": 296900},
            },
        ],
        "model-secondary": [
            {
                **SCAN_RESULT,
                "summary": {
                    "detected_documents": ["Formulir Rawat Jalan", "Salinan Resep", "Kwitansi", "Kwitansi"],
                    "medical_facility_name": "Klinik Utama Makmuri",
                    "service_date": "2025-08-28",
                    "transaction_date": "2025-08-31",
                    "currency": "IDR",
                    "total_amount": 356900,
                },
                "patient_info": {
                    "name": "Nick Frederic Kristian",
                    "date_of_birth": None,
                    "age": 13,
                    "gender": "MALE",
                    "address": None,
                },
                "medical_context": {
                    "anamnesis": "demam, batuk pilek lama, nyeri",
                    "diagnoses": [{"description": "pharyngitis + asthma", "doctor_name": "dr. Makmuri MS, Sp.A(K)"}],
                    "therapy_note": "Policare",
                },
                "medicines": [{"name": "Montelukast", "quantity": 10, "unit": "tablet", "dosage_instruction": "S 1 dd 1", "unit_price": None, "total_price": 78900}],
                "other_charges": [{"category": "SERVICE", "description": "Pembayaran Resep dr. Makmuri", "amount": 278000}],
                "financial_recap": {"total_medicines": 78900, "total_other_charges": 278000, "grand_total": 356900},
            },
            {
                **SCAN_RESULT,
                "summary": {
                    "detected_documents": ["Formulir Rawat Jalan", "Salinan Resep", "Kwitansi", "Kwitansi"],
                    "medical_facility_name": "Klinik Utama Makmuri",
                    "service_date": "2025-08-28",
                    "transaction_date": "2025-08-31",
                    "currency": "IDR",
                    "total_amount": 356900,
                },
                "patient_info": {
                    "name": "Nick Frederic Kristian",
                    "date_of_birth": None,
                    "age": 13,
                    "gender": "MALE",
                    "address": None,
                },
                "medical_context": {
                    "anamnesis": "fans, batuk pilek lama, nyeri",
                    "diagnoses": [{"description": "pharyngitis + asthma", "doctor_name": "dr. Makmuri MS, Sp.A(K)"}],
                    "therapy_note": "Policare",
                },
                "medicines": [{"name": "Montelukast tab 500 mg", "quantity": 10, "unit": "tablet", "dosage_instruction": "S 1 dd 1", "unit_price": None, "total_price": 78900}],
                "other_charges": [{"category": "OTHER", "description": "Resep dr. Makmuri", "amount": 278000}],
                "financial_recap": {"total_medicines": 78900, "total_other_charges": 278000, "grand_total": 356900},
            },
        ],
    }
    call_counts = {model_name: 0 for model_name in outputs}
    financial_judge_calls = 0

    class FakeModel:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        async def ainvoke(self, messages: list[object]) -> AIMessage:
            del messages
            call_index = call_counts[self.model_name]
            call_counts[self.model_name] += 1
            return AIMessage(content=json.dumps(outputs[self.model_name][call_index]))

    async def field_judge(
        field_path: str,
        images: list[object],
        candidates: list[dict[str, object]],
    ) -> tuple[str | None, str | None]:
        del field_path, images, candidates
        return None, "Not needed for this test"

    async def packet_judge(
        images: list[object],
        candidates: list[dict[str, object]],
    ) -> tuple[int | None, str | None]:
        nonlocal financial_judge_calls
        del images, candidates
        financial_judge_calls += 1
        return None, "Ambiguous receipt evidence"

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
            AIModelSpec(provider="google_genai", model="model-secondary"),
        ],
    )
    monkeypatch.setattr(medical_processor, "_judge_field_conflict", field_judge)
    monkeypatch.setattr(medical_processor, "_judge_financial_packet_conflict", packet_judge)
    monkeypatch.setattr(
        MedicalExtractionFusionEngine,
        "_has_dominant_financial_candidate",
        lambda self, plausible_candidates: False,
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

    assert financial_judge_calls == 1
    assert result.result.summary.total_amount == 296900
    assert result.result.financial_recap.grand_total == 296900
    assert result.extraction_meta_payload["financial_judge_attempt_count"] == 1
    assert result.extraction_meta_payload["financial_judge_applied_count"] == 0
    assert result.extraction_meta_payload["financial_judge_skipped_reason"] is None
    assert result.review_required is True


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
    assert grand_total_check["used_derived_value"] is True
    assert result.result.financial_recap.grand_total is not None
    assert result.result.summary.total_amount is not None
    assert result.result.financial_recap.grand_total == result.result.summary.total_amount
    assert result.review_required is True
    assert len(result.review_reasons) > 0
    assert {
        "field_path": "financial_recap.grand_total",
        "value": result.result.financial_recap.grand_total,
    } in result.extraction_meta_payload["derived_financials"]["forced_non_null_fields"]


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
    monkeypatch.setattr(
        medical_processor,
        "_judge_financial_packet_conflict",
        lambda images, candidates: asyncio.sleep(
            0,
            result=(
                next(index for index, candidate in enumerate(candidates) if candidate["grand_total"] == 597900),
                "Receipt-backed 597900 ledger is strongest.",
            ),
        ),
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
    assert result.extraction_meta_payload["financial_resolution_mode"] == "adjudicated"
    assert result.extraction_meta_payload["financial_judge_attempt_count"] == 1
    assert result.extraction_meta_payload["financial_judge_applied_count"] == 1
    assert result.extraction_meta_payload["financial_judge_skipped_reason"] is None
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


def test_financial_dominance_helper_recognizes_clear_receipt_backed_winner() -> None:
    async def noop_judge(
        field_path: str,
        images: list[object],
        candidates: list[dict[str, object]],
    ) -> tuple[str | None, str | None]:
        del field_path, images, candidates
        return None, None

    engine = MedicalExtractionFusionEngine(judge_field=noop_judge)
    plausible_candidates = [
        {
            "evidence_summary": {
                "evidence_strength": 8.2,
                "grand_total_match_score": 2.4,
                "other_total_match_score": 2.2,
            }
        },
        {
            "evidence_summary": {
                "evidence_strength": 6.3,
                "grand_total_match_score": 2.0,
                "other_total_match_score": 1.1,
            }
        },
    ]

    assert engine._has_dominant_financial_candidate(plausible_candidates) is True



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
    assert "confidence_penalty" not in anamnesis_meta
    assert "handwritten_uncertainty_penalty" in anamnesis_meta["review_flags"]
    assert any(
        "handwritten uncertainty" in hit
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
    assert "confidence_penalty" not in anamnesis_meta
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


def test_process_preserves_raw_structured_medical_terms_without_hardcoded_canonicalization(
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

    assert result.result.medical_context.diagnoses[0].description == "pertesis"
    assert result.extraction_meta_payload["normalized_terms"] == []


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


def test_document_validation_processor_accepts_strict_document_identity_from_visible_stamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validation_processor = DocumentValidationProcessor(settings=build_settings())
    stamped_medical_form_result = {
        "images": [
            {
                "image_index": 0,
                "filename": "scan.jpg",
                "document_count_detected": 1,
                "documents": [
                    {
                        "document_index": 0,
                        "document_type": "medical_form",
                        "is_medical_document": True,
                        "readability_ok": True,
                        "facility_identity_visible": False,
                        "stamp_detected": True,
                        "signature_detected": True,
                        "decision": "valid",
                        "reasons": ["Treating doctor stamp is visible but clinic field is blank."],
                    }
                ],
            }
        ]
    }

    class FakeModel:
        async def ainvoke(self, messages: list[object]) -> AIMessage:
            assert len(messages) == 2
            return AIMessage(content=json.dumps(stamped_medical_form_result))

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
        "Image 1 document 1: provider identity accepted from visible official stamp evidence."
    ]
    assert result.model_dump() == {
        "packet_patient_name": None,
        "patient_name_match_status": "not_checked",
        "patient_name_match_reason": None,
        "review_required": True,
        "review_reasons": [
            "Image 1 document 1: provider identity accepted from visible official stamp evidence."
        ],
        **stamped_medical_form_result,
    }


def test_document_validation_processor_rejects_strict_document_identity_from_signature_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validation_processor = DocumentValidationProcessor(settings=build_settings())
    signature_only_result = {
        "images": [
            {
                "image_index": 0,
                "filename": "scan.jpg",
                "document_count_detected": 1,
                "documents": [
                    {
                        "document_index": 0,
                        "document_type": "medical_form",
                        "is_medical_document": True,
                        "readability_ok": True,
                        "facility_identity_visible": False,
                        "stamp_detected": False,
                        "signature_detected": True,
                        "decision": "valid",
                        "reasons": ["Doctor signature is visible but provider identity is not clear."],
                    }
                ],
            }
        ]
    }

    class FakeModel:
        async def ainvoke(self, messages: list[object]) -> AIMessage:
            assert len(messages) == 2
            return AIMessage(content=json.dumps(signature_only_result))

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
    assert "medical facility identity is not clearly visible" in detail["errors"][0]["message"]


def test_document_validation_processor_rejects_receipt_without_provider_identity(
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
                        "facility_identity_visible": False,
                        "stamp_detected": True,
                        "signature_detected": False,
                        "decision": "valid",
                        "reasons": ["Only payer branding is visible."],
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
    assert "medical facility identity is not clearly visible" in detail["errors"][0]["message"]


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
