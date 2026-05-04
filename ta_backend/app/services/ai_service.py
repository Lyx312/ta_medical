import asyncio
import json
from importlib import import_module
from dataclasses import dataclass, replace
from pathlib import Path
import time
from typing import Any, cast

import yaml
from fastapi import HTTPException, status
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from pydantic import ValidationError

from app.core.config import ROOT_DIR, Settings
from app.db.session import get_session_factory
from app.repositories.ai_model_repository import AIModelRepository
from app.repositories.scan_log_repository import AIApiCallLogCreate, ScanLogRepository
from app.schemas.ai import AIMedicalScanResponse, DocumentValidationResponse, ValidationPatientNameMatchStatus
from app.services.ai_ensemble_config import (
    AI_EXTRACTION_MAX_PRODUCERS,
    AI_EXTRACTION_MIN_SUCCESSFUL_RUNS,
    AI_EXTRACTION_RUNS_PER_MODEL,
    AI_EXTRACTION_TEMPERATURE,
    CALIBRATION_VERSION,
    FUSION_VERSION,
)
from app.services.ai_ensemble import (
    MedicalExtractionFusionEngine,
    MedicalExtractionOutcome,
    ProducerInferenceTrace,
    SuccessfulProducerInferenceTrace,
    to_successful_trace,
)
from app.services.ai_provider_catalog import (
    get_ai_provider_catalog_entry,
    is_provider_scan_compatible,
)


PROMPT_FILE_PATH = ROOT_DIR / "app" / "prompts" / "ai" / "medical_document_scan.yaml"
VALIDATION_PROMPT_FILE_PATH = ROOT_DIR / "app" / "prompts" / "ai" / "medical_document_validation.yaml"
FIELD_JUDGE_PROMPT_FILE_PATH = ROOT_DIR / "app" / "prompts" / "ai" / "medical_document_field_judge.yaml"
STRICT_STAMP_OR_SIGNATURE_REQUIRED_TYPES = {"medical_form", "doctor_note", "prescription"}
STAMP_OR_SIGNATURE_OPTIONAL_TYPES = {"receipt", "lab_result", "radiology_result"}
SUPPORTED_MEDICAL_DOCUMENT_TYPES = STRICT_STAMP_OR_SIGNATURE_REQUIRED_TYPES | STAMP_OR_SIGNATURE_OPTIONAL_TYPES
MIN_FUZZY_NAME_LENGTH = 4
OLLAMA_CLOUD_BASE_URL = "https://ollama.com"
ORIENTATION_REASON_KEYWORDS = (
    "upside down",
    "sideways",
    "rotated",
    "rotation",
    "clockwise",
    "counterclockwise",
    "orientation",
)
NON_ORIENTATION_READABILITY_KEYWORDS = (
    "blur",
    "blurry",
    "cropped",
    "crop",
    "occluded",
    "occlusion",
    "glare",
    "shadow",
    "cut off",
    "unreadable",
    "too dark",
    "low resolution",
    "small text",
)


@dataclass(frozen=True)
class DocumentImageInput:
    filename: str
    mime_type: str
    base64_data: str


@dataclass(frozen=True)
class AIModelSpec:
    provider: str
    model: str
    priority: int = 100
    enabled: bool = True
    temperature: float = 0.0


@dataclass(frozen=True)
class PromptBundle:
    system_template: str
    user_template: str
    output_contract: dict[str, Any]


def normalize_patient_name(name: str) -> str:
    return " ".join(name.strip().casefold().split())


def is_single_typo_match(left: str, right: str) -> bool:
    if left == right:
        return True

    if len(left) < MIN_FUZZY_NAME_LENGTH or len(right) < MIN_FUZZY_NAME_LENGTH:
        return False

    if abs(len(left) - len(right)) > 1:
        return False

    if len(left) == len(right):
        mismatches = sum(1 for left_char, right_char in zip(left, right) if left_char != right_char)
        return mismatches <= 1

    shorter, longer = (left, right) if len(left) < len(right) else (right, left)
    short_index = 0
    long_index = 0
    mismatch_found = False

    while short_index < len(shorter) and long_index < len(longer):
        if shorter[short_index] == longer[long_index]:
            short_index += 1
            long_index += 1
            continue

        if mismatch_found:
            return False

        mismatch_found = True
        long_index += 1

    return True


def matches_any_patient_name_candidate(extracted_name: str | None, candidates: list[str]) -> bool:
    if extracted_name is None or not extracted_name.strip():
        return True

    normalized_extracted_name = normalize_patient_name(extracted_name)
    normalized_candidates = [
        normalize_patient_name(candidate)
        for candidate in candidates
        if candidate.strip()
    ]

    if normalized_extracted_name in normalized_candidates:
        return True

    return any(
        is_single_typo_match(normalized_extracted_name, normalized_candidate)
        for normalized_candidate in normalized_candidates
    )


def build_validation_packet_hints(
    validation_result: DocumentValidationResponse | None,
) -> dict[str, Any]:
    if validation_result is None:
        return {}

    ordered_documents: list[dict[str, Any]] = []
    document_type_counts: dict[str, int] = {}
    image_document_counts: list[dict[str, Any]] = []

    for image_result in sorted(validation_result.images, key=lambda item: item.image_index):
        image_document_counts.append(
            {
                "image_index": image_result.image_index,
                "filename": image_result.filename,
                "document_count_detected": image_result.document_count_detected,
                "validated_document_count": len(image_result.documents),
            }
        )
        for document in sorted(image_result.documents, key=lambda item: item.document_index):
            if not document.is_medical_document or document.document_type in {"non_medical", "unknown"}:
                continue
            ordered_documents.append(
                {
                    "image_index": image_result.image_index,
                    "document_index": document.document_index,
                    "document_type": document.document_type,
                    "decision": document.decision,
                }
            )
            document_type_counts[document.document_type] = document_type_counts.get(document.document_type, 0) + 1

    return {
        "ordered_documents": ordered_documents,
        "document_type_counts": document_type_counts,
        "image_document_counts": image_document_counts,
        "total_detected_documents": sum(max(item.document_count_detected, len(item.documents)) for item in validation_result.images),
    }


def build_patient_name_verification_error(message: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        detail={
            "message": "Patient name verification failed.",
            "errors": [
                {
                    "field": "non_field",
                    "message": message,
                }
            ],
        },
    )


class BaseDocumentAIProcessor:
    def __init__(
        self,
        settings: Settings,
        prompt_file_path: Path | None = None,
    ) -> None:
        self._settings = settings
        self._prompt_file_path = prompt_file_path or PROMPT_FILE_PATH

    def _load_prompt_bundle(self, prompt_file_path: Path | None = None) -> PromptBundle:
        target_path = prompt_file_path or self._prompt_file_path
        try:
            with target_path.open("r", encoding="utf-8") as prompt_file:
                prompt_config = yaml.safe_load(prompt_file) or {}
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="AI prompt template file is missing.",
            ) from exc

        system_template = prompt_config.get("system_template")
        user_template = prompt_config.get("user_template")
        output_contract = prompt_config.get("output_contract")

        if not isinstance(system_template, str) or not isinstance(user_template, str):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="AI prompt template file is invalid.",
            )
        if not isinstance(output_contract, dict):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="AI prompt output contract must be a mapping.",
            )

        return PromptBundle(
            system_template=system_template,
            user_template=user_template,
            output_contract=output_contract,
        )

    def _get_enabled_model_specs(self) -> list[AIModelSpec]:
        session_factory = get_session_factory()
        with session_factory() as db:
            repository = AIModelRepository(db)
            active_models = repository.list_active()

        enabled_models = [
            AIModelSpec(
                provider=model.provider,
                model=model.model_name,
                priority=model.priority,
                enabled=model.is_active,
            )
            for model in active_models
        ]
        if not enabled_models:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="No active AI model is configured.",
            )

        scan_compatible_models = [
            model_spec
            for model_spec in enabled_models
            if is_provider_scan_compatible(model_spec.provider)
        ]
        if not scan_compatible_models:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Active AI models are configured, but none currently support this image-based scan pipeline.",
            )
        return scan_compatible_models

    def get_primary_enabled_model_spec(self) -> AIModelSpec:
        return self._get_enabled_model_specs()[0]

    def _create_chat_model(self, model_spec: AIModelSpec) -> Any:
        provider_entry = get_ai_provider_catalog_entry(model_spec.provider)
        if provider_entry is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Unsupported AI provider config: {model_spec.provider}",
            )

        credential = getattr(self._settings, provider_entry.settings_attr, "")
        if not credential:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"{provider_entry.credential_env_var} is not configured.",
            )

        try:
            provider_module = import_module(provider_entry.package_name)
            chat_model_class = getattr(provider_module, provider_entry.class_name)
        except (ImportError, AttributeError) as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"The {provider_entry.package_name.replace('_', '-')} package is not installed.",
            ) from exc

        kwargs: dict[str, Any] = {
            "model": model_spec.model,
            "temperature": model_spec.temperature,
        }

        if model_spec.provider == "google_genai":
            kwargs["google_api_key"] = credential
        elif model_spec.provider == "openrouter":
            kwargs["api_key"] = credential
            # Avoid LangChain OpenRouter's default attribution fields because the
            # installed openrouter SDK version in this environment does not
            # accept the legacy `x_title` constructor argument it forwards.
            kwargs["app_title"] = None
            kwargs["app_url"] = None
        elif model_spec.provider == "ollama":
            kwargs["base_url"] = OLLAMA_CLOUD_BASE_URL
            kwargs["client_kwargs"] = {
                "headers": {
                    "Authorization": f"Bearer {credential}",
                },
            }
        elif model_spec.provider == "groq":
            kwargs["api_key"] = credential
        elif model_spec.provider == "mistralai":
            kwargs["mistral_api_key"] = credential
        elif model_spec.provider == "deepseek":
            kwargs["api_key"] = credential
        elif model_spec.provider == "cohere":
            kwargs["cohere_api_key"] = credential
        elif model_spec.provider == "cerebras":
            kwargs["api_key"] = credential

        return chat_model_class(**kwargs)

    def _normalize_message_content(self, messages: list[Any]) -> list[Any]:
        return messages

    def _parse_ai_response_to_json(self, content: Any) -> dict[str, Any]:
        text_content = self._extract_text_content(content)
        candidate_payload = self._strip_code_fences(text_content)

        if not candidate_payload.startswith("{") or not candidate_payload.endswith("}"):
            start = candidate_payload.find("{")
            end = candidate_payload.rfind("}")
            if start == -1 or end == -1 or start >= end:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail="AI model returned non-JSON response.",
                )
            candidate_payload = candidate_payload[start:end + 1]

        try:
            parsed_payload = json.loads(candidate_payload)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="AI model returned non-JSON response.",
            ) from exc

        if not isinstance(parsed_payload, dict):
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="AI model returned invalid JSON payload.",
            )

        return parsed_payload

    def _extract_text_content(self, content: Any) -> str:
        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            text_parts: list[str] = []
            for block in content:
                if isinstance(block, str):
                    text_parts.append(block)
                    continue
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    text_parts.append(block["text"])

            combined = "\n".join(part.strip() for part in text_parts if part.strip()).strip()
            if combined:
                return combined

        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="AI model did not return textual JSON output.",
        )

    def _strip_code_fences(self, text: str) -> str:
        stripped = text.strip()
        if not stripped.startswith("```"):
            return stripped

        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()

    def _build_image_messages(self, prompt_text: str, images: list[DocumentImageInput]) -> list[Any]:
        human_content: list[dict[str, object]] = [
            {
                "type": "text",
                "text": prompt_text,
            }
        ]
        for image in images:
            human_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{image.mime_type};base64,{image.base64_data}",
                    },
                }
            )
        return cast(list[Any], human_content)

    def _extract_usage_metadata(self, response: Any) -> dict[str, Any]:
        usage_sources: list[Any] = []
        for attr_name in ("usage_metadata", "response_metadata"):
            value = getattr(response, attr_name, None)
            if value is not None:
                usage_sources.append(value)

        prompt_tokens: int | None = None
        completion_tokens: int | None = None
        total_tokens: int | None = None
        raw_usage: dict[str, Any] | None = None

        for source in usage_sources:
            if isinstance(source, dict):
                raw_usage = source if raw_usage is None else {**raw_usage, **source}
                token_usage = source.get("token_usage") if isinstance(source.get("token_usage"), dict) else None
                candidates = [source]
                if token_usage is not None:
                    candidates.insert(0, token_usage)
                for candidate in candidates:
                    if prompt_tokens is None:
                        prompt_tokens = self._coerce_usage_int(candidate.get("prompt_tokens") or candidate.get("input_tokens") or candidate.get("prompt_token_count") or candidate.get("input_token_count"))
                    if completion_tokens is None:
                        completion_tokens = self._coerce_usage_int(candidate.get("completion_tokens") or candidate.get("output_tokens") or candidate.get("candidates_token_count") or candidate.get("output_token_count"))
                    if total_tokens is None:
                        total_tokens = self._coerce_usage_int(candidate.get("total_tokens") or candidate.get("total_token_count"))
            if prompt_tokens is not None or completion_tokens is not None or total_tokens is not None:
                break

        if total_tokens is None and (prompt_tokens is not None or completion_tokens is not None):
            total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)

        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "usage_payload": raw_usage,
        }

    def _coerce_usage_int(self, value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _persist_ai_api_call_log(self, payload: AIApiCallLogCreate) -> None:
        try:
            session_factory = get_session_factory()
            with session_factory() as db:
                ScanLogRepository(db).create_ai_api_call_log(payload)
        except Exception:
            return

    async def _ainvoke_with_tracking(
        self,
        model: Any,
        messages: list[Any],
        *,
        provider: str,
        model_name: str,
        call_kind: str,
        success_context: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        context = dict(success_context or {})
        started_at = time.perf_counter()
        response: Any | None = None
        error_message: str | None = None
        success = False
        try:
            response = await model.ainvoke(messages)
            success = True
            return response, self._build_ai_call_tracking_payload(
                response=response,
                provider=provider,
                model_name=model_name,
                call_kind=call_kind,
                context=context,
                success=True,
                error_message=None,
                started_at=started_at,
            )
        except Exception as exc:  # noqa: BLE001
            error_message = str(exc) or exc.__class__.__name__
            tracking_payload = self._build_ai_call_tracking_payload(
                response=response,
                provider=provider,
                model_name=model_name,
                call_kind=call_kind,
                context=context,
                success=False,
                error_message=error_message,
                started_at=started_at,
            )
            self._persist_ai_api_call_log(AIApiCallLogCreate(**tracking_payload))
            raise
        finally:
            if success and response is not None:
                pass

    def _build_ai_call_tracking_payload(
        self,
        *,
        response: Any,
        provider: str,
        model_name: str,
        call_kind: str,
        context: dict[str, Any],
        success: bool,
        error_message: str | None,
        started_at: float,
    ) -> dict[str, Any]:
        usage = self._extract_usage_metadata(response) if response is not None else {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "usage_payload": None,
        }
        tracking_payload = {
            "scan_log_id": context.get("scan_log_id"),
            "user_id": context.get("user_id"),
            "provider": provider,
            "model_name": model_name,
            "call_kind": call_kind,
            "producer_id": context.get("producer_id"),
            "producer_rank": context.get("producer_rank"),
            "run_index": context.get("run_index"),
            "temperature": context.get("temperature"),
            "success": success,
            "error_message": error_message,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "latency_ms": int(round((time.perf_counter() - started_at) * 1000)),
            "usage_payload": usage.get("usage_payload"),
            "request_context_payload": context.get("request_context_payload"),
        }
        if success:
            self._persist_ai_api_call_log(AIApiCallLogCreate(**tracking_payload))
        return tracking_payload


class MedicalDocumentProcessor(BaseDocumentAIProcessor):
    async def process(
        self,
        images: list[DocumentImageInput],
        patient_name_candidates: list[str] | None = None,
        validation_hints: dict[str, Any] | None = None,
        ai_call_context: dict[str, Any] | None = None,
    ) -> MedicalExtractionOutcome:
        if not images:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="At least one image file is required.",
            )

        prompt_bundle = self._load_prompt_bundle()
        model_specs = self._get_enabled_model_specs()[:AI_EXTRACTION_MAX_PRODUCERS]
        execution_mode = "ensemble" if len(model_specs) > 1 or AI_EXTRACTION_RUNS_PER_MODEL > 1 else "single_model"

        traces = await asyncio.gather(
            *[
                self._run_single_inference_trace(
                    model_spec=replace(model_spec, temperature=AI_EXTRACTION_TEMPERATURE),
                    producer_rank=producer_rank,
                    run_index=run_index,
                    images=images,
                    patient_name_candidates=patient_name_candidates or [],
                    validation_hints=validation_hints or {},
                    prompt_bundle=prompt_bundle,
                    ai_call_context=ai_call_context,
                )
                for producer_rank, model_spec in enumerate(model_specs)
                for run_index in range(AI_EXTRACTION_RUNS_PER_MODEL)
            ]
        )

        successful_traces = [
            to_successful_trace(trace)
            for trace in traces
            if trace.success and trace.payload is not None
        ]
        if len(successful_traces) < AI_EXTRACTION_MIN_SUCCESSFUL_RUNS:
            error = HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="AI ensemble failed to produce enough valid outputs.",
            )
            setattr(error, "ai_execution_mode", execution_mode)
            setattr(error, "ai_runs", list(traces))
            setattr(error, "primary_model_provider", model_specs[0].provider if model_specs else None)
            setattr(error, "primary_model_name", model_specs[0].model if model_specs else None)
            setattr(
                error,
                "extraction_meta_payload",
                {
                    "fusion_version": FUSION_VERSION,
                    "calibration_version": CALIBRATION_VERSION,
                    "review_required": True,
                    "review_reasons": ["AI ensemble did not produce enough successful runs."],
                },
            )
            raise error

        async def judge_field(field_path: str, judge_images: list[Any], candidates: list[dict[str, Any]]) -> tuple[str | None, str | None]:
            try:
                return await self._judge_field_conflict(
                    field_path,
                    judge_images,
                    candidates,
                    ai_call_context=ai_call_context,
                )
            except TypeError:
                return await self._judge_field_conflict(field_path, judge_images, candidates)

        fusion_engine = MedicalExtractionFusionEngine(judge_field=judge_field)
        return await fusion_engine.fuse(
            successful_traces=successful_traces,
            all_traces=list(traces),
            images=images,
            validation_hints=validation_hints or {},
            execution_mode=execution_mode,
            primary_model_provider=model_specs[0].provider if model_specs else None,
            primary_model_name=model_specs[0].model if model_specs else None,
        )

    async def _run_single_inference_trace(
        self,
        *,
        model_spec: AIModelSpec,
        producer_rank: int,
        run_index: int,
        images: list[DocumentImageInput],
        patient_name_candidates: list[str],
        validation_hints: dict[str, Any],
        prompt_bundle: PromptBundle,
        ai_call_context: dict[str, Any] | None,
    ) -> ProducerInferenceTrace:
        producer_id = f"{model_spec.provider}:{model_spec.model}"
        try:
            payload, tracking_payload = await self._run_model_inference(
                model_spec=model_spec,
                images=images,
                patient_name_candidates=patient_name_candidates,
                validation_hints=validation_hints,
                prompt_bundle=prompt_bundle,
                ai_call_context={
                    **(ai_call_context or {}),
                    "producer_id": producer_id,
                    "producer_rank": producer_rank,
                    "run_index": run_index,
                    "temperature": model_spec.temperature,
                    "request_context_payload": {
                        "image_count": len(images),
                        "patient_name_candidate_count": len(patient_name_candidates),
                        "validation_hint_keys": sorted((validation_hints or {}).keys()),
                    },
                },
            )
            return ProducerInferenceTrace(
                producer_id=producer_id,
                provider=model_spec.provider,
                model=model_spec.model,
                producer_rank=producer_rank,
                run_index=run_index,
                temperature=model_spec.temperature,
                success=True,
                call_kind="scan_extraction",
                prompt_tokens=tracking_payload.get("prompt_tokens"),
                completion_tokens=tracking_payload.get("completion_tokens"),
                total_tokens=tracking_payload.get("total_tokens"),
                latency_ms=tracking_payload.get("latency_ms"),
                usage_payload=tracking_payload.get("usage_payload"),
                payload=payload,
            )
        except Exception as exc:  # noqa: BLE001
            return ProducerInferenceTrace(
                producer_id=producer_id,
                provider=model_spec.provider,
                model=model_spec.model,
                producer_rank=producer_rank,
                run_index=run_index,
                temperature=model_spec.temperature,
                success=False,
                call_kind="scan_extraction",
                error=self._describe_exception(exc),
            )

    async def _run_model_inference(
        self,
        model_spec: AIModelSpec,
        images: list[DocumentImageInput],
        patient_name_candidates: list[str],
        validation_hints: dict[str, Any],
        prompt_bundle: PromptBundle,
        ai_call_context: dict[str, Any] | None = None,
    ) -> tuple[AIMedicalScanResponse, dict[str, Any]]:
        model = self._create_chat_model(model_spec)
        messages = self._normalize_message_content(
            self._build_messages(
                images=images,
                patient_name_candidates=patient_name_candidates,
                validation_hints=validation_hints,
                prompt_bundle=prompt_bundle,
            )
        )
        response, tracking_payload = await self._ainvoke_with_tracking(
            model,
            messages,
            provider=model_spec.provider,
            model_name=model_spec.model,
            call_kind="scan_extraction",
            success_context=ai_call_context,
        )

        payload = self._parse_ai_response_to_json(response.content)

        try:
            return AIMedicalScanResponse.model_validate(payload), tracking_payload
        except ValidationError as exc:
            print(f"AI response validation error: {exc}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="AI model returned invalid response shape.",
            ) from exc

    def _build_messages(
        self,
        images: list[DocumentImageInput],
        patient_name_candidates: list[str],
        prompt_bundle: PromptBundle,
        validation_hints: dict[str, Any] | None = None,
    ) -> list[Any]:
        template_input = {
            "image_count": len(images),
            "output_contract_json": json.dumps(prompt_bundle.output_contract, indent=2),
            "patient_name_candidates_json": json.dumps(patient_name_candidates, ensure_ascii=False),
            "validation_packet_hints_json": json.dumps(validation_hints or {}, ensure_ascii=False, indent=2),
        }

        system_prompt = PromptTemplate.from_template(
            prompt_bundle.system_template,
        ).format(**template_input)
        user_prompt = PromptTemplate.from_template(
            prompt_bundle.user_template,
        ).format(**template_input)

        return [
            SystemMessage(content=system_prompt),
            HumanMessage(content=self._build_image_messages(user_prompt, images)),
        ]

    async def _judge_field_conflict(
        self,
        field_path: str,
        images: list[Any],
        candidates: list[dict[str, Any]],
        ai_call_context: dict[str, Any] | None = None,
    ) -> tuple[str | None, str | None]:
        judge_model_spec = replace(self.get_primary_enabled_model_spec(), temperature=0.0)
        prompt_bundle = self._load_prompt_bundle(FIELD_JUDGE_PROMPT_FILE_PATH)
        template_input = {
            "field_path": field_path,
            "candidates_json": json.dumps(
                [
                    {
                        "index": index,
                        "value": candidate["raw_value"],
                        "support_count": candidate["run_count"],
                        "producer_count": len(candidate["producers"]),
                    }
                    for index, candidate in enumerate(candidates)
                ],
                ensure_ascii=False,
                indent=2,
            ),
            "output_contract_json": json.dumps(prompt_bundle.output_contract, indent=2),
        }
        system_prompt = PromptTemplate.from_template(prompt_bundle.system_template).format(**template_input)
        user_prompt = PromptTemplate.from_template(prompt_bundle.user_template).format(**template_input)
        model = self._create_chat_model(judge_model_spec)
        response, _tracking_payload = await self._ainvoke_with_tracking(
            model,
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=self._build_image_messages(user_prompt, cast(list[DocumentImageInput], images))),
            ],
            provider=judge_model_spec.provider,
            model_name=judge_model_spec.model,
            call_kind="field_judge",
            success_context={
                **(ai_call_context or {}),
                "temperature": judge_model_spec.temperature,
                "request_context_payload": {
                    "field_path": field_path,
                    "candidate_count": len(candidates),
                    "image_count": len(images),
                },
            },
        )
        payload = self._parse_ai_response_to_json(response.content)
        selected_index = payload.get("selected_candidate_index")
        if selected_index is None or not isinstance(selected_index, int):
            return None, cast(str | None, payload.get("reason"))
        if selected_index < 0 or selected_index >= len(candidates):
            return None, cast(str | None, payload.get("reason"))
        return cast(str, candidates[selected_index]["normalized_key"]), cast(str | None, payload.get("reason"))

    def _describe_exception(self, exc: Exception) -> str:
        if isinstance(exc, HTTPException):
            if isinstance(exc.detail, str):
                return exc.detail
            if isinstance(exc.detail, dict):
                return str(exc.detail.get("message", "HTTP error"))
        return str(exc) or exc.__class__.__name__

class DocumentValidationProcessor(BaseDocumentAIProcessor):
    def __init__(
        self,
        settings: Settings,
        prompt_file_path: Path | None = None,
    ) -> None:
        super().__init__(settings=settings, prompt_file_path=prompt_file_path or VALIDATION_PROMPT_FILE_PATH)

    async def process(
        self,
        images: list[DocumentImageInput],
        patient_name_candidates: list[str] | None = None,
        expected_patient_name: str | None = None,
        ai_call_context: dict[str, Any] | None = None,
    ) -> DocumentValidationResponse:
        if not images:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="At least one image file is required.",
            )

        prompt_bundle = self._load_prompt_bundle()
        model_specs = self._get_enabled_model_specs()

        inference_results: list[DocumentValidationResponse] = []
        for model_spec in model_specs[:1]:
            inference_results.append(
                await self._run_model_inference(
                    model_spec=model_spec,
                    images=images,
                    prompt_bundle=prompt_bundle,
                    patient_name_candidates=patient_name_candidates or [],
                    expected_patient_name=expected_patient_name,
                    ai_call_context={
                        **(ai_call_context or {}),
                        "temperature": model_spec.temperature,
                        "request_context_payload": {
                            "image_count": len(images),
                            "patient_name_candidate_count": len(patient_name_candidates or []),
                            "has_expected_patient_name": expected_patient_name is not None,
                        },
                    },
                )
            )

        validation_result = self.combine_inference_results(inference_results)
        self._enforce_validation_rules(
            validation_result,
            patient_name_candidates=patient_name_candidates or [],
            expected_patient_name=expected_patient_name,
        )
        return validation_result

    def combine_inference_results(
        self,
        results: list[DocumentValidationResponse],
    ) -> DocumentValidationResponse:
        if not results:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="No AI validation result was produced.",
            )
        return results[0]

    async def _run_model_inference(
        self,
        model_spec: AIModelSpec,
        images: list[DocumentImageInput],
        prompt_bundle: PromptBundle,
        patient_name_candidates: list[str],
        expected_patient_name: str | None,
        ai_call_context: dict[str, Any] | None = None,
    ) -> DocumentValidationResponse:
        model = self._create_chat_model(model_spec)
        messages = self._build_messages(
            images=images,
            prompt_bundle=prompt_bundle,
            patient_name_candidates=patient_name_candidates,
            expected_patient_name=expected_patient_name,
        )
        response, _tracking_payload = await self._ainvoke_with_tracking(
            model,
            messages,
            provider=model_spec.provider,
            model_name=model_spec.model,
            call_kind="scan_validation",
            success_context=ai_call_context,
        )
        payload = self._parse_ai_response_to_json(response.content)

        try:
            return DocumentValidationResponse.model_validate(payload)
        except ValidationError as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="AI model returned invalid validation response shape.",
            ) from exc

    def _build_messages(
        self,
        images: list[DocumentImageInput],
        prompt_bundle: PromptBundle,
        patient_name_candidates: list[str],
        expected_patient_name: str | None,
    ) -> list[Any]:
        template_input = {
            "image_count": len(images),
            "filenames_json": json.dumps([image.filename for image in images], ensure_ascii=False),
            "patient_name_candidates_json": json.dumps(patient_name_candidates, ensure_ascii=False),
            "expected_patient_name_json": json.dumps(expected_patient_name, ensure_ascii=False),
            "output_contract_json": json.dumps(prompt_bundle.output_contract, indent=2),
        }

        system_prompt = PromptTemplate.from_template(
            prompt_bundle.system_template,
        ).format(**template_input)
        user_prompt = PromptTemplate.from_template(
            prompt_bundle.user_template,
        ).format(**template_input)

        return [
            SystemMessage(content=system_prompt),
            HumanMessage(content=self._build_image_messages(user_prompt, images)),
        ]

    def _enforce_validation_rules(
        self,
        result: DocumentValidationResponse,
        *,
        patient_name_candidates: list[str],
        expected_patient_name: str | None,
    ) -> None:
        failures: list[str] = []
        review_reasons: list[str] = []

        if not result.images:
            failures.append("No document images could be validated.")

        for image_result in result.images:
            image_label = f"Image {image_result.image_index + 1}"
            is_multi_document_image = image_result.document_count_detected > 1

            if image_result.document_count_detected <= 0:
                failures.append(f"{image_label}: no medical document could be detected.")

            if image_result.document_count_detected != len(image_result.documents):
                message = f"{image_label}: multiple documents were present but could not be validated clearly."
                if is_multi_document_image:
                    review_reasons.append(message)
                else:
                    failures.append(message)

            if not image_result.documents:
                failures.append(f"{image_label}: no documents could be validated.")
                continue

            for document in image_result.documents:
                document_label = f"{image_label} document {document.document_index + 1}"
                has_orientation_only_issue = self._is_non_blocking_orientation_issue(document.reasons)
                has_collage_ambiguity_issue = is_multi_document_image and self._is_collage_ambiguity_issue(document.reasons)

                if document.decision != "valid":
                    if has_orientation_only_issue:
                        continue
                    if has_collage_ambiguity_issue:
                        review_reasons.append(
                            self._build_document_failure(
                                document_label=document_label,
                                reasons=document.reasons,
                                fallback_message="multiple papers are present and should be uploaded separately for clearer validation.",
                            )
                        )
                        continue
                    failures.append(
                        self._build_document_failure(
                            document_label=document_label,
                            reasons=document.reasons,
                            fallback_message="the document could not be validated confidently. Please retake or reupload a clearer image.",
                        )
                    )
                    continue

                if document.document_type not in SUPPORTED_MEDICAL_DOCUMENT_TYPES:
                    failures.append(f"{document_label}: unsupported or non-medical document type.")
                    continue

                if not document.is_medical_document:
                    failures.append(f"{document_label}: the uploaded document is not recognized as a medical claim document.")
                    continue

                if not document.readability_ok and not has_orientation_only_issue:
                    failures.append(f"{document_label}: the document is unreadable, blurred, or cropped.")

                if not document.facility_identity_visible:
                    failures.append(f"{document_label}: the medical facility identity is not clearly visible.")

                if (
                    document.document_type in STRICT_STAMP_OR_SIGNATURE_REQUIRED_TYPES
                    and not (document.stamp_detected or document.signature_detected)
                ):
                    message = f"{document_label}: a visible facility stamp or doctor signature is required."
                    if is_multi_document_image:
                        review_reasons.append(message)
                    else:
                        failures.append(message)

        if failures:
            deduped_failures = list(dict.fromkeys(failures))
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail={
                    "message": "Document validation failed.",
                    "errors": [
                        {
                            "field": "non_field",
                            "message": self._build_validation_failure_message(deduped_failures),
                        }
                    ],
                },
            )

        (
            result.patient_name_match_status,
            result.patient_name_match_reason,
            patient_name_review_reason,
        ) = self._resolve_patient_name_match(
            packet_patient_name=result.packet_patient_name,
            model_status=result.patient_name_match_status,
            model_reason=result.patient_name_match_reason,
            patient_name_candidates=patient_name_candidates,
            expected_patient_name=expected_patient_name,
        )
        if patient_name_review_reason:
            review_reasons.append(patient_name_review_reason)

        result.review_reasons = list(dict.fromkeys(review_reasons))
        result.review_required = bool(result.review_reasons)

    def _build_document_failure(
        self,
        document_label: str,
        reasons: list[str],
        fallback_message: str,
    ) -> str:
        normalized_reasons = [reason.strip() for reason in reasons if reason.strip()]
        if normalized_reasons:
            return f"{document_label}: {normalized_reasons[0]}"

        return f"{document_label}: {fallback_message}"

    def _build_validation_failure_message(self, failures: list[str]) -> str:
        if len(failures) <= 3:
            return "; ".join(failures)

        return "; ".join([*failures[:3], f"and {len(failures) - 3} more issue(s)."])

    def _is_non_blocking_orientation_issue(self, reasons: list[str]) -> bool:
        normalized_reasons = [reason.strip().casefold() for reason in reasons if reason.strip()]
        if not normalized_reasons:
            return False

        for reason in normalized_reasons:
            if not any(keyword in reason for keyword in ORIENTATION_REASON_KEYWORDS):
                return False
            if any(keyword in reason for keyword in NON_ORIENTATION_READABILITY_KEYWORDS):
                return False

        return True

    def _is_collage_ambiguity_issue(self, reasons: list[str]) -> bool:
        normalized_reasons = [reason.strip().casefold() for reason in reasons if reason.strip()]
        if not normalized_reasons:
            return False

        collage_keywords = (
            "multiple papers",
            "multiple documents",
            "cannot be separated",
            "separation is unclear",
            "separate clearly",
            "collage",
        )
        blocking_keywords = (
            *NON_ORIENTATION_READABILITY_KEYWORDS,
            "non-medical",
            "not medical",
            "unsupported",
            "facility identity",
            "identity is not clearly visible",
        )

        has_collage_reason = any(
            any(keyword in reason for keyword in collage_keywords)
            for reason in normalized_reasons
        )
        if not has_collage_reason:
            return False

        return not any(
            any(keyword in reason for keyword in blocking_keywords)
            for reason in normalized_reasons
        )

    def _resolve_patient_name_match(
        self,
        *,
        packet_patient_name: str | None,
        model_status: ValidationPatientNameMatchStatus,
        model_reason: str | None,
        patient_name_candidates: list[str],
        expected_patient_name: str | None,
    ) -> tuple[ValidationPatientNameMatchStatus, str | None, str | None]:
        normalized_packet_name = packet_patient_name.strip() if isinstance(packet_patient_name, str) else ""
        normalized_model_reason = model_reason.strip() if isinstance(model_reason, str) and model_reason.strip() else None

        if not expected_patient_name and not patient_name_candidates:
            return "not_checked", None, None

        if model_status == "uncertain":
            review_reason = normalized_model_reason or "Patient name could not be verified confidently from the uploaded documents."
            return "uncertain", review_reason, review_reason

        if not normalized_packet_name:
            reason = normalized_model_reason or "Patient name is not visible clearly enough in the uploaded documents."
            return "not_found", reason, reason

        if expected_patient_name:
            if matches_any_patient_name_candidate(normalized_packet_name, [expected_patient_name]):
                return "matched", None, None

            return (
                "mismatched",
                f"Document patient name '{normalized_packet_name}' does not match the selected patient.",
                None,
            )

        if matches_any_patient_name_candidate(normalized_packet_name, patient_name_candidates):
            return "matched", None, None

        return (
            "mismatched",
            f"Document patient name '{normalized_packet_name}' does not match the user or any saved family member.",
            None,
        )
