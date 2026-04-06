import json
from importlib import import_module
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml
from fastapi import HTTPException, status
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from pydantic import ValidationError

from app.core.config import ROOT_DIR, Settings
from app.db.session import get_session_factory
from app.repositories.ai_model_repository import AIModelRepository
from app.schemas.ai import AIMedicalScanResponse, DocumentValidationResponse
from app.services.ai_provider_catalog import (
    get_ai_provider_catalog_entry,
    is_provider_scan_compatible,
)


PROMPT_FILE_PATH = ROOT_DIR / "app" / "prompts" / "ai" / "medical_document_scan.yaml"
VALIDATION_PROMPT_FILE_PATH = ROOT_DIR / "app" / "prompts" / "ai" / "medical_document_validation.yaml"
STAMP_OR_SIGNATURE_REQUIRED_TYPES = {"receipt", "medical_form", "doctor_note", "prescription"}
STAMP_OR_SIGNATURE_OPTIONAL_TYPES = {"lab_result", "radiology_result"}
SUPPORTED_MEDICAL_DOCUMENT_TYPES = STAMP_OR_SIGNATURE_REQUIRED_TYPES | STAMP_OR_SIGNATURE_OPTIONAL_TYPES
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


class BaseDocumentAIProcessor:
    def __init__(
        self,
        settings: Settings,
        prompt_file_path: Path | None = None,
    ) -> None:
        self._settings = settings
        self._prompt_file_path = prompt_file_path or PROMPT_FILE_PATH

    def _load_prompt_bundle(self) -> PromptBundle:
        try:
            with self._prompt_file_path.open("r", encoding="utf-8") as prompt_file:
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


class MedicalDocumentProcessor(BaseDocumentAIProcessor):
    async def process(
        self,
        images: list[DocumentImageInput],
        patient_name_candidates: list[str] | None = None,
    ) -> AIMedicalScanResponse:
        if not images:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="At least one image file is required.",
            )

        prompt_bundle = self._load_prompt_bundle()
        model_specs = self._get_enabled_model_specs()

        inference_results: list[AIMedicalScanResponse] = []
        for model_spec in model_specs[:1]:
            inference_results.append(
                await self._run_model_inference(
                    model_spec=model_spec,
                    images=images,
                    patient_name_candidates=patient_name_candidates or [],
                    prompt_bundle=prompt_bundle,
                )
            )

        return self.combine_inference_results(inference_results)

    def combine_inference_results(
        self,
        results: list[AIMedicalScanResponse],
    ) -> AIMedicalScanResponse:
        if not results:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="No AI inference result was produced.",
            )
        return results[0]

    async def _run_model_inference(
        self,
        model_spec: AIModelSpec,
        images: list[DocumentImageInput],
        patient_name_candidates: list[str],
        prompt_bundle: PromptBundle,
    ) -> AIMedicalScanResponse:
        model = self._create_chat_model(model_spec)
        messages = self._normalize_message_content(
            self._build_messages(
                images=images,
                patient_name_candidates=patient_name_candidates,
                prompt_bundle=prompt_bundle,
            )
        )
        response = await model.ainvoke(messages)

        payload = self._parse_ai_response_to_json(response.content)

        try:
            return AIMedicalScanResponse.model_validate(payload)
        except ValidationError as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="AI model returned invalid response shape.",
            ) from exc

    def _build_messages(
        self,
        images: list[DocumentImageInput],
        patient_name_candidates: list[str],
        prompt_bundle: PromptBundle,
    ) -> list[Any]:
        template_input = {
            "image_count": len(images),
            "output_contract_json": json.dumps(prompt_bundle.output_contract, indent=2),
            "patient_name_candidates_json": json.dumps(patient_name_candidates, ensure_ascii=False),
        }

        system_prompt = PromptTemplate.from_template(
            prompt_bundle.system_template,
        ).format(**template_input)
        user_prompt = PromptTemplate.from_template(
            prompt_bundle.user_template,
        ).format(**template_input)

        human_content: list[dict[str, object]] = [
            {
                "type": "text",
                "text": user_prompt,
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

        return [
            SystemMessage(content=system_prompt),
            HumanMessage(content=cast(list[str | dict[object, object]], human_content)),
        ]

class DocumentValidationProcessor(BaseDocumentAIProcessor):
    def __init__(
        self,
        settings: Settings,
        prompt_file_path: Path | None = None,
    ) -> None:
        super().__init__(settings=settings, prompt_file_path=prompt_file_path or VALIDATION_PROMPT_FILE_PATH)

    async def process(self, images: list[DocumentImageInput]) -> DocumentValidationResponse:
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
                )
            )

        validation_result = self.combine_inference_results(inference_results)
        self._enforce_validation_rules(validation_result)
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
    ) -> DocumentValidationResponse:
        model = self._create_chat_model(model_spec)
        messages = self._build_messages(images=images, prompt_bundle=prompt_bundle)
        response = await model.ainvoke(messages)
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
    ) -> list[Any]:
        template_input = {
            "image_count": len(images),
            "filenames_json": json.dumps([image.filename for image in images], ensure_ascii=False),
            "output_contract_json": json.dumps(prompt_bundle.output_contract, indent=2),
        }

        system_prompt = PromptTemplate.from_template(
            prompt_bundle.system_template,
        ).format(**template_input)
        user_prompt = PromptTemplate.from_template(
            prompt_bundle.user_template,
        ).format(**template_input)

        human_content: list[dict[str, object]] = [
            {
                "type": "text",
                "text": user_prompt,
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

        return [
            SystemMessage(content=system_prompt),
            HumanMessage(content=cast(list[str | dict[object, object]], human_content)),
        ]

    def _enforce_validation_rules(self, result: DocumentValidationResponse) -> None:
        failures: list[str] = []

        if not result.images:
            failures.append("No document images could be validated.")

        for image_result in result.images:
            image_label = f"Image {image_result.image_index + 1}"

            if image_result.document_count_detected <= 0:
                failures.append(f"{image_label}: no medical document could be detected.")

            if image_result.document_count_detected != len(image_result.documents):
                failures.append(
                    f"{image_label}: multiple documents were present but could not be validated clearly."
                )

            if not image_result.documents:
                failures.append(f"{image_label}: no documents could be validated.")
                continue

            for document in image_result.documents:
                document_label = f"{image_label} document {document.document_index + 1}"
                has_orientation_only_issue = self._is_non_blocking_orientation_issue(document.reasons)

                if document.decision != "valid":
                    if has_orientation_only_issue:
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
                    document.document_type in STAMP_OR_SIGNATURE_REQUIRED_TYPES
                    and not (document.stamp_detected or document.signature_detected)
                ):
                    failures.append(f"{document_label}: a visible facility stamp or doctor signature is required.")

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
