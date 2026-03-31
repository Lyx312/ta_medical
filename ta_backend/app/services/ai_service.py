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
from app.schemas.ai import AIMedicalScanResponse


PROMPT_FILE_PATH = ROOT_DIR / "app" / "prompts" / "ai" / "medical_document_scan.yaml"


@dataclass(frozen=True)
class DocumentImageInput:
    filename: str
    mime_type: str
    base64_data: str


@dataclass(frozen=True)
class AIModelSpec:
    provider: str
    model: str
    enabled: bool = True
    temperature: float = 0.0


@dataclass(frozen=True)
class PromptBundle:
    system_template: str
    user_template: str
    output_contract: dict[str, Any]


ACTIVE_AI_MODELS = [
    AIModelSpec(provider="google_genai", model="gemini-3.1-flash-lite-preview", enabled=True),
]


class MedicalDocumentProcessor:
    def __init__(
        self,
        settings: Settings,
        prompt_file_path: Path | None = None,
    ) -> None:
        self._settings = settings
        self._prompt_file_path = prompt_file_path or PROMPT_FILE_PATH

    async def process(self, images: list[DocumentImageInput]) -> AIMedicalScanResponse:
        if not images:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
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
        print("Inference results:", results)
        return results[0]

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
        enabled_models = [model for model in ACTIVE_AI_MODELS if model.enabled]
        if not enabled_models:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="No enabled AI model is configured.",
            )
        return enabled_models

    async def _run_model_inference(
        self,
        model_spec: AIModelSpec,
        images: list[DocumentImageInput],
        prompt_bundle: PromptBundle,
    ) -> AIMedicalScanResponse:
        model = self._create_chat_model(model_spec)
        messages = self._normalize_message_content(
            self._build_messages(images=images, prompt_bundle=prompt_bundle)
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

    def _create_chat_model(self, model_spec: AIModelSpec) -> Any:
        if model_spec.provider != "google_genai":
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Unsupported AI provider config: {model_spec.provider}",
            )

        if not self._settings.google_api_key:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="GOOGLE_API_KEY is not configured.",
            )

        try:
            chat_google_genai = import_module("langchain_google_genai")
            chat_google_generative_ai = getattr(chat_google_genai, "ChatGoogleGenerativeAI")
        except (ImportError, AttributeError) as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="The langchain-google-genai package is not installed.",
            ) from exc

        return chat_google_generative_ai(
            model=model_spec.model,
            google_api_key=self._settings.google_api_key,
            temperature=model_spec.temperature,
        )

    def _build_messages(
        self,
        images: list[DocumentImageInput],
        prompt_bundle: PromptBundle,
    ) -> list[Any]:
        template_input = {
            "image_count": len(images),
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
