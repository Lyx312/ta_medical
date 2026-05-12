from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, cast

from fastapi import HTTPException, status
from langchain_core.messages import HumanMessage

from app.core.config import Settings
from app.schemas.test import ModelInvokeTestResponse
from app.services.ai_provider_catalog import get_ai_provider_catalog_entry
from app.services.ai_service import AIModelSpec, BaseDocumentAIProcessor, DocumentImageInput


@dataclass(frozen=True)
class ModelInvokeParsedResult:
    raw_text: str
    parsed_json: dict[str, Any] | list[Any] | None


class ModelInvocationTestService(BaseDocumentAIProcessor):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings=settings)

    def _persist_ai_api_call_log(self, payload: Any) -> None:
        del payload
        return

    def _parse_optional_json(self, raw_text: str) -> dict[str, Any] | list[Any] | None:
        candidate_payload = self._strip_code_fences(raw_text)
        try:
            parsed_payload = json.loads(candidate_payload)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed_payload, (dict, list)):
            return cast(dict[str, Any] | list[Any], parsed_payload)
        return None

    def _build_test_messages(
        self,
        *,
        message: str,
        image: DocumentImageInput | None,
    ) -> list[Any]:
        if image is None:
            return [HumanMessage(content=message)]
        return [HumanMessage(content=self._build_image_messages(message, [image]))]

    async def invoke_model(
        self,
        *,
        provider: str,
        model_name: str,
        message: str,
        image: DocumentImageInput | None,
    ) -> ModelInvokeTestResponse:
        provider_entry = get_ai_provider_catalog_entry(provider)
        if provider_entry is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail={
                    "message": "Input validation failed.",
                    "errors": [
                        {
                            "field": "provider",
                            "message": "Provider is not registered.",
                        }
                    ],
                },
            )
        if image is not None and not provider_entry.supports_image_input:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail={
                    "message": "Input validation failed.",
                    "errors": [
                        {
                            "field": "image",
                            "message": "This provider does not support image input.",
                        }
                    ],
                },
            )

        model_spec = AIModelSpec(provider=provider, model=model_name, temperature=0.0)
        model = self._create_chat_model(model_spec)
        messages = self._normalize_message_content(
            self._build_test_messages(message=message, image=image)
        )
        try:
            response, tracking_payload = await self._ainvoke_with_tracking(
                model,
                messages,
                provider=provider,
                model_name=model_name,
                call_kind="model_test",
                success_context={
                    "temperature": model_spec.temperature,
                    "request_context_payload": {
                        "image_supplied": image is not None,
                    },
                },
            )
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail={
                    "message": "AI model invocation failed.",
                    "errors": [
                        {
                            "field": "non_field",
                            "message": str(exc) or exc.__class__.__name__,
                        }
                    ],
                },
            ) from exc

        raw_text = self._extract_text_content(response.content)
        parsed_json = self._parse_optional_json(raw_text)
        return ModelInvokeTestResponse(
            success=True,
            provider=provider,
            model_name=model_name,
            supports_image_input=provider_entry.supports_image_input,
            image_supplied=image is not None,
            latency_ms=int(tracking_payload.get("latency_ms") or 0),
            raw_text=raw_text,
            parsed_json=parsed_json,
            prompt_tokens=tracking_payload.get("prompt_tokens"),
            completion_tokens=tracking_payload.get("completion_tokens"),
            total_tokens=tracking_payload.get("total_tokens"),
        )
