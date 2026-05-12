from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
import logging
import os
import time
from typing import Any, Iterator


AI_PIPELINE_LOGGER_NAME = "app.ai_pipeline"
_TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}
_OWNED_HANDLER_FLAG = "_ai_pipeline_owned_handler"
_MAX_TERMINAL_ERROR_LEN = 255
_SUPPRESSED_EVENT_KEYS = {
    ("document_validation_processor", "document_validation_started"),
    ("document_validation_processor", "document_validation_completed"),
    ("ai_scan_route", "text_extraction_started"),
    ("ai_scan_route", "text_extraction_completed"),
    ("ai_scan_route", "medicine_prioritization_started"),
    ("ai_scan_route", "medicine_prioritization_completed"),
    ("medical_document_processor", "field_judge_started"),
}


@lru_cache(maxsize=1)
def is_ai_pipeline_debug_logging_enabled() -> bool:
    return os.getenv("AI_PIPELINE_DEBUG_LOGS", "").strip().casefold() in _TRUTHY_ENV_VALUES


def clear_ai_pipeline_debug_logging_cache() -> None:
    is_ai_pipeline_debug_logging_enabled.cache_clear()
    _reset_ai_pipeline_logger()


def _owned_ai_pipeline_handlers(logger: logging.Logger) -> list[logging.Handler]:
    return [
        handler
        for handler in logger.handlers
        if bool(getattr(handler, _OWNED_HANDLER_FLAG, False))
    ]


def _remove_owned_ai_pipeline_handlers(logger: logging.Logger) -> None:
    for handler in _owned_ai_pipeline_handlers(logger):
        logger.removeHandler(handler)
        handler.close()


def _configure_ai_pipeline_logger(logger: logging.Logger) -> logging.Logger:
    if not is_ai_pipeline_debug_logging_enabled():
        _remove_owned_ai_pipeline_handlers(logger)
        logger.propagate = False
        logger.setLevel(logging.NOTSET)
        return logger

    logger.propagate = False
    logger.setLevel(logging.INFO)

    if not _owned_ai_pipeline_handlers(logger):
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter("%(message)s"))
        setattr(handler, _OWNED_HANDLER_FLAG, True)
        logger.addHandler(handler)

    return logger


def _reset_ai_pipeline_logger() -> None:
    logger = logging.getLogger(AI_PIPELINE_LOGGER_NAME)
    _remove_owned_ai_pipeline_handlers(logger)
    logger.propagate = False
    logger.setLevel(logging.NOTSET)


def get_ai_pipeline_logger() -> logging.Logger:
    logger = logging.getLogger(AI_PIPELINE_LOGGER_NAME)
    return _configure_ai_pipeline_logger(logger)


def ai_pipeline_context_fields(ai_call_context: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(ai_call_context, dict):
        return {}
    context_fields: dict[str, Any] = {}
    for key in ("scan_log_id", "user_id"):
        value = ai_call_context.get(key)
        if value is not None:
            context_fields[key] = value
    return context_fields


def _sanitize_error_message(message: str) -> str:
    normalized = " ".join(message.strip().split())
    if "For further information visit" in normalized:
        normalized = normalized.split("For further information visit", 1)[0].strip()
    if len(normalized) <= _MAX_TERMINAL_ERROR_LEN:
        return normalized
    return f"{normalized[:_MAX_TERMINAL_ERROR_LEN - 3]}..."


def ai_pipeline_exception_fields(exc: Exception) -> dict[str, Any]:
    message = str(exc).strip() or exc.__class__.__name__
    return {
        "error_type": exc.__class__.__name__,
        "error_message": _sanitize_error_message(message),
    }


def _bool_to_yes_no(value: Any) -> str:
    return "yes" if bool(value) else "no"


def _format_duration(latency_ms: Any) -> str | None:
    if latency_ms is None:
        return None
    try:
        milliseconds = float(latency_ms)
    except (TypeError, ValueError):
        return None
    return f"{milliseconds / 1000.0:.1f}s"


def _format_confidence(value: Any) -> str | None:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return None


def _scan_prefix(fields: dict[str, Any]) -> str:
    scan_log_id = fields.get("scan_log_id")
    if scan_log_id is None:
        return "[AI]"
    return f"[AI][scan:{scan_log_id}]"


def _compact_model_name(model: Any) -> str | None:
    if not isinstance(model, str) or not model.strip():
        return None
    return model.strip()


def _compact_run_label(fields: dict[str, Any]) -> str:
    producer_rank = fields.get("producer_rank")
    run_index = fields.get("run_index")
    try:
        if producer_rank is not None and run_index is not None:
            return f"p{int(producer_rank) + 1}r{int(run_index) + 1}"
    except (TypeError, ValueError):
        pass
    return "run"


def _join_parts(*parts: str | None) -> str:
    return " ".join(part for part in parts if part)


def format_ai_pipeline_terminal_line(event: str, fields: dict[str, Any]) -> str | None:
    component = str(fields.get("component") or "")
    if (component, event) in _SUPPRESSED_EVENT_KEYS:
        return None

    prefix = _scan_prefix(fields)
    model = _compact_model_name(fields.get("model"))
    duration = _format_duration(fields.get("latency_ms"))
    total_tokens = fields.get("total_tokens")
    raw_error_message = fields.get("error_message")
    error_message = _sanitize_error_message(str(raw_error_message)) if raw_error_message else None
    review_required = _bool_to_yes_no(fields.get("review_required"))

    if event == "scan_request_started":
        return _join_parts(prefix, "SCAN", "start", f"files={fields.get('upload_count', 0)}")
    if event == "image_preprocessing_completed":
        return _join_parts(prefix, "PREP", "done", f"images={fields.get('image_count', 0)}")
    if event == "document_validation_started":
        return _join_parts(prefix, "VALIDATE", "start", f"images={fields.get('image_count', 0)}")
    if event == "validation_inference_started":
        return _join_parts(prefix, "VALIDATE", f"model={model}" if model else None, "start", f"images={fields.get('image_count', 0)}")
    if event == "validation_inference_completed":
        return _join_parts(
            prefix,
            "VALIDATE",
            f"model={model}" if model else None,
            "done",
            duration,
            f"tok={total_tokens}" if total_tokens is not None else None,
        )
    if event == "validation_inference_failed":
        return _join_parts(prefix, "VALIDATE", f"model={model}" if model else None, "failed", f"err={error_message}" if error_message else None)
    if event == "document_validation_completed":
        return _join_parts(
            prefix,
            "VALIDATE",
            "done",
            f"status={fields.get('patient_name_match_status', 'unknown')}",
            f"review={review_required}",
        )
    if event == "document_validation_failed":
        return _join_parts(prefix, "VALIDATE", "failed", f"err={error_message}" if error_message else None)
    if event == "medical_document_process_started":
        return _join_parts(
            prefix,
            "EXTRACT",
            "start",
            f"mode={fields.get('execution_mode')}" if fields.get("execution_mode") else None,
            f"images={fields.get('image_count', 0)}",
            f"producers={fields.get('producer_count', 0)}x{fields.get('runs_per_model', 0)}",
        )
    if event == "extraction_inference_started":
        return _join_parts(prefix, "RUN", _compact_run_label(fields), model, "start")
    if event == "extraction_inference_completed":
        return _join_parts(
            prefix,
            "RUN",
            _compact_run_label(fields),
            model,
            "done",
            duration,
            f"tok={total_tokens}" if total_tokens is not None else None,
        )
    if event == "extraction_inference_failed":
        return _join_parts(prefix, "RUN", _compact_run_label(fields), model, "failed", f"err={error_message}" if error_message else None)
    if event == "medical_document_inference_runs_completed":
        return _join_parts(
            prefix,
            "EXTRACT",
            "runs",
            "done",
            f"ok={fields.get('successful_run_count', 0)}/{fields.get('total_run_count', 0)}",
        )
    if event == "field_judge_completed":
        return _join_parts(
            prefix,
            "JUDGE",
            "done",
            f"field={fields.get('field_path')}" if fields.get("field_path") else None,
            f"candidates={fields.get('candidate_count')}" if fields.get("candidate_count") is not None else None,
        )
    if event == "field_judge_failed":
        return _join_parts(prefix, "JUDGE", "failed", f"field={fields.get('field_path')}" if fields.get("field_path") else None, f"err={error_message}" if error_message else None)
    if event == "ensemble_fusion_started":
        return _join_parts(prefix, "FUSE", "start", f"runs={fields.get('successful_run_count', 0)}/{fields.get('total_run_count', 0)}")
    if event == "detected_document_fusion_completed":
        return _join_parts(
            prefix,
            "FUSE",
            "docs",
            f"count={fields.get('detected_document_count', 0)}",
            f"inventory_conflicts={fields.get('inventory_conflict_count', 0)}",
        )
    if event == "list_fusion_completed":
        list_name = str(fields.get("list_name") or "items")
        short_name = {"diagnoses": "diag", "medicines": "meds", "other_charges": "charges"}.get(list_name, list_name)
        return _join_parts(prefix, "FUSE", short_name, f"count={fields.get('item_count', 0)}")
    if event == "financial_reconciliation_completed":
        return _join_parts(
            prefix,
            "RECON",
            "done",
            f"corrected={fields.get('corrected_field_count', 0)}",
            f"unresolved={fields.get('unresolved_conflict_count', 0)}",
        )
    if event == "ensemble_fusion_completed":
        confidence = _format_confidence(fields.get("overall_confidence"))
        return _join_parts(
            prefix,
            "FUSE",
            "done",
            f"judges={fields.get('judge_invocation_count', 0)}",
            f"nulls={fields.get('nullified_field_count', 0)}",
            f"conf={confidence}" if confidence else None,
            f"review={review_required}",
        )
    if event == "medical_document_process_completed":
        confidence = _format_confidence(fields.get("overall_confidence"))
        return _join_parts(
            prefix,
            "EXTRACT",
            "done",
            f"runs={fields.get('successful_run_count', 0)}/{fields.get('total_run_count', 0)}",
            f"conf={confidence}" if confidence else None,
            f"review={review_required}",
        )
    if event == "medical_document_process_failed":
        return _join_parts(prefix, "EXTRACT", "failed", f"err={error_message}" if error_message else None)
    if event == "medicine_prioritization_started":
        return _join_parts(
            prefix,
            "MEDS",
            "start",
            f"count={fields.get('medicine_count', 0)}",
            f"mode={fields.get('interpretation_mode')}" if fields.get("interpretation_mode") else None,
        )
    if event == "medicine_prioritization_batch_prepared":
        return _join_parts(prefix, "MEDS", "batch", f"ambiguous={fields.get('ambiguous_count', 0)}")
    if event == "medicine_prompt_started":
        return _join_parts(
            prefix,
            "MEDS",
            "prompt",
            f"model={model}" if model else None,
            "start",
            f"ambiguous={fields.get('ambiguous_medicine_count', 0)}",
        )
    if event == "medicine_prompt_completed":
        return _join_parts(prefix, "MEDS", "prompt", "done", f"judgements={fields.get('judgement_count', 0)}")
    if event == "medicine_prompt_failed":
        return _join_parts(prefix, "MEDS", "prompt", "failed", f"err={error_message}" if error_message else None)
    if event == "medicine_prioritization_batch_failed":
        return _join_parts(prefix, "MEDS", "batch", "failed", f"err={error_message}" if error_message else None)
    if event == "medicine_prioritization_completed":
        return _join_parts(
            prefix,
            "MEDS",
            "done",
            f"count={fields.get('medicine_count', 0)}",
            f"ambiguous={fields.get('ambiguous_count', 0)}" if fields.get("ambiguous_count") is not None else None,
            f"applied={_bool_to_yes_no(fields.get('applied'))}",
            f"batch={_bool_to_yes_no(fields.get('batch_success'))}" if fields.get("batch_success") is not None else None,
        )
    if event == "medicine_prioritization_failed":
        return _join_parts(prefix, "MEDS", "failed", f"err={error_message}" if error_message else None)
    if event == "scan_request_completed":
        return _join_parts(
            prefix,
            "SCAN",
            "done",
            f"mode={fields.get('ai_execution_mode')}" if fields.get("ai_execution_mode") else None,
            f"runs={fields.get('ai_run_count', 0)}",
            f"review={review_required}",
        )
    if event == "scan_request_failed":
        return _join_parts(prefix, "SCAN", "failed", f"err={error_message}" if error_message else None)

    extra_parts = []
    for key in ("model", "field_path", "image_count", "candidate_count", "medicine_count", "latency_ms", "total_tokens"):
        value = fields.get(key)
        if value is not None:
            if key == "latency_ms":
                formatted = _format_duration(value)
                if formatted is not None:
                    extra_parts.append(formatted)
            elif key == "total_tokens":
                extra_parts.append(f"tok={value}")
            else:
                extra_parts.append(f"{key}={value}")
    return _join_parts(prefix, event, *extra_parts)


def ai_pipeline_log(*, event: str, component: str, **fields: Any) -> None:
    if not is_ai_pipeline_debug_logging_enabled():
        return
    payload = {
        "event": event,
        "component": component,
        **{key: value for key, value in fields.items() if value is not None},
    }
    rendered_line = format_ai_pipeline_terminal_line(event, payload)
    if rendered_line is None:
        return
    get_ai_pipeline_logger().info(rendered_line)


@contextmanager
def timed_stage(
    *,
    started_event: str,
    completed_event: str,
    failed_event: str,
    component: str,
    **fields: Any,
) -> Iterator[None]:
    started_at = time.perf_counter()
    ai_pipeline_log(event=started_event, component=component, **fields)
    try:
        yield
    except Exception as exc:
        ai_pipeline_log(
            event=failed_event,
            component=component,
            latency_ms=int(round((time.perf_counter() - started_at) * 1000)),
            success=False,
            **fields,
            **ai_pipeline_exception_fields(exc),
        )
        raise
    ai_pipeline_log(
        event=completed_event,
        component=component,
        latency_ms=int(round((time.perf_counter() - started_at) * 1000)),
        success=True,
        **fields,
    )
