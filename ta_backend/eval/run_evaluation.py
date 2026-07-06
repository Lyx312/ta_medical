import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Literal
from itertools import zip_longest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db.init import initialize_database
from app.db.session import get_session_factory
from app.main import create_app
from app.repositories.scan_log_repository import ScanLogRepository
from app.repositories.user_repository import UserRepository
from eval.cli_utils import ensure_family_candidates, guess_mime_type, register_test_user
from eval.config import DEFAULT_THRESHOLDS
from eval.evaluator import EVALUATOR_SCHEMA_VERSION, PRIMARY_METRIC, MedicalClaimEvaluator
from eval.ground_truth import GroundTruthError, load_ground_truth
from eval.result_store import (
    ClaimFingerprintInput,
    append_run_record,
    build_claim_fingerprint,
    build_latest_snapshot,
    build_run_record,
    fingerprint_input_payload,
    load_latest_snapshot,
    read_json,
    resolve_run_record_path,
    save_latest_snapshot,
    sha256_file,
)
from fastapi.testclient import TestClient

EvalMode = Literal["auto", "eval-only", "scan-only"]
LogLevel = Literal["quiet", "info", "debug"]

LOG_LEVEL_PRIORITY: dict[str, int] = {
    "quiet": 0,
    "info": 1,
    "debug": 2,
}

EXCLUDED_FOLDER_NAMES = frozenset({"results", ".git", "__pycache__", ".venv", "venv"})

try:
    import colorama
    colorama.init()
    C_RESET = colorama.Fore.RESET
    C_GREEN = colorama.Fore.GREEN
    C_RED = colorama.Fore.RED
    C_YELLOW = colorama.Fore.YELLOW
    C_CYAN = colorama.Fore.CYAN
    C_BOLD = colorama.Style.BRIGHT
except Exception:  # noqa: BLE001
    C_RESET = ""
    C_GREEN = ""
    C_RED = ""
    C_YELLOW = ""
    C_CYAN = ""
    C_BOLD = ""


def _truncate_str(val: Any, max_len: int = 15) -> str:
    if val is None:
        return "-"
    s = str(val)
    if len(s) > max_len:
        return s[:max_len - 3] + "..."
    return s if s else "-"


def _zero_metric() -> dict[str, Any]:
    return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0.0, "fp": 0.0, "fn": 0.0, "applicable": False}


def _metric_from_counts(tp: float, fp: float, fn: float) -> dict[str, Any]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": round(tp, 4),
        "fp": round(fp, 4),
        "fn": round(fn, 4),
        "applicable": (tp + fp + fn) > 0,
    }


def _aggregate_metric_counts(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(float(metric.get("tp", 0.0)) for metric in metrics)
    fp = sum(float(metric.get("fp", 0.0)) for metric in metrics)
    fn = sum(float(metric.get("fn", 0.0)) for metric in metrics)
    return _metric_from_counts(tp, fp, fn)


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _applicable_metrics(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    applicable = [metric for metric in metrics if metric.get("applicable", True)]
    return applicable or metrics


def _get_primary_metric(score: dict[str, Any]) -> dict[str, Any]:
    metric = score.get("strict_kie_micro")
    return metric if isinstance(metric, dict) else _zero_metric()


def _get_usability_diagnostics(score: dict[str, Any]) -> dict[str, Any]:
    diagnostics = score.get("diagnostics", {})
    if not isinstance(diagnostics, dict):
        return {}
    usability = diagnostics.get("usability", {})
    return usability if isinstance(usability, dict) else {}


def _aggregate_accuracy(scores: list[dict[str, Any]], key: str) -> dict[str, Any]:
    correct = 0
    total = 0
    incorrect_fields: set[str] = set()
    for score in scores:
        accuracy = score.get(key, {})
        if not isinstance(accuracy, dict):
            continue
        correct += int(accuracy.get("correct", 0))
        total += int(accuracy.get("total", 0))
        incorrect_fields.update(str(field) for field in accuracy.get("incorrect_fields", []))
    return {
        "accuracy": round(correct / total, 4) if total else 0.0,
        "correct": correct,
        "total": total,
        "incorrect_fields": sorted(incorrect_fields),
    }


def _aggregate_relaxed_text(scores: list[dict[str, Any]]) -> dict[str, Any]:
    total_count = 0
    weighted_anls = 0.0
    weighted_cer = 0.0
    for score in scores:
        relaxed = score.get("relaxed_text", {})
        if not isinstance(relaxed, dict):
            continue
        count = int(relaxed.get("count", 0))
        total_count += count
        weighted_anls += float(relaxed.get("average_anls", 0.0)) * count
        weighted_cer += float(relaxed.get("average_cer", 0.0)) * count
    return {
        "average_anls": round(weighted_anls / total_count, 4) if total_count else 0.0,
        "average_cer": round(weighted_cer / total_count, 4) if total_count else 0.0,
        "count": total_count,
    }


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_ai_api_calls(result: dict[str, Any]) -> list[dict[str, Any]]:
    run_record = result.get("run_record")
    if not isinstance(run_record, dict):
        return []

    scan_log = run_record.get("scan_log")
    if not isinstance(scan_log, dict):
        return []

    calls = scan_log.get("ai_api_calls")
    if not isinstance(calls, list):
        return []

    return [call for call in calls if isinstance(call, dict)]


def _aggregate_token_usage(results: list[dict[str, Any]]) -> dict[str, Any]:
    claims_with_usage = 0
    call_count = 0
    successful_call_count = 0
    failed_call_count = 0
    prompt_total = 0.0
    completion_total = 0.0
    total_token_total = 0.0
    latency_total = 0.0
    latency_count = 0

    for result in results:
        claim_prompt_total = 0.0
        claim_completion_total = 0.0
        claim_total_token_total = 0.0
        claim_has_usage = False

        for call in _extract_ai_api_calls(result):
            prompt_tokens = _safe_float(call.get("prompt_tokens"))
            completion_tokens = _safe_float(call.get("completion_tokens"))
            total_tokens = _safe_float(call.get("total_tokens"))
            latency_ms = _safe_float(call.get("latency_ms"))

            has_token_usage = prompt_tokens is not None or completion_tokens is not None or total_tokens is not None
            if has_token_usage:
                call_count += 1
                if call.get("success") is False:
                    failed_call_count += 1
                else:
                    successful_call_count += 1

            if prompt_tokens is not None:
                claim_prompt_total += prompt_tokens
                claim_has_usage = True
            if completion_tokens is not None:
                claim_completion_total += completion_tokens
                claim_has_usage = True
            if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
                total_tokens = prompt_tokens + completion_tokens
            if total_tokens is not None:
                claim_total_token_total += total_tokens
                claim_has_usage = True

            if latency_ms is not None:
                latency_total += latency_ms
                latency_count += 1

        if not claim_has_usage:
            continue

        claims_with_usage += 1
        prompt_total += claim_prompt_total
        completion_total += claim_completion_total
        total_token_total += claim_total_token_total

    return {
        "claims_with_usage": claims_with_usage,
        "api_call_count": call_count,
        "successful_api_call_count": successful_call_count,
        "failed_api_call_count": failed_call_count,
        "total_input_tokens": int(prompt_total),
        "total_output_tokens": int(completion_total),
        "total_tokens": int(total_token_total),
        "average_input_tokens": round(prompt_total / claims_with_usage, 4) if claims_with_usage else 0.0,
        "average_output_tokens": round(completion_total / claims_with_usage, 4) if claims_with_usage else 0.0,
        "average_total_tokens": round(total_token_total / claims_with_usage, 4) if claims_with_usage else 0.0,
        "average_input_tokens_per_call": round(prompt_total / call_count, 4) if call_count else 0.0,
        "average_output_tokens_per_call": round(completion_total / call_count, 4) if call_count else 0.0,
        "average_total_tokens_per_call": round(total_token_total / call_count, 4) if call_count else 0.0,
        "average_latency_ms_per_call": round(latency_total / latency_count, 4) if latency_count else 0.0,
    }


def _format_html_cell(val: Any) -> str:
    """Format value for HTML without truncation."""
    if val is None:
        return "-"
    if isinstance(val, dict):
        return json.dumps(val, ensure_ascii=False)
    return str(val)


def _extract_list_item_name(val: Any, section: str) -> str:
    """Extract display name from list item based on section type."""
    if val is None:
        return "-"
    if not isinstance(val, dict):
        return str(val)
    
    if section == "medicines":
        name = val.get("name", "")
        qty = val.get("quantity")
        unit = val.get("unit", "")
        return f"{name} {qty} {unit}".strip() if name else "-"
    elif section == "other_charges":
        cat = val.get("category", "")
        desc = val.get("description", "")
        return f"{cat}: {desc}".strip() if cat else "-"
    elif section == "diagnoses":
        desc = val.get("description", "")
        doc = val.get("doctor_name", "")
        return f"{desc} ({doc})".strip() if desc else "-"
    return json.dumps(val, ensure_ascii=False)


def _generate_html_diff(claim_id: str, predicted: dict[str, Any], ground_truth: dict[str, Any], score: dict[str, Any], medicine_matches: list, other_charges_matches: list) -> list[str]:
    per_field = score.get("per_field", {})
    primary = _get_primary_metric(score)
    field_exact = score.get("field_exact_accuracy", {})
    financial_exact = score.get("financial_exact_accuracy", {})
    relaxed_text = score.get("relaxed_text", {})
    usability = _get_usability_diagnostics(score)
    blocking_fail_reasons = usability.get("blocking_fail_reasons", usability.get("critical_fail_reasons", []))
    advisory_fail_reasons = usability.get("advisory_fail_reasons", [])
    blocking_fail_text = "\n".join(blocking_fail_reasons) if blocking_fail_reasons else "None"
    advisory_fail_text = "\n".join(advisory_fail_reasons) if advisory_fail_reasons else "None"
    html = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'><title>Claim Diff Report</title>",
        "<style>",
        "body{font-family:Consolas,monospace;margin:20px;background:#1e1e1e;color:#d4d4d4}",
        "table{border-collapse:collapse;width:100%;margin:10px 0;table-layout:fixed}",
        "th,td{padding:8px;border:1px solid #444;text-align:left;word-wrap:break-word}",
        "th{background:#2d2d2d}",
        "td:nth-child(1){width:20%}",
        "td:nth-child(2){width:35%}",
        "td:nth-child(3){width:35%}",
        "td:nth-child(4){width:10%}",
        ".g{color:#4ec9b0}.r{color:#f14c4c}.y{color:#dcdcaa}.b{color:#569cd6}",
        ".muted{color:#808080}",
        ".section{margin:20px 0 5px;font-weight:bold;font-size:18px;color:#569cd6}",
        ".field{color:#9cdcfe}",
        ".summary-card{background:#252526;border:1px solid #444;padding:12px;margin:12px 0}",
        "pre{margin:0;white-space:pre-wrap;word-wrap:break-word}",
        "</style></head><body>",
        f"<h2>Claim Diff: {claim_id}</h2>",
        "<div class='summary-card'>",
        f"<div>Primary Metric: <span class='b'>{score.get('primary_metric', PRIMARY_METRIC)}</span></div>",
        f"<div>Strict KIE Micro F1: <span class='b'>{primary.get('f1', 0):.4f}</span></div>",
        f"<div>Precision / Recall: <span class='b'>{primary.get('precision', 0):.4f} / {primary.get('recall', 0):.4f}</span></div>",
        f"<div>Critical Exact Accuracy: <span class='b'>{field_exact.get('accuracy', 0):.4f}</span></div>",
        f"<div>Financial Exact Accuracy: <span class='b'>{financial_exact.get('accuracy', 0):.4f}</span></div>",
        f"<div>Relaxed Text ANLS / CER: <span class='b'>{relaxed_text.get('average_anls', 0):.4f} / {relaxed_text.get('average_cer', 0):.4f}</span></div>",
        f"<div>Edit Burden Score: <span class='b'>{usability.get('edit_burden_score', 0):.4f}</span></div>",
        f"<div>Pass Verify Only: <span class='{'g' if usability.get('pass_verify_only', False) else 'r'}'>{usability.get('pass_verify_only', False)}</span></div>",
        "<div>Blocking Fail Reasons:</div>",
        f"<pre>{blocking_fail_text}</pre>",
        "<div>Advisory Review Reasons:</div>",
        f"<pre>{advisory_fail_text}</pre>",
        "</div>",
    ]

    sections = ["summary", "patient_info", "medical_context", "medicines", "other_charges", "financial_recap"]
    for section in sections:
        pred_s = predicted.get(section, {})
        gt_s = ground_truth.get(section, {})
        html.append(f"<div class='section'>SECTION: {section}</div>")
        html.append("<table><tr><th>Field</th><th>Ground Truth</th><th>Predicted</th><th>Status</th></tr>")

        if isinstance(pred_s, dict):
            for field in sorted(set(gt_s.keys()) | set(pred_s.keys())):
                gv, pv = gt_s.get(field), pred_s.get(field)
                fp = f"{section}.{field}"
                m = per_field.get(fp, {})
                f1, cer = m.get("f1", 0), m.get("cer")
                strict_f1 = m.get("strict_f1")

                if gv is None and pv is None:
                    stat, cls = "✓", "g"
                elif gv is None:
                    stat, cls = "extra", "r"
                elif pv is None:
                    stat, cls = "missing", "r"
                elif cer:
                    stat, cls = f"CER={cer:.2f}", "y"
                elif f1 >= 1:
                    stat, cls = "✓", "g"
                elif f1 >= 0.5:
                    stat, cls = "~", "y"
                else:
                    stat, cls = "✗", "r"

                if isinstance(strict_f1, (int, float)):
                    stat = f"{stat} / strict={strict_f1:.2f}"

                html.append(f"<tr><td class='field'>{fp}</td><td>{_format_html_cell(gv)}</td><td>{_format_html_cell(pv)}</td><td class='{cls}'>{stat}</td></tr>")

        elif isinstance(pred_s, list) and pred_s:
            if section == "medicines":
                med_details = score.get("medicine_details", {})
                matches = med_details.get("matches", [])
                unmatched_pred = med_details.get("unmatched_predicted", [])
                unmatched_gt = med_details.get("unmatched_ground_truth", [])

                idx = 0
                strict_matches = per_field.get("medicines", {}).get("strict_matches", [])
                for m in matches:
                    gt_name = m.get("ground_truth", {}).get("name", "?")
                    pred_name = m.get("predicted", {}).get("name", "?")
                    sim = m.get("similarity", 0.0)
                    strict_pass = any(sm.get("predicted") == m.get("predicted") and sm.get("ground_truth") == m.get("ground_truth") for sm in strict_matches)
                    if sim >= 0.8:
                        stat, cls = "✓", "g"
                    elif sim >= 0.5:
                        stat, cls = "~", "y"
                    else:
                        stat, cls = "✗", "r"
                    html.append(f"<tr><td>[{idx}]</td><td>{_format_html_cell(gt_name)}</td><td>{_format_html_cell(pred_name)}</td><td class='{cls}'>{stat} (sim={sim:.2f}, strict={'pass' if strict_pass else 'fail'})</td></tr>")
                    idx += 1
                for m in unmatched_pred:
                    html.append(f"<tr><td>[{idx}]</td><td>-</td><td>{_format_html_cell(m.get('name', '?'))}</td><td class='r'>extra</td></tr>")
                    idx += 1
                for m in unmatched_gt:
                    html.append(f"<tr><td>[{idx}]</td><td>{_format_html_cell(m.get('name', '?'))}</td><td>-</td><td class='r'>missing</td></tr>")
                    idx += 1

            elif section == "other_charges":
                oc_details = score.get("other_charges_details", {})
                matches = oc_details.get("matches", [])
                unmatched_pred = oc_details.get("unmatched_predicted", [])
                unmatched_gt = oc_details.get("unmatched_ground_truth", [])
                strict_matches = per_field.get("other_charges", {}).get("strict_matches", [])

                idx = 0
                for m in matches:
                    gt_cat = m.get("ground_truth", {}).get("category", "?")
                    pred_cat = m.get("predicted", {}).get("category", "?")
                    gt_canon = m.get("ground_truth_category_canonical")
                    pred_canon = m.get("predicted_category_canonical")
                    sim = m.get("similarity", 0.0)
                    strict_pass = any(sm.get("predicted") == m.get("predicted") and sm.get("ground_truth") == m.get("ground_truth") for sm in strict_matches)
                    if sim >= 0.8:
                        stat, cls = "✓", "g"
                    elif sim >= 0.5:
                        stat, cls = "~", "y"
                    else:
                        stat, cls = "✗", "r"
                    gt_label = f"{gt_cat} (canon: {gt_canon})" if gt_canon else gt_cat
                    pred_label = f"{pred_cat} (canon: {pred_canon})" if pred_canon else pred_cat
                    html.append(f"<tr><td>[{idx}]</td><td>{_format_html_cell(gt_label)}</td><td>{_format_html_cell(pred_label)}</td><td class='{cls}'>{stat} (sim={sim:.2f}, strict={'pass' if strict_pass else 'fail'})</td></tr>")
                    idx += 1
                for m in unmatched_pred:
                    html.append(f"<tr><td>[{idx}]</td><td>-</td><td>{_format_html_cell(m.get('category', '?'))}</td><td class='r'>extra</td></tr>")
                    idx += 1
                for m in unmatched_gt:
                    html.append(f"<tr><td>[{idx}]</td><td>{_format_html_cell(m.get('category', '?'))}</td><td>-</td><td class='r'>missing</td></tr>")
                    idx += 1

            else:
                for i, (g, p) in enumerate(zip_longest(pred_s, gt_s)):
                    if g and p:
                        stat, cls = "✓", "g"
                    elif g:
                        stat, cls = "missing", "r"
                    else:
                        stat, cls = "extra", "r"
                    html.append(f"<tr><td>[{i}]</td><td>{_format_html_cell(g)}</td><td>{_format_html_cell(p)}</td><td class='{cls}'>{stat}</td></tr>")

        html.append("</table>")

    html.extend(["</body></html>"])
    return html


def _emit_log(
    message: str,
    *,
    configured_level: str,
    level: str = "info",
    claim_id: str | None = None,
    claim_index: int | None = None,
    total_claims: int | None = None,
    force: bool = False,
) -> None:
    configured_priority = LOG_LEVEL_PRIORITY.get(configured_level, LOG_LEVEL_PRIORITY["info"])
    message_priority = LOG_LEVEL_PRIORITY.get(level, LOG_LEVEL_PRIORITY["info"])
    if not force and configured_priority < message_priority:
        return

    prefix_parts: list[str] = []
    if claim_index is not None and total_claims is not None:
        prefix_parts.append(f"[{claim_index}/{total_claims}]")
    if claim_id:
        prefix_parts.append(f"[{claim_id}]")
    prefix_parts.append(f"[{level.upper()}]")

    print(" ".join(prefix_parts) + f" {message}")


class ScanAttemptError(RuntimeError):
    def __init__(self, message: str, run_record_path: str | None = None):
        super().__init__(message)
        self.run_record_path = run_record_path


def _safe_claim_email(claim_id: str) -> str:
    normalized = "".join(ch.lower() if ch.isalnum() else "-" for ch in claim_id).strip("-")
    return f"eval-{(normalized or 'claim')}@example.com"


def _short_fingerprint(fingerprint: str | None) -> str:
    if not fingerprint:
        return "-"
    if len(fingerprint) <= 12:
        return fingerprint
    return f"{fingerprint[:12]}..."


def export_scan_log(scan_log_id: int | None) -> dict[str, Any] | None:
    if scan_log_id is None:
        return None
    session_factory = get_session_factory()
    with session_factory() as db:
        repository = ScanLogRepository(db)
        scan_log = repository.get_by_id_for_eval_export(scan_log_id)
        if scan_log is None:
            return None
        return {
            "scan_log_id": scan_log.id,
            "status": scan_log.status,
            "ai_execution_mode": scan_log.ai_execution_mode,
            "primary_model_provider": scan_log.primary_model_provider,
            "primary_model_name": scan_log.primary_model_name,
            "validation_payload": scan_log.validation_payload,
            "extraction_payload": scan_log.extraction_payload,
            "extraction_meta_payload": scan_log.extraction_meta_payload,
            "error_payload": scan_log.error_payload,
            "created_at": scan_log.created_at.isoformat() if scan_log.created_at else None,
            "updated_at": scan_log.updated_at.isoformat() if scan_log.updated_at else None,
            "ai_model_runs": [
                {
                    "producer_id": run.producer_id,
                    "provider": run.provider,
                    "model_name": run.model_name,
                    "producer_rank": run.producer_rank,
                    "run_index": run.run_index,
                    "temperature": run.temperature,
                    "success": run.success,
                    "error_message": run.error_message,
                    "payload": run.payload,
                    "created_at": run.created_at.isoformat() if run.created_at else None,
                }
                for run in scan_log.ai_runs
            ],
            "ai_api_calls": [
                {
                    "provider": call.provider,
                    "model_name": call.model_name,
                    "call_kind": call.call_kind,
                    "producer_id": call.producer_id,
                    "producer_rank": call.producer_rank,
                    "run_index": call.run_index,
                    "temperature": call.temperature,
                    "success": call.success,
                    "error_message": call.error_message,
                    "prompt_tokens": call.prompt_tokens,
                    "completion_tokens": call.completion_tokens,
                    "total_tokens": call.total_tokens,
                    "latency_ms": call.latency_ms,
                    "usage_payload": call.usage_payload,
                    "prompt_file_path": call.prompt_file_path,
                    "response_validation_error": call.response_validation_error,
                    "created_at": call.created_at.isoformat() if call.created_at else None,
                }
                for call in scan_log.ai_api_call_logs
            ],
        }


def _get_latest_scan_log_for_user(user_email: str) -> dict[str, Any] | None:
    session_factory = get_session_factory()
    with session_factory() as db:
        user = UserRepository(db).get_by_email(user_email)
        if user is None:
            return None
        scan_log = ScanLogRepository(db).get_latest_for_user_for_eval_export(user.id)
        if scan_log is None:
            return None
        return export_scan_log(scan_log.id)


def _compute_fingerprint_inputs(
    *,
    claim_id: str,
    claim_folder: Path,
    gt_data: dict[str, Any],
    image_paths: list[Path],
) -> tuple[str, dict[str, Any]]:
    image_entries: list[tuple[str, str]] = []
    for path in image_paths:
        relative = str(path.relative_to(claim_folder)).replace("\\", "/")
        image_entries.append((relative, sha256_file(path)))

    patient_candidates = [
        candidate
        for candidate in gt_data.get("patient_name_candidates", [])
        if isinstance(candidate, str)
    ]
    fingerprint_payload = fingerprint_input_payload(
        claim_id=claim_id,
        image_entries=image_entries,
        patient_name_candidates=patient_candidates,
    )
    fingerprint = build_claim_fingerprint(
        ClaimFingerprintInput(
            claim_id=claim_id,
            image_entries=image_entries,
            patient_name_candidates=patient_candidates,
        )
    )
    return fingerprint, fingerprint_payload


def _load_cached_predicted(
    *,
    claim_id: str,
    results_dir: Path,
    fingerprint: str,
    allow_stale_cache: bool,
) -> tuple[dict[str, Any] | None, str | None, dict[str, Any] | None, bool]:
    latest = load_latest_snapshot(results_dir, claim_id)
    if latest is None:
        return None, None, None, False

    cached_fingerprint = latest.get("input_fingerprint")
    is_stale = cached_fingerprint != fingerprint
    if is_stale and not allow_stale_cache:
        return None, None, latest, True

    predicted = latest.get("predicted")
    if not isinstance(predicted, dict):
        return None, None, latest, is_stale

    run_path = resolve_run_record_path(results_dir, claim_id, latest)
    run_record = read_json(run_path) if run_path is not None else None
    return predicted, (str(run_path) if run_path else None), run_record, is_stale


async def _scan_and_persist(
    *,
    claim_id: str,
    claim_folder: Path,
    image_paths: list[Path],
    patient_candidates: list[str],
    fingerprint: str,
    fingerprint_payload: dict[str, Any],
    results_dir: Path,
    logger: Callable[[str, str], None] | None = None,
) -> tuple[dict[str, Any], str, dict[str, Any], float]:
    primary_name = patient_candidates[0] if patient_candidates else f"Eval User {claim_id}"
    user_email = _safe_claim_email(claim_id)

    files_for_request = []
    for path in image_paths:
        with open(path, "rb") as file_handle:
            files_for_request.append((str(path.name), file_handle.read(), guess_mime_type(path)))

    files = [("files", (name, content, mime_type)) for name, content, mime_type in files_for_request]

    if logger is not None:
        logger(
            "debug",
            f"scan start: images={len(files_for_request)}, candidates={len(patient_candidates)}, fingerprint={_short_fingerprint(fingerprint)}",
        )

    start = time.time()
    with TestClient(create_app()) as client:
        token = register_test_user(client, user_email, primary_name)
        ensure_family_candidates(client, token, patient_candidates, primary_name)
        response = client.post(
            "/api/ai/scan-documents",
            headers={"Authorization": f"Bearer {token}"},
            files=files,
        )
    elapsed = time.time() - start

    if logger is not None:
        logger("debug", f"scan finished: status_code={response.status_code}, elapsed={elapsed:.2f}s")

    if response.status_code != 200:
        scan_log_export = _get_latest_scan_log_for_user(user_email)
        run_record = build_run_record(
            claim_id=claim_id,
            fingerprint=fingerprint,
            fingerprint_input=fingerprint_payload,
            scan_log_export=scan_log_export,
            predicted={},
            elapsed_seconds=elapsed,
            status="failed",
            source="fresh_scan",
            error={
                "http_status": response.status_code,
                "response_text": response.text,
            },
        )
        run_path = append_run_record(results_dir, claim_id, run_record)
        if logger is not None:
            logger("debug", f"failed scan run record saved: {run_path}")
        raise ScanAttemptError(f"HTTP {response.status_code}: {response.text}", str(run_path))

    predicted = response.json()
    if not isinstance(predicted, dict):
        run_record = build_run_record(
            claim_id=claim_id,
            fingerprint=fingerprint,
            fingerprint_input=fingerprint_payload,
            scan_log_export=_get_latest_scan_log_for_user(user_email),
            predicted={},
            elapsed_seconds=elapsed,
            status="failed",
            source="fresh_scan",
            error={"message": "Invalid scan response payload type."},
        )
        run_path = append_run_record(results_dir, claim_id, run_record)
        if logger is not None:
            logger("debug", f"invalid payload run record saved: {run_path}")
        raise ScanAttemptError("Invalid scan response payload type.", str(run_path))

    scan_log_id = predicted.get("scan_log_id")
    scan_log_export = export_scan_log(int(scan_log_id)) if isinstance(scan_log_id, int) else _get_latest_scan_log_for_user(user_email)

    run_record = build_run_record(
        claim_id=claim_id,
        fingerprint=fingerprint,
        fingerprint_input=fingerprint_payload,
        scan_log_export=scan_log_export,
        predicted=predicted,
        elapsed_seconds=elapsed,
        status="succeeded",
        source="fresh_scan",
    )
    run_path = append_run_record(results_dir, claim_id, run_record)
    if logger is not None:
        logger("debug", f"fresh scan run record saved: {run_path}")

    latest_snapshot = build_latest_snapshot(
        claim_id=claim_id,
        fingerprint=fingerprint,
        fingerprint_input=fingerprint_payload,
        run_record_path=run_path,
        predicted=predicted,
        scan_log_id=int(scan_log_id) if isinstance(scan_log_id, int) else None,
        source="fresh_scan",
    )
    save_latest_snapshot(results_dir, claim_id, latest_snapshot)
    if logger is not None:
        logger(
            "debug",
            f"latest snapshot updated: {results_dir / claim_id / 'latest.json'} (scan_log_id={latest_snapshot.get('scan_log_id')})",
        )

    return predicted, str(run_path), run_record, elapsed


async def run_single_evaluation(
    folder: Path,
    *,
    results_dir: Path | None = None,
    mode: EvalMode = "auto",
    refresh: bool = False,
    allow_stale_cache: bool = False,
    logger: Callable[[str, str], None] | None = None,
) -> dict[str, Any]:
    if results_dir is None:
        results_dir = folder.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    gt_data, image_paths = load_ground_truth(folder)
    claim_id = folder.name
    patient_candidates = [
        candidate
        for candidate in gt_data.get("patient_name_candidates", [])
        if isinstance(candidate, str)
    ]

    fingerprint, fingerprint_payload = _compute_fingerprint_inputs(
        claim_id=claim_id,
        claim_folder=folder,
        gt_data=gt_data,
        image_paths=image_paths,
    )
    if logger is not None:
        logger(
            "debug",
            f"fingerprint computed: {_short_fingerprint(fingerprint)} (images={len(image_paths)}, candidates={len(patient_candidates)})",
        )

    predicted: dict[str, Any] | None = None
    run_record_path: str | None = None
    run_record: dict[str, Any] | None = None
    scan_source = "fresh_scan"
    elapsed_seconds: float = 0.0

    cache_allowed = mode in {"auto", "eval-only"}
    if cache_allowed and not refresh:
        cached_predicted, cached_run_path, cached_run_record, stale = _load_cached_predicted(
            claim_id=claim_id,
            results_dir=results_dir,
            fingerprint=fingerprint,
            allow_stale_cache=allow_stale_cache,
        )
        if cached_predicted is not None:
            if logger is not None:
                logger(
                    "info",
                    f"cache hit: using latest snapshot ({cached_run_path or 'no-run-path'})",
                )
            predicted = cached_predicted
            run_record_path = cached_run_path
            run_record = cached_run_record
            scan_source = "cache"
            elapsed_seconds = 0.0
        elif mode == "eval-only":
            reason = "stale or invalid cache" if stale else "cache not found"
            if logger is not None:
                logger("info", f"cache miss in eval-only mode: {reason}")
            return {
                "claim_id": claim_id,
                "error": f"eval-only mode: {reason} for claim '{claim_id}'.",
                "elapsed_seconds": 0.0,
                "scan_source": "cache",
                "run_record_path": cached_run_path,
            }
        else:
            if logger is not None:
                cache_reason = "stale fingerprint" if stale else "missing cache"
                logger("info", f"cache miss: {cache_reason}, running fresh scan")
    elif refresh and logger is not None:
        logger("info", "refresh enabled: skipping cache and running fresh scan")
    elif mode == "scan-only" and logger is not None:
        logger("debug", "scan-only mode: fresh scan will run and scoring will be skipped")

    if predicted is None:
        try:
            predicted, run_record_path, run_record, elapsed_seconds = await _scan_and_persist(
                claim_id=claim_id,
                claim_folder=folder,
                image_paths=image_paths,
                patient_candidates=patient_candidates,
                fingerprint=fingerprint,
                fingerprint_payload=fingerprint_payload,
                results_dir=results_dir,
                logger=logger,
            )
            scan_source = "fresh_scan"
        except Exception as exc:  # noqa: BLE001
            run_path = exc.run_record_path if isinstance(exc, ScanAttemptError) else None
            if logger is not None:
                logger("info", f"scan failed: {str(exc) or exc.__class__.__name__}")
            return {
                "claim_id": claim_id,
                "error": str(exc) or exc.__class__.__name__,
                "elapsed_seconds": round(elapsed_seconds, 2),
                "scan_source": "fresh_scan",
                "run_record_path": run_path,
            }

    if mode == "scan-only":
        if logger is not None:
            logger("info", "scan-only complete: run saved, evaluation skipped")
        return {
            "claim_id": claim_id,
            "elapsed_seconds": round(elapsed_seconds, 2),
            "scan_source": scan_source,
            "run_record_path": run_record_path,
            "scan_log_id": predicted.get("scan_log_id") if isinstance(predicted, dict) else None,
            "status": "scan_saved",
            "run_record": run_record,
        }

    evaluator = MedicalClaimEvaluator(DEFAULT_THRESHOLDS)
    score_result = evaluator.evaluate(predicted, gt_data["expected"])
    if logger is not None:
        primary = _get_primary_metric(score_result)
        logger("debug", f"evaluation complete: strict_kie_micro_f1={primary.get('f1', 0.0):.4f}")
    return {
        "claim_id": claim_id,
        "elapsed_seconds": round(elapsed_seconds, 2),
        "predicted": predicted,
        "score": score_result,
        "scan_source": scan_source,
        "run_record_path": run_record_path,
        "scan_log_id": predicted.get("scan_log_id") if isinstance(predicted, dict) else None,
        "run_record": run_record,
    }


def build_aggregate_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [result for result in results if "error" not in result and "score" in result]
    claims_from_cache = sum(1 for result in results if result.get("scan_source") == "cache")
    claims_from_fresh_scan = sum(1 for result in results if result.get("scan_source") == "fresh_scan")

    if not successful:
        return {
            "schema_version": EVALUATOR_SCHEMA_VERSION,
            "primary_metric": PRIMARY_METRIC,
            "strict_kie_micro": _zero_metric(),
            "strict_kie_macro_f1": 0.0,
            "strict_kie_section_macro_f1": 0.0,
            "successful_claims": 0,
            "failed_claims": len(results),
            "claims_from_cache": claims_from_cache,
            "claims_from_fresh_scan": claims_from_fresh_scan,
            "per_section_micro": {},
            "per_section_macro_f1": {},
            "per_field_macro_f1": {},
            "per_field_micro": {},
            "field_exact_accuracy": _aggregate_accuracy([], "field_exact_accuracy"),
            "financial_exact_accuracy": _aggregate_accuracy([], "financial_exact_accuracy"),
            "relaxed_text": _aggregate_relaxed_text([]),
            "token_usage": _aggregate_token_usage(results),
            "diagnostics": {
                "average_edit_burden_score": 0.0,
                "pass_verify_only_rate": 0.0,
            },
        }

    scores = [result["score"] for result in successful]
    primary_metrics = [_get_primary_metric(score) for score in scores]
    primary_micro = _aggregate_metric_counts(primary_metrics)
    strict_kie_macro_f1 = _mean([float(metric.get("f1", 0.0)) for metric in _applicable_metrics(primary_metrics)])
    section_macro_values = [
        float(score.get("strict_kie_section_macro_f1", 0.0))
        for score in scores
        if isinstance(score.get("strict_kie_section_macro_f1", 0.0), (int, float))
    ]

    section_names = {section for score in scores for section in score.get("per_section", {}).keys()}
    per_section_micro: dict[str, dict[str, Any]] = {}
    per_section_macro_f1: dict[str, float] = {}
    for section in sorted(section_names):
        metrics = [
            score.get("per_section", {}).get(section, _zero_metric())
            for score in scores
            if isinstance(score.get("per_section", {}).get(section, _zero_metric()), dict)
        ]
        per_section_micro[section] = _aggregate_metric_counts(metrics)
        per_section_macro_f1[section] = _mean([float(metric.get("f1", 0.0)) for metric in _applicable_metrics(metrics)])

    field_names = {field for score in scores for field in score.get("per_field", {}).keys()}
    per_field_micro: dict[str, dict[str, Any]] = {}
    per_field_macro_f1: dict[str, float] = {}
    for field in sorted(field_names):
        metrics = [
            score.get("per_field", {}).get(field, _zero_metric())
            for score in scores
            if isinstance(score.get("per_field", {}).get(field, _zero_metric()), dict)
        ]
        per_field_micro[field] = _aggregate_metric_counts(metrics)
        per_field_macro_f1[field] = _mean([float(metric.get("f1", 0.0)) for metric in _applicable_metrics(metrics)])

    usability_diagnostics = [_get_usability_diagnostics(score) for score in scores]
    edit_burden_scores = [
        float(diagnostic.get("edit_burden_score", 0.0))
        for diagnostic in usability_diagnostics
        if isinstance(diagnostic.get("edit_burden_score", 0.0), (int, float))
    ]
    pass_verify_count = sum(1 for diagnostic in usability_diagnostics if diagnostic.get("pass_verify_only") is True)

    return {
        "schema_version": EVALUATOR_SCHEMA_VERSION,
        "primary_metric": PRIMARY_METRIC,
        "strict_kie_micro": primary_micro,
        "strict_kie_macro_f1": strict_kie_macro_f1,
        "strict_kie_section_macro_f1": _mean(section_macro_values),
        "successful_claims": len(successful),
        "failed_claims": len(results) - len(successful),
        "claims_from_cache": claims_from_cache,
        "claims_from_fresh_scan": claims_from_fresh_scan,
        "per_section_micro": per_section_micro,
        "per_section_macro_f1": per_section_macro_f1,
        "per_field_micro": per_field_micro,
        "per_field_macro_f1": per_field_macro_f1,
        "field_exact_accuracy": _aggregate_accuracy(scores, "field_exact_accuracy"),
        "financial_exact_accuracy": _aggregate_accuracy(scores, "financial_exact_accuracy"),
        "relaxed_text": _aggregate_relaxed_text(scores),
        "token_usage": _aggregate_token_usage(successful),
        "diagnostics": {
            "average_edit_burden_score": _mean(edit_burden_scores),
            "pass_verify_only_rate": round(pass_verify_count / len(successful), 4) if successful else 0.0,
        },
    }


def print_results(results: list[dict[str, Any]], mode: EvalMode) -> None:
    print("\n" + "=" * 132)
    print(f"{'CLAIM':<25} {'KIE_F1':>7} {'PREC':>7} {'REC':>7} {'ANLS':>7} {'CER':>7} {'SRC':>10} {'ELAPSED':>8}  STATUS")
    print("-" * 132)

    f1_values: list[float] = []
    count = 0
    for result in results:
        source = str(result.get("scan_source", "-"))
        if "error" in result:
            print(
                f"{result['claim_id']:<25} {'ERR':>7} {'-':>7} {'-':>7} {'-':>7} {'-':>7} {source:>10} "
                f"{result.get('elapsed_seconds', 0):>8}s  {str(result['error'])[:40]}"
            )
            continue

        if mode == "scan-only" or "score" not in result:
            print(
                f"{result['claim_id']:<25} {'-':>7} {'-':>7} {'-':>7} {'-':>7} {'-':>7} {source:>10} "
                f"{result.get('elapsed_seconds', 0):>8}s  scan_saved"
            )
            continue

        score = result["score"]
        primary = _get_primary_metric(score)
        relaxed = score.get("relaxed_text", {})
        claim_f1 = float(primary.get("f1", 0.0))
        claim_prec = float(primary.get("precision", 0.0))
        claim_rec = float(primary.get("recall", 0.0))
        anls = relaxed.get("average_anls") if isinstance(relaxed, dict) else None
        cer = relaxed.get("average_cer") if isinstance(relaxed, dict) else None
        anls_str = f"{anls:.4f}" if isinstance(anls, (int, float)) else "-"
        cer_str = f"{cer:.4f}" if isinstance(cer, (int, float)) else "-"

        f1_values.append(claim_f1)
        count += 1
        print(
            f"{result['claim_id']:<25} {claim_f1:>7.4f} {claim_prec:>7.4f} {claim_rec:>7.4f} "
            f"{anls_str:>7} {cer_str:>7} {source:>10} "
            f"{result.get('elapsed_seconds', 0):>8}s  evaluated"
        )

    print("-" * 132)
    if count > 0:
        print(f"\nStrict KIE Macro F1={_mean(f1_values):.4f}  ({count} claims evaluated)")

    summary = build_aggregate_summary(results)
    print(
        "Scan Sources: "
        f"cache={summary.get('claims_from_cache', 0)}, "
        f"fresh_scan={summary.get('claims_from_fresh_scan', 0)}"
    )

    print_evaluation_summary(summary)


def print_evaluation_summary(summary: dict[str, Any]) -> None:
    print(f"\n{C_BOLD}EVALUATION SUMMARY{C_RESET}")
    print("=" * 80)
    primary = summary.get("strict_kie_micro", {})
    field_exact = summary.get("field_exact_accuracy", {})
    financial_exact = summary.get("financial_exact_accuracy", {})
    relaxed = summary.get("relaxed_text", {})
    token_usage = summary.get("token_usage", {})
    diagnostics = summary.get("diagnostics", {})
    print(f"Schema: {summary.get('schema_version', EVALUATOR_SCHEMA_VERSION)}")
    print(
        f"Primary: {C_BOLD}{summary.get('primary_metric', PRIMARY_METRIC)}={primary.get('f1', 0.0):.4f}{C_RESET}  "
        f"{C_BOLD}P={primary.get('precision', 0.0):.4f}{C_RESET}  "
        f"{C_BOLD}R={primary.get('recall', 0.0):.4f}{C_RESET}"
    )
    print(
        f"Claim Macro F1={summary.get('strict_kie_macro_f1', 0.0):.4f}  "
        f"Section Macro F1={summary.get('strict_kie_section_macro_f1', 0.0):.4f}"
    )
    print(
        f"Critical Exact={field_exact.get('accuracy', 0.0):.4f} "
        f"({field_exact.get('correct', 0)}/{field_exact.get('total', 0)})  "
        f"Financial Exact={financial_exact.get('accuracy', 0.0):.4f} "
        f"({financial_exact.get('correct', 0)}/{financial_exact.get('total', 0)})"
    )
    print(
        f"Relaxed Text: ANLS={relaxed.get('average_anls', 0.0):.4f}  "
        f"CER={relaxed.get('average_cer', 0.0):.4f}  "
        f"n={relaxed.get('count', 0)}"
    )
    print(
        "Token Usage: "
        f"avg_input={token_usage.get('average_input_tokens', 0.0):.2f}  "
        f"avg_output={token_usage.get('average_output_tokens', 0.0):.2f}  "
        f"avg_total={token_usage.get('average_total_tokens', 0.0):.2f}  "
        f"claims_with_usage={token_usage.get('claims_with_usage', 0)}  "
        f"api_calls={token_usage.get('api_call_count', 0)}"
    )
    print(
        f"Usability Diagnostics: avg_edit_burden={diagnostics.get('average_edit_burden_score', 0.0):.4f}  "
        f"pass_verify_only_rate={diagnostics.get('pass_verify_only_rate', 0.0):.4f}"
    )

    print(f"\n{C_BOLD}SECTION BREAKDOWN:{C_RESET}")
    print(f"{'SECTION':<20} {'MICRO_F1':>8} {'MACRO_F1':>8} {'PREC':>8} {'REC':>8}")
    print("-" * 60)
    section_micro = summary.get("per_section_micro", {})
    section_macro = summary.get("per_section_macro_f1", {})
    for section in sorted(section_micro.keys()):
        metric = section_micro.get(section, {})
        print(
            f"{section:<20} {metric.get('f1', 0.0):>8.4f} {section_macro.get(section, 0.0):>8.4f} "
            f"{metric.get('precision', 0.0):>8.4f} {metric.get('recall', 0.0):>8.4f}"
        )

    field_f1s = summary.get("per_field_macro_f1", {})
    if field_f1s:
        sorted_fields = sorted(field_f1s.items(), key=lambda x: x[1])
        bottom_fields = sorted_fields[:5]
        print(f"\n{C_BOLD}FIELDS NEEDING ATTENTION (lowest strict KIE F1):{C_RESET}")
        for i, (field, f1) in enumerate(bottom_fields, 1):
            color = C_RED if f1 < 0.5 else C_YELLOW if f1 < 0.8 else C_RESET
            print(f"  {i}. {field:<40} F1={color}{f1:.4f}{C_RESET}")


def print_claim_detail(result: dict[str, Any], gt_data: dict[str, Any]) -> None:
    score = result.get("score", {})
    per_section = score.get("per_section", {})
    per_field = score.get("per_field", {})

    print(f"\n{C_BOLD}CLAIM SUMMARY:{C_RESET}")
    primary = _get_primary_metric(score)
    field_exact = score.get("field_exact_accuracy", {})
    financial_exact = score.get("financial_exact_accuracy", {})
    relaxed = score.get("relaxed_text", {})
    usability = _get_usability_diagnostics(score)
    print(f"  schema_version: {score.get('schema_version', EVALUATOR_SCHEMA_VERSION)}")
    print(f"  primary_metric: {score.get('primary_metric', PRIMARY_METRIC)}")
    print(
        f"  strict_kie_micro: f1={primary.get('f1', 0.0):.4f} "
        f"precision={primary.get('precision', 0.0):.4f} recall={primary.get('recall', 0.0):.4f}"
    )
    print(f"  strict_kie_section_macro_f1: {score.get('strict_kie_section_macro_f1', 0.0):.4f}")
    print(
        f"  critical_exact_accuracy: {field_exact.get('accuracy', 0.0):.4f} "
        f"({field_exact.get('correct', 0)}/{field_exact.get('total', 0)})"
    )
    print(
        f"  financial_exact_accuracy: {financial_exact.get('accuracy', 0.0):.4f} "
        f"({financial_exact.get('correct', 0)}/{financial_exact.get('total', 0)})"
    )
    print(
        f"  relaxed_text: ANLS={relaxed.get('average_anls', 0.0):.4f} "
        f"CER={relaxed.get('average_cer', 0.0):.4f} n={relaxed.get('count', 0)}"
    )
    print(f"  edit_burden_score: {usability.get('edit_burden_score', 0.0):.4f}")
    print(f"  pass_verify_only: {usability.get('pass_verify_only', False)}")
    blocking_fail_reasons = usability.get("blocking_fail_reasons", usability.get("critical_fail_reasons", []))
    advisory_fail_reasons = usability.get("advisory_fail_reasons", [])
    if blocking_fail_reasons:
        print("  blocking_fail_reasons:")
        for reason in blocking_fail_reasons:
            print(f"    - {reason}")
    if advisory_fail_reasons:
        print("  advisory_fail_reasons:")
        for reason in advisory_fail_reasons:
            print(f"    - {reason}")

    print(f"\n{C_BOLD}SECTION BREAKDOWN:{C_RESET}")
    print(f"{'SECTION':<20} {'F1':>8} {'PREC':>8} {'REC':>8} {'TP':>8} {'FP':>6} {'FN':>6}")
    print("-" * 80)
    for section in sorted(per_section.keys()):
        s = per_section[section]
        print(
            f"{section:<20} {s.get('f1', 0.0):>8.4f} {s.get('precision', 0.0):>8.4f} "
            f"{s.get('recall', 0.0):>8.4f} {s.get('tp', 0):>8} {s.get('fp', 0):>6} {s.get('fn', 0):>6}"
        )

    print(f"\n{C_BOLD}PER-FIELD BREAKDOWN:{C_RESET}")
    for field, metrics in sorted(per_field.items()):
        f1 = metrics.get("f1", 0.0)
        precision = metrics.get("precision", 0.0)
        recall = metrics.get("recall", 0.0)
        cer = metrics.get("cer")
        anls = metrics.get("anls")
        if cer is not None:
            anls_extra = f"  ANLS={anls:.4f}" if isinstance(anls, (int, float)) else ""
            print(f"  {field:<40} F1={f1:.4f} P={precision:.4f} R={recall:.4f}{anls_extra}  CER={cer:.4f}")
        else:
            print(f"  {field:<40} F1={f1:.4f} P={precision:.4f} R={recall:.4f}")

    med_details = score.get("medicine_details", {})
    if med_details:
        matches = med_details.get("matches", [])
        unmatched_pred = med_details.get("unmatched_predicted", [])
        unmatched_gt = med_details.get("unmatched_ground_truth", [])
        avg_similarity = per_field.get("medicines", {}).get("avg_similarity")
        strict_matches = per_field.get("medicines", {}).get("strict_matches", [])
        print(f"\n{C_BOLD}MEDICINE MATCHES:{C_RESET}")
        if avg_similarity is not None:
            print(f"  Avg similarity: {avg_similarity:.4f}")
        for m in matches:
            gt_name = m.get("ground_truth", {}).get("name", "?")
            pred_name = m.get("predicted", {}).get("name", "?")
            sim = m.get("similarity", 0.0)
            status = f"{C_GREEN}✓{C_RESET}" if sim >= 0.8 else f"{C_YELLOW}~{C_RESET}"
            strict_pass = any(sm.get("predicted") == m.get("predicted") and sm.get("ground_truth") == m.get("ground_truth") for sm in strict_matches)
            strict_tag = " strict=pass" if strict_pass else " strict=fail"
            print(f"  {status} GT: {gt_name} → PRED: {pred_name} (sim={sim:.2f},{strict_tag})")
        for m in unmatched_pred:
            print(f"  {C_RED}[EXTRA]{C_RESET} PRED: {m.get('name', '?')}")
        for m in unmatched_gt:
            print(f"  {C_RED}[MISSING]{C_RESET} GT: {m.get('name', '?')}")

    diag_metrics = per_field.get("medical_context.diagnoses", {})
    diag_matches = diag_metrics.get("matches", []) if isinstance(diag_metrics, dict) else []
    if diag_matches or diag_metrics.get("avg_similarity") is not None:
        print(f"\n{C_BOLD}DIAGNOSIS MATCHES:{C_RESET}")
        if diag_metrics.get("avg_similarity") is not None:
            print(f"  Avg similarity: {diag_metrics.get('avg_similarity', 0.0):.4f}")
        for m in diag_matches:
            gt_desc = m.get("ground_truth", {}).get("description", "?")
            pred_desc = m.get("predicted", {}).get("description", "?")
            sim = m.get("similarity", 0.0)
            doc_sim = m.get("doctor_similarity")
            status = f"{C_GREEN}✓{C_RESET}" if sim >= 0.8 else f"{C_YELLOW}~{C_RESET}"
            doc_extra = f", doc={doc_sim:.2f}" if isinstance(doc_sim, (int, float)) else ""
            print(f"  {status} GT: {gt_desc} → PRED: {pred_desc} (sim={sim:.2f}{doc_extra})")

    oc_details = score.get("other_charges_details", {})
    oc_matches = oc_details.get("matches", []) if isinstance(oc_details, dict) else []
    if oc_matches or oc_details.get("avg_similarity") is not None:
        print(f"\n{C_BOLD}OTHER CHARGE MATCHES:{C_RESET}")
        if oc_details.get("avg_similarity") is not None:
            print(f"  Avg similarity: {oc_details.get('avg_similarity', 0.0):.4f}")
        for m in oc_matches:
            gt_cat = m.get("ground_truth", {}).get("category", "?")
            pred_cat = m.get("predicted", {}).get("category", "?")
            gt_canon = m.get("ground_truth_category_canonical")
            pred_canon = m.get("predicted_category_canonical")
            sim = m.get("similarity", 0.0)
            desc_sim = m.get("description_similarity")
            status = f"{C_GREEN}✓{C_RESET}" if sim >= 0.8 else f"{C_YELLOW}~{C_RESET}"
            desc_extra = f", desc={desc_sim:.2f}" if isinstance(desc_sim, (int, float)) else ""
            canon_extra = f", canon={gt_canon}->{pred_canon}" if gt_canon or pred_canon else ""
            print(f"  {status} GT: {gt_cat} → PRED: {pred_cat} (sim={sim:.2f}{desc_extra}{canon_extra})")


def print_claim_diff(predicted: dict[str, Any], ground_truth: dict[str, Any], score: dict[str, Any], html_file: Path | None = None, claim_id: str | None = None) -> None:
    per_field = score.get("per_field", {})
    medicine_matches = score.get("medicine_details", {}).get("matches", [])
    other_charges_matches = score.get("other_charges_details", {}).get("matches", [])

    sections = ["summary", "patient_info", "medical_context", "medicines", "other_charges", "financial_recap"]

    if html_file:
        html_lines = _generate_html_diff(claim_id or "unknown", predicted, ground_truth, score, medicine_matches, other_charges_matches)
        html_file.write_text("\n".join(html_lines), encoding="utf-8")
        print(f"{C_CYAN}HTML diff written to: {html_file}{C_RESET}")

    for section in sections:
        pred_section = predicted.get(section, {})
        gt_section = ground_truth.get(section, {})

        print(f"\n{C_BOLD}SECTION: {section}{C_RESET}")
        print(f"{'FIELD':<35} {'GT':<15} {'PRED':<15} STATUS")
        print("-" * 70)

        if isinstance(pred_section, dict):
            for field in sorted(set(gt_section.keys()) | set(pred_section.keys())):
                gt_val = gt_section.get(field)
                pred_val = pred_section.get(field)
                field_path = f"{section}.{field}"

                metrics = per_field.get(field_path, {})
                f1 = metrics.get("f1", 0.0)
                cer = metrics.get("cer")

                if gt_val is None and pred_val is None:
                    status = f"{C_GREEN}✓{C_RESET}"
                elif gt_val is None:
                    status = f"{C_RED}extra{C_RESET}"
                elif pred_val is None:
                    status = f"{C_RED}missing{C_RESET}"
                elif cer is not None:
                    status = f"{C_YELLOW}CER={cer:.2f}{C_RESET}"
                elif f1 >= 1.0:
                    status = f"{C_GREEN}✓{C_RESET}"
                elif f1 >= 0.5:
                    status = f"{C_YELLOW}F1={f1:.2f}{C_RESET}"
                else:
                    status = f"{C_RED}✗{C_RESET}"

                gt_str = _truncate_str(gt_val, 15)
                pred_str = _truncate_str(pred_val, 15)

                print(f"{field_path:<35} {gt_str:<15} {pred_str:<15} {status}")

        elif isinstance(pred_section, list) and pred_section:
            pred_list = pred_section
            gt_list = gt_section

            if section == "medicines":
                med_details = score.get("medicine_details", {})
                matches = med_details.get("matches", [])
                unmatched_pred = med_details.get("unmatched_predicted", [])
                unmatched_gt = med_details.get("unmatched_ground_truth", [])
                strict_matches = per_field.get("medicines", {}).get("strict_matches", [])

                for m in matches:
                    gt_name = m.get("ground_truth", {}).get("name", "?")
                    pred_name = m.get("predicted", {}).get("name", "?")
                    sim = m.get("similarity", 0.0)
                    strict_pass = any(sm.get("predicted") == m.get("predicted") and sm.get("ground_truth") == m.get("ground_truth") for sm in strict_matches)
                    if sim >= 0.8:
                        status = f"{C_GREEN}✓{C_RESET}"
                    elif sim >= 0.5:
                        status = f"{C_YELLOW}~{C_RESET}"
                    else:
                        status = f"{C_RED}✗{C_RESET}"
                    print(f"  [{status}] GT: {gt_name:<25} PRED: {pred_name:<25} (sim={sim:.2f}, strict={'pass' if strict_pass else 'fail'})")
                for m in unmatched_pred:
                    print(f"  [{C_RED}EXTRA{C_RESET}] PRED: {m.get('name', '?')}")
                for m in unmatched_gt:
                    print(f"  [{C_RED}MISSING{C_RESET}] GT: {m.get('name', '?')}")

            elif section == "other_charges":
                oc_details = score.get("other_charges_details", {})
                matches = oc_details.get("matches", [])
                unmatched_pred = oc_details.get("unmatched_predicted", [])
                unmatched_gt = oc_details.get("unmatched_ground_truth", [])
                strict_matches = per_field.get("other_charges", {}).get("strict_matches", [])

                for m in matches:
                    gt_cat = m.get("ground_truth", {}).get("category", "?")
                    pred_cat = m.get("predicted", {}).get("category", "?")
                    gt_canon = m.get("ground_truth_category_canonical")
                    pred_canon = m.get("predicted_category_canonical")
                    sim = m.get("similarity", 0.0)
                    strict_pass = any(sm.get("predicted") == m.get("predicted") and sm.get("ground_truth") == m.get("ground_truth") for sm in strict_matches)
                    if sim >= 0.8:
                        status = f"{C_GREEN}✓{C_RESET}"
                    elif sim >= 0.5:
                        status = f"{C_YELLOW}~{C_RESET}"
                    else:
                        status = f"{C_RED}✗{C_RESET}"
                    canon_extra = f", canon={gt_canon}->{pred_canon}" if gt_canon or pred_canon else ""
                    print(f"  [{status}] GT: {gt_cat:<15} PRED: {pred_cat:<15} (sim={sim:.2f}, strict={'pass' if strict_pass else 'fail'}{canon_extra})")
                for m in unmatched_pred:
                    print(f"  [{C_RED}EXTRA{C_RESET}] PRED: {m.get('category', '?')}")
                for m in unmatched_gt:
                    print(f"  [{C_RED}MISSING{C_RESET}] GT: {m.get('category', '?')}")

            else:
                for i, (gt_item, pred_item) in enumerate(zip_longest(gt_list, pred_list)):
                    if gt_item and pred_item:
                        print(f"  [{i}] {C_GREEN}✓{C_RESET} both match")
                    elif gt_item:
                        gt_str = str(gt_item)[:20]
                        print(f"  [{i}] GT: {gt_str:<20} {C_RED}[MISSING]{C_RESET}")
                    elif pred_item:
                        pred_str = str(pred_item)[:20]
                        print(f"  [{i}] GT: {'-':<20} PRED: {pred_str:<20} {C_RED}[EXTRA]{C_RESET}")

        elif isinstance(gt_section, list) and gt_section:
            for i, gt_item in enumerate(gt_section):
                gt_str = str(gt_item)[:20]
                print(f"  [{i}] GT: {gt_str:<20} {C_RED}[MISSING]{C_RESET}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evaluation on medical claim extraction")
    parser.add_argument("--folder", type=str, required=True, help="Path to folder containing claim sub-folders")
    parser.add_argument("--db-url", type=str, default=None, help="Database URL")
    parser.add_argument("--output", type=str, default=None, help="Path to write JSON results")
    parser.add_argument("--results-dir", type=str, default=None, help="Directory for per-claim scan cache")
    parser.add_argument("--mode", type=str, choices=["auto", "eval-only", "scan-only"], default="auto")
    parser.add_argument("--refresh", action="store_true", help="Force fresh scan")
    parser.add_argument("--allow-stale-cache", action="store_true", help="Allow stale cache")
    parser.add_argument("--log-level", type=str, choices=["quiet", "info", "debug"], default="info")
    parser.add_argument("--detail", type=str, default=None, help="Show detailed breakdown for a claim")
    parser.add_argument("--diff", action="store_true", help="Show visual diff for the --detail claim")
    parser.add_argument("--diff-out", type=str, default=None, help="Output diff to HTML file")
    args = parser.parse_args()

    root_folder = Path(args.folder)
    if not root_folder.exists():
        print(f"Error: folder not found: {root_folder}")
        sys.exit(1)

    claim_folders = sorted([f for f in root_folder.iterdir() if f.is_dir() and f.name not in EXCLUDED_FOLDER_NAMES], key=lambda x: x.name)
    if not claim_folders:
        print(f"No sub-folders found in {root_folder}")
        sys.exit(1)

    os.environ["APP_ENV"] = "test"
    if args.db_url:
        os.environ["DATABASE_URL"] = args.db_url
    try:
        from app.core.config import get_settings
        get_settings.cache_clear()
    except Exception:
        pass
    initialize_database()

    results_dir = Path(args.results_dir) if args.results_dir else root_folder / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    _emit_log(f"Found {len(claim_folders)} claim folder(s)", configured_level=args.log_level, level="info", force=True)

    if args.detail:
        detail_folder = root_folder / args.detail
        if not detail_folder.exists():
            print(f"Error: claim folder not found: {detail_folder}")
            sys.exit(1)
        if args.detail not in {f.name for f in claim_folders}:
            print(f"Error: claim '{args.detail}' not found")
            sys.exit(1)

        async def run_detail(claim_folder: Path) -> dict[str, Any]:
            def logger(level: str, message: str) -> None:
                _emit_log(message, configured_level="info", level=level, claim_id=claim_folder.name, claim_index=1, total_claims=1)
            return await run_single_evaluation(folder=claim_folder, results_dir=results_dir, mode=args.mode, refresh=args.refresh, allow_stale_cache=args.allow_stale_cache, logger=logger)

        result = asyncio.run(run_detail(detail_folder))
        if "error" in result:
            print(f"{C_RED}Error: {result['error']}{C_RESET}")
            sys.exit(1)

        gt_data, _ = load_ground_truth(detail_folder)
        print(f"\n{C_BOLD}CLAIM: {args.detail}{C_RESET}")
        print_claim_detail(result, gt_data)

        if args.diff:
            html_path = Path(args.diff_out) if args.diff_out else None
            print(f"\n{C_BOLD}VISUAL DIFF{C_RESET}")
            print_claim_diff(result.get("predicted", {}), gt_data.get("expected", {}), result.get("score", {}), html_path, result.get("claim_id"))
        sys.exit(0)

    results: list[dict[str, Any]] = []
    total_claims = len(claim_folders)
    run_started_at = time.time()

    for index, folder in enumerate(claim_folders, start=1):
        _emit_log(f"Processing claim folder: {folder}", configured_level=args.log_level, level="info", claim_id=folder.name, claim_index=index, total_claims=total_claims)

        async def run_single(claim_folder: Path, claim_idx: int, total: int) -> dict[str, Any]:
            def logger(level: str, message: str) -> None:
                _emit_log(message, configured_level=args.log_level, level=level, claim_id=claim_folder.name, claim_index=claim_idx, total_claims=total)
            return await run_single_evaluation(folder=claim_folder, results_dir=results_dir, mode=args.mode, refresh=args.refresh, allow_stale_cache=args.allow_stale_cache, logger=logger)

        try:
            result = asyncio.run(run_single(folder, index, total_claims))
        except GroundTruthError as exc:
            result = {"claim_id": folder.name, "error": str(exc), "elapsed_seconds": 0.0, "scan_source": "cache"}
        results.append(result)

        if "error" in result:
            _emit_log(f"ERROR: {result['error'][:80]}", configured_level=args.log_level, level="info", claim_id=folder.name, claim_index=index, total_claims=total_claims, force=True)
        else:
            primary = _get_primary_metric(result.get("score", {}))
            _emit_log(f"KIE_F1={primary.get('f1', 0.0):.4f}", configured_level=args.log_level, level="info", claim_id=folder.name, claim_index=index, total_claims=total_claims)
            _emit_log(f"saved ({result.get('scan_source')}, {result.get('elapsed_seconds', 0):.1f}s)", configured_level=args.log_level, level="info", claim_id=folder.name, claim_index=index, total_claims=total_claims)

    if args.log_level != "quiet":
        print_results(results, args.mode)

    output_data = {
        "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_claims": len(claim_folders),
        "mode": args.mode,
        "results_dir": str(results_dir),
        "evaluation_schema": EVALUATOR_SCHEMA_VERSION,
        "primary_metric": PRIMARY_METRIC,
        "aggregate_summary": build_aggregate_summary(results),
        "results": results,
    }
    output_path = Path(args.output) if args.output else root_folder / "eval_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    total_elapsed = time.time() - run_started_at
    _emit_log(f"Results written to {output_path}", configured_level=args.log_level, level="info", force=True)
    _emit_log(f"Total wall-clock: {total_elapsed:.2f}s", configured_level=args.log_level, level="debug", force=True)


if __name__ == "__main__":
    main()
