from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ClaimFingerprintInput:
    claim_id: str
    image_entries: list[tuple[str, str]]
    patient_name_candidates: list[str]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_candidate(candidate: str) -> str:
    return " ".join(candidate.strip().casefold().split())


def _normalize_path(path: str) -> str:
    return path.replace("\\", "/").strip()


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file_handle:
        while True:
            chunk = file_handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def build_claim_fingerprint(fingerprint_input: ClaimFingerprintInput) -> str:
    canonical = {
        "claim_id": fingerprint_input.claim_id,
        "images": [
            {
                "path": _normalize_path(relative_path),
                "sha256": image_sha,
            }
            for relative_path, image_sha in fingerprint_input.image_entries
        ],
        "patient_name_candidates": sorted(
            _normalize_candidate(candidate)
            for candidate in fingerprint_input.patient_name_candidates
            if candidate.strip()
        ),
        "fingerprint_schema": "v1",
    }
    payload = json.dumps(canonical, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def get_claim_results_dir(results_dir: Path, claim_id: str) -> Path:
    return results_dir / claim_id


def get_claim_runs_dir(results_dir: Path, claim_id: str) -> Path:
    return get_claim_results_dir(results_dir, claim_id) / "runs"


def get_claim_latest_path(results_dir: Path, claim_id: str) -> Path:
    return get_claim_results_dir(results_dir, claim_id) / "latest.json"


def _safe_claim_id(claim_id: str) -> str:
    normalized = "".join(ch.lower() if ch.isalnum() else "-" for ch in claim_id).strip("-")
    return normalized or "claim"


def make_run_record_filename(*, claim_id: str, scan_log_id: int | None) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    scan_part = f"scan-{scan_log_id}" if scan_log_id is not None else "scan-unknown"
    return f"{timestamp}__{_safe_claim_id(claim_id)}__{scan_part}.json"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2, ensure_ascii=False)


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as file_handle:
            data = json.load(file_handle)
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def load_latest_snapshot(results_dir: Path, claim_id: str) -> dict[str, Any] | None:
    return read_json(get_claim_latest_path(results_dir, claim_id))


def append_run_record(results_dir: Path, claim_id: str, run_record: dict[str, Any]) -> Path:
    runs_dir = get_claim_runs_dir(results_dir, claim_id)
    run_filename = make_run_record_filename(
        claim_id=claim_id,
        scan_log_id=_safe_int(run_record.get("scan_log_id")),
    )
    run_path = runs_dir / run_filename
    write_json(run_path, run_record)
    return run_path


def save_latest_snapshot(results_dir: Path, claim_id: str, snapshot: dict[str, Any]) -> Path:
    latest_path = get_claim_latest_path(results_dir, claim_id)
    write_json(latest_path, snapshot)
    return latest_path


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def build_latest_snapshot(
    *,
    claim_id: str,
    fingerprint: str,
    fingerprint_input: dict[str, Any],
    run_record_path: Path,
    predicted: dict[str, Any],
    scan_log_id: int | None,
    source: str,
) -> dict[str, Any]:
    return {
        "snapshot_schema": "v1",
        "claim_id": claim_id,
        "updated_at": _utc_now_iso(),
        "input_fingerprint": fingerprint,
        "fingerprint_input": fingerprint_input,
        "scan_log_id": scan_log_id,
        "source": source,
        "run_record_path": run_record_path.name,
        "predicted": predicted,
    }


def build_run_record(
    *,
    claim_id: str,
    fingerprint: str,
    fingerprint_input: dict[str, Any],
    scan_log_export: dict[str, Any] | None,
    predicted: dict[str, Any],
    elapsed_seconds: float,
    status: str,
    source: str,
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    scan_log_export = scan_log_export or {}
    return {
        "run_schema": "v1",
        "created_at": _utc_now_iso(),
        "claim_id": claim_id,
        "status": status,
        "source": source,
        "elapsed_seconds": round(elapsed_seconds, 4),
        "input_fingerprint": fingerprint,
        "fingerprint_input": fingerprint_input,
        "scan_log_id": _safe_int(scan_log_export.get("scan_log_id")),
        "scan_log": scan_log_export,
        "predicted": predicted,
        "error": error,
    }


def resolve_run_record_path(results_dir: Path, claim_id: str, latest_snapshot: dict[str, Any]) -> Path | None:
    run_name = latest_snapshot.get("run_record_path")
    if not isinstance(run_name, str) or not run_name.strip():
        return None
    return get_claim_runs_dir(results_dir, claim_id) / run_name


def fingerprint_input_payload(
    *,
    claim_id: str,
    image_entries: list[tuple[str, str]],
    patient_name_candidates: list[str],
) -> dict[str, Any]:
    return {
        "claim_id": claim_id,
        "images": [
            {
                "path": _normalize_path(relative_path),
                "sha256": image_sha,
            }
            for relative_path, image_sha in image_entries
        ],
        "patient_name_candidates": patient_name_candidates,
    }
