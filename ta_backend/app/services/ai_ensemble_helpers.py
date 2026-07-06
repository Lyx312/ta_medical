from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Collection, cast

from app.core.config import ROOT_DIR
from app.services.ai_ensemble_config import (
    CALIBRATION_VERSION,
    CONSULTATION_KEYWORDS,
    CRITICAL_FIELD_PATHS,
    DATE_FIELD_SUFFIXES,
    FINANCIAL_FIELD_PATHS,
    GENERIC_MEDICINE_KEYWORDS,
    NUMERIC_FIELD_PATHS,
    OTHER_CHARGE_DESCRIPTOR_STOPWORDS,
    PACKAGE_LEVEL_MEDICINE_KEYWORDS,
    PHARMACY_PAYMENT_KEYWORDS,
    SERVICE_KEYWORDS,
    SELF_CONSISTENCY_CALIBRATION_FEATURES,
    SEMANTIC_FIELD_PATHS,
    SEMANTIC_FIELD_SUFFIXES,
)
from app.services.standard_vocabulary import (
    map_category_to_standard,
    normalize_facility_name,
    normalize_to_standard_document_type,
)


COMPOUND_MEDICINE_SPLIT_PATTERN = re.compile(r",|\+|/|&|\bdan\b", re.IGNORECASE)
ALPHANUMERIC_TOKEN_SPLIT_PATTERN = re.compile(r"[^a-z0-9]+")
LIST_ITEM_FIELD_PATH_PATTERN = re.compile(r"^(diagnoses|medicines|other_charges)\[([^\]]+)\]\.([a-z_]+)$")
TERBILANG_MATCH_PATTERN = re.compile(r"\b(?:terbilang|rupiah)\b", re.IGNORECASE)
FINANCIAL_TOTAL_FIELD_SEQUENCE = (
    "summary.total_amount",
    "financial_recap.total_medicines",
    "financial_recap.total_other_charges",
    "financial_recap.grand_total",
)
TERBILANG_CONFLICT_TOLERANCE_RATIO = 0.01
TERBILANG_CONFLICT_TOLERANCE_ABSOLUTE = 5000.0
TERBILANG_OVERRIDE_MARGIN = 0.15

INDONESIAN_NUMBER_WORDS: dict[str, int] = {
    "nol": 0,
    "satu": 1,
    "dua": 2,
    "tiga": 3,
    "empat": 4,
    "lima": 5,
    "enam": 6,
    "tujuh": 7,
    "delapan": 8,
    "sembilan": 9,
    "sepuluh": 10,
    "sebelas": 11,
}
INDONESIAN_SCALE_WORDS = {"belas", "puluh", "ratus", "ribu", "juta"}
INDONESIAN_IGNORED_AMOUNT_WORDS = {"rupiah", "rp", "terbilang", "dan"}
INDONESIAN_ALIAS_WORD_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "seratus": ("satu", "ratus"),
    "seribu": ("satu", "ribu"),
    "sejuta": ("satu", "juta"),
}
INDONESIAN_WORD_TOKEN_PATTERN = re.compile(r"[a-z]+", re.IGNORECASE)


def contains_keyword(text: str, keywords: Collection[str]) -> bool:
    normalized_text = re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()
    padded_text = f" {normalized_text} "
    for keyword in keywords:
        normalized_keyword = re.sub(r"[^a-z0-9]+", " ", keyword.casefold()).strip()
        if normalized_keyword and f" {normalized_keyword} " in padded_text:
            return True
    return False


def sequence_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, left, right).ratio()


def is_critical_field_path(field_path: str) -> bool:
    return field_path in CRITICAL_FIELD_PATHS


def is_semantic_field_path(field_path: str) -> bool:
    return field_path in SEMANTIC_FIELD_PATHS or field_path.endswith(SEMANTIC_FIELD_SUFFIXES)


def is_financial_field_path(field_path: str) -> bool:
    return field_path in FINANCIAL_FIELD_PATHS


def is_generic_medicine_key(value: str) -> bool:
    lowered_value = value.casefold()
    return lowered_value in {
        "obat",
        "obatan",
        "obatobatan",
        "medicine",
        "medication",
        "drug",
        "pharmacy",
        "apotek",
        "apotik",
        "farmasi",
        "resep",
        "salinanresep",
    }


def clamp_confidence(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 4)


@lru_cache(maxsize=2048)
def normalize_whitespace(value: str) -> str:
    return " ".join(value.strip().split())


@lru_cache(maxsize=4096)
def normalize_comparison_text(value: str) -> str:
    return re.sub(r"[\W_]+", "", normalize_whitespace(value).casefold())


def is_package_level_medicine_name(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    normalized_value = normalize_whitespace(value)
    if not normalized_value:
        return False
    lowered_value = normalized_value.casefold()
    normalized_key = normalize_comparison_text(normalized_value)
    return (
        contains_keyword(lowered_value, PACKAGE_LEVEL_MEDICINE_KEYWORDS)
        or is_generic_medicine_key(normalized_key)
    )


def extract_compound_medicine_components(value: Any) -> list[str]:
    if not isinstance(value, str):
        return []
    normalized_value = normalize_whitespace(value)
    if not normalized_value:
        return []
    parts = COMPOUND_MEDICINE_SPLIT_PATTERN.split(normalized_value)
    components: list[str] = []
    seen: set[str] = set()
    for part in parts:
        normalized_part = normalize_candidate_key("medicines.name", part)
        if normalized_part in {"__null__", ""} or is_generic_medicine_key(normalized_part):
            continue
        if len(normalized_part) < 4 or normalized_part in seen:
            continue
        seen.add(normalized_part)
        components.append(normalized_part)
    return components


def normalize_other_charge_bucket(raw_kind: str) -> str:
    normalized_kind = normalize_whitespace(str(raw_kind or "")).casefold()
    return normalized_kind if normalized_kind in {"medicine", "consultation", "service"} else "other"


def classify_other_charge_kind(category: Any, description: Any) -> str:
    category_text = normalize_whitespace(str(category or "")).casefold()
    description_text = normalize_whitespace(str(description or "")).casefold()
    combined_text = f"{category_text} {description_text}".strip()

    has_consultation_keyword = contains_keyword(combined_text, CONSULTATION_KEYWORDS)
    has_service_keyword = contains_keyword(combined_text, SERVICE_KEYWORDS)
    category_standard = map_category_to_standard(category_text)
    prescription_only_keywords = ("prescription", "resep", "salinan resep")
    strong_pharmacy_keywords = tuple(keyword for keyword in PHARMACY_PAYMENT_KEYWORDS if keyword not in prescription_only_keywords)
    has_pharmacy_payment_keyword = contains_keyword(combined_text, strong_pharmacy_keywords)
    has_prescription_keyword = contains_keyword(combined_text, prescription_only_keywords)
    category_is_service_like = category_standard in {"CONSULTATION", "SERVICE"} or has_consultation_keyword or has_service_keyword
    if has_pharmacy_payment_keyword or (has_prescription_keyword and not category_is_service_like):
        return "MEDICINE"

    standard_cat = map_category_to_standard(combined_text)
    if standard_cat and standard_cat != "OTHER":
        return standard_cat

    has_medicine_keyword = contains_keyword(combined_text, GENERIC_MEDICINE_KEYWORDS)

    if has_medicine_keyword and (has_consultation_keyword or has_service_keyword):
        return "OTHER"
    if has_medicine_keyword:
        return "MEDICINE"
    if has_service_keyword:
        return "SERVICE"
    if has_consultation_keyword:
        return "CONSULTATION"
    return "OTHER"


def extract_other_charge_descriptor_signature(category: Any, description: Any) -> str:
    raw_parts = [normalize_whitespace(str(description or "")).casefold(), normalize_whitespace(str(category or "")).casefold()]
    tokens: list[str] = []
    seen_tokens: set[str] = set()
    for part in raw_parts:
        for token in ALPHANUMERIC_TOKEN_SPLIT_PATTERN.split(part):
            normalized_token = normalize_comparison_text(token)
            if (
                not normalized_token
                or normalized_token in OTHER_CHARGE_DESCRIPTOR_STOPWORDS
                or is_generic_medicine_key(normalized_token)
            ):
                continue
            if normalized_token in seen_tokens:
                continue
            seen_tokens.add(normalized_token)
            tokens.append(normalized_token)

    if tokens:
        return "".join(tokens)

    description_key = normalize_candidate_key("other_charges.description", description)
    if description_key != "__null__":
        return description_key

    category_key = normalize_candidate_key("other_charges.category", category)
    return category_key if category_key != "__null__" else "other"


def _tokenize_indonesian_amount_text(text: str) -> list[str]:
    raw_tokens = INDONESIAN_WORD_TOKEN_PATTERN.findall(normalize_whitespace(text).casefold())
    normalized_tokens: list[str] = []
    for token in raw_tokens:
        expansion = INDONESIAN_ALIAS_WORD_EXPANSIONS.get(token)
        if expansion is not None:
            normalized_tokens.extend(expansion)
            continue
        normalized_tokens.append(token)
    return normalized_tokens


def _extract_indonesian_number_sequences(tokens: list[str]) -> list[list[str]]:
    number_words = set(INDONESIAN_NUMBER_WORDS) | INDONESIAN_SCALE_WORDS
    sequences: list[list[str]] = []
    current_sequence: list[str] = []

    for token in tokens:
        if token in number_words:
            current_sequence.append(token)
            continue
        if token in INDONESIAN_IGNORED_AMOUNT_WORDS:
            continue
        if current_sequence:
            sequences.append(current_sequence)
            current_sequence = []

    if current_sequence:
        sequences.append(current_sequence)
    return sequences


def _parse_indonesian_sub_thousand(tokens: list[str]) -> tuple[int | None, int]:
    if not tokens:
        return None, 0

    total = 0
    current = 0
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in INDONESIAN_NUMBER_WORDS:
            value = INDONESIAN_NUMBER_WORDS[token]
            next_token = tokens[index + 1] if index + 1 < len(tokens) else None
            if next_token == "belas":
                current += 10 + value
                index += 2
                continue
            if next_token == "puluh":
                current += value * 10
                index += 2
                continue
            if next_token == "ratus":
                current += value * 100
                index += 2
                continue
            current += value
            index += 1
            continue
        if token == "ratus":
            current += 100
            index += 1
            continue
        break

    total += current
    return total if index > 0 else None, index


def _parse_indonesian_number_sequence(tokens: list[str]) -> tuple[int | None, int]:
    if not tokens:
        return None, 0

    total = 0
    index = 0
    consumed_any = False

    while index < len(tokens):
        token = tokens[index]
        if token == "juta":
            total += 1_000_000
            consumed_any = True
            index += 1
            continue
        if token == "ribu":
            total += 1_000
            consumed_any = True
            index += 1
            continue

        chunk_value, consumed = _parse_indonesian_sub_thousand(tokens[index:])
        if chunk_value is None or consumed <= 0:
            break
        index += consumed
        scale_token = tokens[index] if index < len(tokens) else None
        if scale_token == "juta":
            total += chunk_value * 1_000_000
            index += 1
        elif scale_token == "ribu":
            total += chunk_value * 1_000
            index += 1
        else:
            total += chunk_value
        consumed_any = True

    return (total if consumed_any else None), index


def parse_indonesian_amount_words(text: str) -> tuple[float | int | None, float]:
    tokens = _tokenize_indonesian_amount_text(text)
    sequences = _extract_indonesian_number_sequences(tokens)
    if not sequences:
        numeric_match = re.search(r"[0-9][0-9.,]*", text)
        numeric_fallback = coerce_numeric_value(numeric_match.group(0)) if numeric_match is not None else None
        return numeric_fallback, (0.55 if numeric_fallback is not None else 0.0)

    best_value: float | int | None = None
    best_confidence = 0.0
    for sequence in sequences:
        parsed_value, consumed = _parse_indonesian_number_sequence(sequence)
        if parsed_value is None or consumed <= 0:
            continue
        normalized_number = " ".join(sequence[:consumed])
        parsed_numeric = coerce_numeric_value(normalized_number)
        if parsed_numeric is not None:
            parsed_value = parsed_numeric
        coverage = consumed / max(len(sequence), 1)
        confidence = round(min(1.0, 0.55 + (0.45 * coverage)), 4)
        if confidence > best_confidence:
            best_value = parsed_value
            best_confidence = confidence

    return best_value, best_confidence


def parse_other_charge_group_key(group_key: str) -> tuple[str, str | None, str]:
    raw_kind, _, payload = group_key.partition(":")
    if "|" in payload:
        amount_text, _, descriptor_key = payload.partition("|")
    else:
        amount_text, _, descriptor_key = payload.partition(":")
    return (raw_kind or "other").casefold(), amount_text or None, descriptor_key or "other"


def other_charge_kinds_share_family(left_kind: str, right_kind: str) -> bool:
    left_kind = normalize_whitespace(left_kind).casefold()
    right_kind = normalize_whitespace(right_kind).casefold()
    return left_kind == right_kind or {left_kind, right_kind} <= {"service", "consultation"}


def consensus_sort_tuple(
    *,
    run_count: int,
    producer_count: int,
    stable_key: Any,
    null_last: bool = False,
) -> tuple[Any, ...]:
    return (-int(run_count), -int(producer_count), bool(null_last), stable_key)


def choose_target_list_count(counts: list[int]) -> int:
    if not counts:
        return 0

    frequencies: dict[int, int] = {}
    for count in counts:
        frequencies[count] = frequencies.get(count, 0) + 1

    return sorted(frequencies.items(), key=lambda item: (-item[1], -item[0]))[0][0]


DEFAULT_SELF_CONSISTENCY_CALIBRATION_ARTIFACT_PATH = (
    ROOT_DIR / "app" / "artifacts" / "ai_ensemble_self_consistency_logistic_v1.json"
)


@dataclass(frozen=True)
class SelfConsistencyCalibrationArtifact:
    calibration_version: str
    feature_names: tuple[str, ...]
    coefficients: tuple[float, ...]
    intercept: float
    training_claim_count: int
    created_at: str | None = None

    def predict_probability(self, features: dict[str, float]) -> float:
        logit = self.intercept
        for index, feature_name in enumerate(self.feature_names):
            logit += self.coefficients[index] * float(features.get(feature_name, 0.0))
        return clamp_confidence(1.0 / (1.0 + math.exp(-max(min(logit, 35.0), -35.0))))

    @property
    def artifact_status(self) -> str:
        return "fitted" if self.training_claim_count > 0 else "bootstrap_unfitted"

    @property
    def warning_note(self) -> str | None:
        if self.training_claim_count > 0:
            return None
        return "Self-consistency calibration artifact is bootstrap-only and has not been fit on labeled claims yet."


def _resolve_self_consistency_calibration_artifact_path() -> Path:
    override_path = os.getenv("AI_ENSEMBLE_CALIBRATION_ARTIFACT", "").strip()
    if override_path:
        return Path(override_path)
    return DEFAULT_SELF_CONSISTENCY_CALIBRATION_ARTIFACT_PATH


@lru_cache(maxsize=1)
def load_self_consistency_calibration_artifact() -> SelfConsistencyCalibrationArtifact:
    artifact_path = _resolve_self_consistency_calibration_artifact_path()
    try:
        with artifact_path.open("r", encoding="utf-8") as artifact_file:
            payload = json.load(artifact_file)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Self-consistency calibration artifact is missing: {artifact_path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Self-consistency calibration artifact is invalid JSON: {artifact_path}") from exc

    if not isinstance(payload, dict):
        raise RuntimeError(f"Self-consistency calibration artifact must be a JSON object: {artifact_path}")

    raw_feature_names = payload.get("feature_names")
    raw_coefficients = payload.get("coefficients")
    if not isinstance(raw_feature_names, list) or not all(isinstance(name, str) for name in raw_feature_names):
        raise RuntimeError(f"Self-consistency calibration artifact has invalid feature names: {artifact_path}")
    if tuple(raw_feature_names) != SELF_CONSISTENCY_CALIBRATION_FEATURES:
        raise RuntimeError(
            "Self-consistency calibration artifact feature names do not match runtime expectation."
        )
    if not isinstance(raw_coefficients, list) or len(raw_coefficients) != len(raw_feature_names):
        raise RuntimeError(f"Self-consistency calibration artifact has invalid coefficients: {artifact_path}")

    try:
        coefficients = tuple(float(value) for value in raw_coefficients)
        intercept = float(payload.get("intercept", 0.0))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Self-consistency calibration artifact contains non-numeric weights: {artifact_path}") from exc

    training_claim_count = payload.get("training_claim_count", 0)
    try:
        training_claim_count = int(training_claim_count)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Self-consistency calibration artifact has invalid training claim count: {artifact_path}") from exc

    return SelfConsistencyCalibrationArtifact(
        calibration_version=str(payload.get("calibration_version") or CALIBRATION_VERSION),
        feature_names=tuple(raw_feature_names),
        coefficients=coefficients,
        intercept=intercept,
        training_claim_count=training_claim_count,
        created_at=cast(str | None, payload.get("created_at")),
    )


def clear_self_consistency_calibration_cache() -> None:
    load_self_consistency_calibration_artifact.cache_clear()


def infer_document_type_from_label(label: str) -> str:
    normalized = normalize_comparison_text(label)
    if any(keyword in normalized for keyword in ("resep", "prescription")):
        return "prescription"
    if any(keyword in normalized for keyword in ("formulir", "medicalform", "rawatjalan", "doctornote", "clinicalnote")):
        return "medical_form"
    if any(keyword in normalized for keyword in ("lab", "laboratorium")):
        return "lab_result"
    if any(keyword in normalized for keyword in ("radiology", "radiologi", "xray")):
        return "radiology_result"
    if any(keyword in normalized for keyword in ("receipt", "kwitansi", "invoice", "struk", "apotek")):
        return "receipt"
    return "unknown"


def slugify_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-") or "item"


def coerce_numeric_value(value: Any) -> float | int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            return None
        return int(value) if isinstance(value, float) and value.is_integer() else value
    if not isinstance(value, str):
        return None

    cleaned = value.strip().replace(" ", "").replace("IDR", "").replace("idr", "")
    if "," in cleaned and "." in cleaned:
        cleaned = cleaned.replace(".", "").replace(",", ".") if cleaned.rfind(",") > cleaned.rfind(".") else cleaned.replace(",", "")
    elif cleaned.count(",") > 1:
        cleaned = cleaned.replace(",", "")
    elif cleaned.count(".") > 1:
        cleaned = cleaned.replace(".", "")
    else:
        cleaned = cleaned.replace(",", "")

    cleaned = re.sub(r"[^0-9.\-]", "", cleaned)
    if cleaned in {"", "-", ".", "-."}:
        return None
    try:
        numeric_value = float(cleaned)
    except ValueError:
        return None
    return int(numeric_value) if numeric_value.is_integer() else numeric_value


def coerce_date_value(value: Any) -> str | None:
    if value in {None, ""}:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if not isinstance(value, str):
        return None

    normalized = normalize_whitespace(value)
    if not normalized:
        return None
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(normalized, fmt).date().isoformat()
        except ValueError:
            continue
    return normalized if re.fullmatch(r"\d{4}-\d{2}-\d{2}", normalized) else None


def normalize_gender_value(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = normalize_whitespace(value).casefold()
    if normalized in {"male", "m", "l", "lakilaki", "laki laki", "pria"}:
        return "MALE"
    if normalized in {"female", "f", "p", "perempuan", "wanita"}:
        return "FEMALE"
    if normalized in {"male", "female"}:
        return normalized.upper()
    return None


def normalize_candidate_key(field_path: str, value: Any) -> str:
    if value in {None, ""}:
        return "__null__"
    if field_path.endswith(tuple(DATE_FIELD_SUFFIXES)):
        return coerce_date_value(value) or "__null__"
    if field_path in NUMERIC_FIELD_PATHS or field_path.endswith((".amount", ".unit_price", ".total_price")):
        numeric_value = coerce_numeric_value(value)
        return "__null__" if numeric_value is None else str(numeric_value)
    if field_path.endswith(".gender") or field_path == "patient_info.gender":
        return normalize_gender_value(value) or "__null__"
    if field_path == "summary.medical_facility_name":
        normalized_facility = normalize_facility_name(str(value))
        return normalize_comparison_text(normalized_facility) or "__null__"
    if field_path == "summary.detected_documents":
        normalized_document_type = normalize_to_standard_document_type(str(value))
        return normalize_comparison_text(normalized_document_type) or "__null__"
    return normalize_comparison_text(str(value)) or "__null__"


def format_output_value(field_path: str, value: Any) -> Any:
    if value in {None, ""}:
        return None
    if field_path.endswith(tuple(DATE_FIELD_SUFFIXES)):
        return coerce_date_value(value)
    if field_path in NUMERIC_FIELD_PATHS or field_path.endswith((".amount", ".unit_price", ".total_price")):
        return coerce_numeric_value(value)
    if field_path == "summary.medical_facility_name":
        return normalize_facility_name(str(value))
    if field_path == "summary.detected_documents":
        return normalize_to_standard_document_type(str(value))
    if field_path.endswith(".gender") or field_path == "patient_info.gender":
        return normalize_gender_value(value)
    return normalize_whitespace(str(value))


def serialize_supported_runs(traces: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "producer_id": trace.producer_id,
            "provider": trace.provider,
            "model": trace.model,
            "run_index": trace.run_index,
            "producer_rank": trace.producer_rank,
            "temperature": trace.temperature,
        }
        for trace in traces
    ]


def serialize_field_candidates(candidates: list[dict[str, Any]], *, selected_key: str) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for candidate in candidates:
        traces = cast(list[Any], candidate.get("traces", []))
        producer_count = int(candidate.get("producer_count", len(cast(set[str], candidate.get("producers", set())))))
        serialized.append(
            {
                "normalized_key": candidate["normalized_key"],
                "value": candidate["raw_value"],
                "selected": candidate["normalized_key"] == selected_key,
                "support_count": candidate.get("run_count", 0),
                "producer_count": producer_count,
                "support": serialize_supported_runs(traces),
            }
        )
    return serialized


def build_self_consistency_features(
    field_path: str,
    *,
    vote_count: int,
    producer_count: int,
    runner_up_vote_count: int,
    total_runs: int,
    total_producers: int,
    judge_used: bool,
) -> dict[str, float]:
    vote_share = clamp_confidence(vote_count / max(total_runs, 1))
    producer_share = clamp_confidence(producer_count / max(total_producers, 1))
    margin = clamp_confidence(max(vote_count - runner_up_vote_count, 0) / max(total_runs, 1))
    return {
        "vote_share": vote_share,
        "margin": margin,
        "producer_share": producer_share,
        "judge_used": 1.0 if judge_used else 0.0,
        "is_critical": 1.0 if is_critical_field_path(field_path) else 0.0,
        "is_semantic": 1.0 if is_semantic_field_path(field_path) else 0.0,
    }
