from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy.orm import Session

from app.core.config import ROOT_DIR, get_settings
from app.repositories.claim_medicine_knowledge_repository import (
    ClaimMedicineKnowledgeObservation,
    ClaimMedicineKnowledgeRepository,
)
from app.schemas.claim import ClaimMedicine, ClaimMedicineEvaluation, ClaimMedicineInput
from app.services.ai_service import BaseDocumentAIProcessor

TOKEN_SPLIT_RE = re.compile(r"[^a-z0-9]+")
STOPWORDS = {
    "dan",
    "yang",
    "untuk",
    "dengan",
    "pada",
    "dari",
    "atau",
    "di",
    "ke",
    "the",
    "and",
    "for",
    "with",
    "from",
    "this",
    "that",
}
MEDICINE_CONTEXT_HINTS: dict[str, tuple[str, ...]] = {
    "paracetamol": ("demam", "flu", "influenza", "fever", "headache", "sakit", "nyeri"),
    "ibuprofen": ("demam", "nyeri", "inflam", "radang", "pain"),
    "amoxicillin": ("infeksi", "infection", "bacterial", "bakter"),
    "azithromycin": ("infeksi", "infection", "bacterial", "bakter"),
    "cetirizine": ("alergi", "allergy", "gatal", "rhinitis"),
    "loratadine": ("alergi", "allergy", "gatal", "rhinitis"),
    "omeprazole": ("maag", "gastr", "lambung", "reflux", "gerd"),
    "lansoprazole": ("maag", "gastr", "lambung", "reflux", "gerd"),
    "salbutamol": ("asma", "asthma", "sesak", "wheez"),
    "ambroxol": ("batuk", "cough", "dahak"),
    "dextromethorphan": ("batuk", "cough"),
    "vitamin": ("supportive", "pemulihan", "recovery", "supplement"),
}
TERM_SYNONYMS: dict[str, tuple[str, ...]] = {
    "demam": ("febris", "fever", "panas"),
    "batuk": ("cough",),
    "pilek": ("rhinitis", "commoncold", "flu"),
    "infeksi": ("infection", "bakterial", "bacterial"),
    "nyeri": ("pain", "sakit"),
    "maag": ("gastritis", "gerd", "lambung"),
}
HISTORY_SIMILARITY_THRESHOLD = 0.65
HISTORY_MIN_SUPPORT = 2
HISTORY_DOMINANCE_THRESHOLD = 0.70
DEFAULT_RETRIEVAL_LIMIT = 100
DEFAULT_EVIDENCE_LIMIT = 5
MEDICINE_RELEVANCE_JUDGE_PROMPT_FILE_PATH = (
    ROOT_DIR / "app" / "prompts" / "ai" / "medical_claim_medicine_relevance_judge.yaml"
)

RelevanceLabel = Literal["relevant", "uncertain", "not_relevant"]
DecisionSource = Literal["history_strong", "llm_judge", "heuristic_fallback"]


def _normalize_text(value: str | None) -> str:
    if not value:
        return ""

    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return normalized.casefold().strip()


def normalize_medicine_name_for_matching(value: str | None) -> str:
    return _normalize_text(value)


def _tokenize(value: str | None) -> set[str]:
    normalized = _normalize_text(value)
    if not normalized:
        return set()

    raw_tokens = {token for token in TOKEN_SPLIT_RE.split(normalized) if token}
    return {token for token in raw_tokens if len(token) >= 3 and token not in STOPWORDS}


def _expand_terms(tokens: set[str]) -> set[str]:
    expanded = set(tokens)
    for token in list(tokens):
        synonyms = TERM_SYNONYMS.get(token)
        if synonyms:
            expanded.update(synonyms)
    return expanded


def _weighted_jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    union = len(left | right)
    if union == 0:
        return 0.0
    return intersection / union


def _build_context_signature(diagnosis_text_normalized: str, anamnesis_text_normalized: str) -> str:
    payload = f"{diagnosis_text_normalized}||{anamnesis_text_normalized}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _top_label_and_ratio(counts: dict[RelevanceLabel, float]) -> tuple[RelevanceLabel, float]:
    total = max(sum(max(value, 0.0) for value in counts.values()), 0.0)
    if total <= 0:
        return "uncertain", 0.0

    label = max(counts, key=lambda k: counts[k])
    ratio = max(counts.get(label, 0.0), 0.0) / total
    return label, ratio


@dataclass(frozen=True)
class ContextBundle:
    diagnosis_text_normalized: str
    anamnesis_text_normalized: str
    diagnosis_terms: set[str]
    anamnesis_terms: set[str]
    context_signature: str

    @property
    def expanded_diagnosis_terms(self) -> set[str]:
        return _expand_terms(self.diagnosis_terms)

    @property
    def expanded_anamnesis_terms(self) -> set[str]:
        return _expand_terms(self.anamnesis_terms)


@dataclass(frozen=True)
class KnowledgeCandidate:
    row_id: int
    similarity: float
    support_count: int
    dominant_label: RelevanceLabel
    dominant_ratio: float
    avg_confidence: float
    diagnosis_text_normalized: str
    anamnesis_text_normalized: str
    diagnosis_terms: list[str]
    anamnesis_terms: list[str]
    last_reason: str | None
    last_decision_source: DecisionSource


class LLMMedicineJudgement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    relevance: RelevanceLabel
    confidence: float = Field(ge=0, le=1)
    reason: str = Field(min_length=1)
    matched_diagnosis: bool
    matched_anamnesis: bool


class MedicineRelevanceLLMJudge(BaseDocumentAIProcessor):
    def __init__(self, *, prompt_file_path: Path | None = None) -> None:
        super().__init__(
            settings=get_settings(),
            prompt_file_path=prompt_file_path or MEDICINE_RELEVANCE_JUDGE_PROMPT_FILE_PATH,
        )

    async def judge(
        self,
        *,
        medicine_name: str,
        diagnosis_text: str,
        anamnesis_text: str,
        evidence_rows: list[KnowledgeCandidate],
        ai_call_context: dict[str, Any] | None = None,
    ) -> LLMMedicineJudgement:
        model_spec = self.get_primary_enabled_model_spec()
        model = self._create_chat_model(model_spec)
        prompt_bundle = self._load_prompt_bundle()
        evidence_payload = [
            {
                "row_id": row.row_id,
                "similarity": row.similarity,
                "support_count": row.support_count,
                "dominant_label": row.dominant_label,
                "dominant_ratio": row.dominant_ratio,
                "avg_confidence": row.avg_confidence,
                "diagnosis_text": row.diagnosis_text_normalized,
                "anamnesis_text": row.anamnesis_text_normalized,
                "diagnosis_terms": row.diagnosis_terms,
                "anamnesis_terms": row.anamnesis_terms,
                "last_reason": row.last_reason,
                "last_decision_source": row.last_decision_source,
            }
            for row in evidence_rows
        ]
        template_input = {
            "medicine_name_json": json.dumps(medicine_name, ensure_ascii=False),
            "diagnosis_text_json": json.dumps(diagnosis_text, ensure_ascii=False),
            "anamnesis_text_json": json.dumps(anamnesis_text, ensure_ascii=False),
            "historical_evidence_json": json.dumps(evidence_payload, ensure_ascii=False, indent=2),
            "output_contract_json": json.dumps(prompt_bundle.output_contract, indent=2),
        }
        system_prompt = PromptTemplate.from_template(prompt_bundle.system_template).format(**template_input)
        user_prompt = PromptTemplate.from_template(prompt_bundle.user_template).format(**template_input)
        response, _tracking_payload = await self._ainvoke_with_tracking(
            model,
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ],
            provider=model_spec.provider,
            model_name=model_spec.model,
            call_kind="medicine_relevance",
            success_context={
                **(ai_call_context or {}),
                "temperature": model_spec.temperature,
                "request_context_payload": {
                    "medicine_name": medicine_name,
                    "evidence_count": len(evidence_rows),
                },
            },
        )
        payload = self._parse_ai_response_to_json(response.content)
        try:
            return LLMMedicineJudgement.model_validate(payload)
        except ValidationError as exc:
            raise RuntimeError("LLM medicine judgement returned invalid payload.") from exc


@dataclass(frozen=True)
class EvaluatedMedicineDecision:
    medicine: ClaimMedicine
    decision_source: DecisionSource
    trust_weight: float


def _coerce_decision_source(value: Any) -> DecisionSource:
    normalized = _normalize_text(str(value or ""))
    if normalized == "history_strong":
        return "history_strong"
    if normalized == "llm_judge":
        return "llm_judge"
    if normalized == "heuristic_fallback":
        return "heuristic_fallback"
    return "heuristic_fallback"


def _build_context_bundle(diagnosis_text: str, anamnesis_text: str) -> ContextBundle:
    diagnosis_text_normalized = _normalize_text(diagnosis_text)
    anamnesis_text_normalized = _normalize_text(anamnesis_text)
    diagnosis_terms = _tokenize(diagnosis_text_normalized)
    anamnesis_terms = _tokenize(anamnesis_text_normalized)
    context_signature = _build_context_signature(diagnosis_text_normalized, anamnesis_text_normalized)
    return ContextBundle(
        diagnosis_text_normalized=diagnosis_text_normalized,
        anamnesis_text_normalized=anamnesis_text_normalized,
        diagnosis_terms=diagnosis_terms,
        anamnesis_terms=anamnesis_terms,
        context_signature=context_signature,
    )


def _build_history_candidate(row: Any, context: ContextBundle) -> KnowledgeCandidate:
    diagnosis_terms_raw = row.diagnosis_terms_json if isinstance(row.diagnosis_terms_json, list) else []
    anamnesis_terms_raw = row.anamnesis_terms_json if isinstance(row.anamnesis_terms_json, list) else []
    diagnosis_terms = {str(term) for term in diagnosis_terms_raw if isinstance(term, str)}
    anamnesis_terms = {str(term) for term in anamnesis_terms_raw if isinstance(term, str)}

    diagnosis_similarity = _weighted_jaccard(context.expanded_diagnosis_terms, _expand_terms(diagnosis_terms))
    anamnesis_similarity = _weighted_jaccard(context.expanded_anamnesis_terms, _expand_terms(anamnesis_terms))
    similarity = (diagnosis_similarity * 0.7) + (anamnesis_similarity * 0.3)

    counts: dict[RelevanceLabel, float] = {
        "relevant": float(getattr(row, "relevant_count", 0.0) or 0.0),
        "uncertain": float(getattr(row, "uncertain_count", 0.0) or 0.0),
        "not_relevant": float(getattr(row, "not_relevant_count", 0.0) or 0.0),
    }
    dominant_label, dominant_ratio = _top_label_and_ratio(counts)

    return KnowledgeCandidate(
        row_id=int(row.id),
        similarity=similarity,
        support_count=int(getattr(row, "support_count", 0) or 0),
        dominant_label=dominant_label,
        dominant_ratio=dominant_ratio,
        avg_confidence=float(getattr(row, "avg_confidence", 0.0) or 0.0),
        diagnosis_text_normalized=str(getattr(row, "diagnosis_text_normalized", "") or ""),
        anamnesis_text_normalized=str(getattr(row, "anamnesis_text_normalized", "") or ""),
        diagnosis_terms=sorted(diagnosis_terms),
        anamnesis_terms=sorted(anamnesis_terms),
        last_reason=getattr(row, "last_reason", None),
        last_decision_source=_coerce_decision_source(getattr(row, "last_decision_source", "heuristic_fallback")),
    )


def _history_is_strong(candidate: KnowledgeCandidate) -> bool:
    return (
        candidate.similarity >= HISTORY_SIMILARITY_THRESHOLD
        and candidate.support_count >= HISTORY_MIN_SUPPORT
        and candidate.dominant_ratio >= HISTORY_DOMINANCE_THRESHOLD
    )


@dataclass(frozen=True)
class HistoryDecision:
    relevance: RelevanceLabel
    confidence: float
    reason: str
    matched_diagnosis: bool
    matched_anamnesis: bool


def _resolve_history_decision(candidates: list[KnowledgeCandidate], context: ContextBundle) -> HistoryDecision | None:
    if not candidates:
        return None

    strong_candidates = [candidate for candidate in candidates if candidate.similarity >= HISTORY_SIMILARITY_THRESHOLD]
    if not strong_candidates:
        return None

    total_support = sum(max(candidate.support_count, 0) for candidate in strong_candidates)
    if total_support < HISTORY_MIN_SUPPORT:
        return None

    weighted_votes: dict[RelevanceLabel, float] = {
        "relevant": 0.0,
        "uncertain": 0.0,
        "not_relevant": 0.0,
    }
    weighted_confidence = 0.0
    total_weight = 0.0
    diagnosis_match = False
    anamnesis_match = False

    for candidate in strong_candidates:
        support = max(candidate.support_count, 0)
        weight = max(candidate.similarity, 0.0) * max(support, 1)
        weighted_votes[candidate.dominant_label] += weight
        weighted_confidence += max(candidate.avg_confidence, 0.0) * weight
        total_weight += weight

        diagnosis_match = diagnosis_match or bool(context.diagnosis_terms & set(candidate.diagnosis_terms))
        anamnesis_match = anamnesis_match or bool(context.anamnesis_terms & set(candidate.anamnesis_terms))

    top_label, ratio = _top_label_and_ratio(weighted_votes)
    if ratio < HISTORY_DOMINANCE_THRESHOLD:
        return None

    confidence = (weighted_confidence / total_weight) if total_weight > 0 else 0.5
    return HistoryDecision(
        relevance=top_label,
        confidence=max(0.0, min(confidence, 1.0)),
        reason=(
            "Used historical context consensus "
            f"(candidates={len(strong_candidates)}, support={total_support}, dominance={ratio:.2f})."
        ),
        matched_diagnosis=diagnosis_match,
        matched_anamnesis=anamnesis_match,
    )


def _build_medicine_result(
    medicine_input: ClaimMedicineInput,
    *,
    relevance: RelevanceLabel,
    confidence: float,
    reason: str,
    matched_diagnosis: bool,
    matched_anamnesis: bool,
    decision_source: DecisionSource | None,
) -> ClaimMedicine:
    return ClaimMedicine(
        name=medicine_input.name,
        quantity=medicine_input.quantity,
        unit=medicine_input.unit,
        dosage_instruction=medicine_input.dosage_instruction,
        source=medicine_input.source,
        evaluation=ClaimMedicineEvaluation(
            relevance=relevance,
            confidence=max(0.0, min(float(confidence), 1.0)),
            reason=reason.strip() or "No reason provided.",
            matched_diagnosis=matched_diagnosis,
            matched_anamnesis=matched_anamnesis,
            decision_source=decision_source,
        ),
    )


def _evaluate_with_heuristic(medicine: ClaimMedicineInput, context: ContextBundle) -> ClaimMedicine:
    medicine_name = medicine.name.strip()
    medicine_tokens = _tokenize(medicine_name)
    context_tokens = context.diagnosis_terms | context.anamnesis_terms

    if not medicine_name:
        return _build_medicine_result(
            medicine,
            relevance="not_relevant",
            confidence=0.99,
            reason="Medicine name is empty and cannot be evaluated against diagnosis/anamnesis.",
            matched_diagnosis=False,
            matched_anamnesis=False,
            decision_source="heuristic_fallback",
        )

    diagnosis_match = bool(medicine_tokens & context.diagnosis_terms)
    anamnesis_match = bool(medicine_tokens & context.anamnesis_terms)

    hint_match = False
    context_blob = " ".join(sorted(context_tokens))
    for token in medicine_tokens:
        hint_keywords = MEDICINE_CONTEXT_HINTS.get(token)
        if hint_keywords and any(hint in context_blob for hint in hint_keywords):
            hint_match = True
            break

    if diagnosis_match or anamnesis_match or hint_match:
        reason_parts: list[str] = []
        if diagnosis_match:
            reason_parts.append("Matched diagnosis terms")
        if anamnesis_match:
            reason_parts.append("Matched anamnesis terms")
        if hint_match and not reason_parts:
            reason_parts.append("Matched medicine context hints")
        return _build_medicine_result(
            medicine,
            relevance="relevant",
            confidence=0.86 if (diagnosis_match or anamnesis_match) else 0.72,
            reason="; ".join(reason_parts) if reason_parts else "Medicine appears relevant to provided clinical context.",
            matched_diagnosis=diagnosis_match,
            matched_anamnesis=anamnesis_match,
            decision_source="heuristic_fallback",
        )

    if not context_tokens:
        return _build_medicine_result(
            medicine,
            relevance="uncertain",
            confidence=0.5,
            reason="Diagnosis and anamnesis context are insufficient for reliable relevance scoring.",
            matched_diagnosis=False,
            matched_anamnesis=False,
            decision_source="heuristic_fallback",
        )

    return _build_medicine_result(
        medicine,
        relevance="not_relevant",
        confidence=0.83,
        reason="No strong relationship found between medicine terms and diagnosis/anamnesis context.",
        matched_diagnosis=False,
        matched_anamnesis=False,
        decision_source="heuristic_fallback",
    )


async def evaluate_claim_medicines(
    db: Session,
    medicines: list[ClaimMedicineInput],
    diagnosis_text: str,
    anamnesis_text: str,
    *,
    claim_id: int | None = None,
    user_id: int | None = None,
    scan_log_id: int | None = None,
    persist: bool = True,
) -> tuple[list[ClaimMedicine], bool, str | None]:
    knowledge_repository = ClaimMedicineKnowledgeRepository(db)
    llm_judge = MedicineRelevanceLLMJudge()
    context = _build_context_bundle(diagnosis_text, anamnesis_text)
    decisions: list[EvaluatedMedicineDecision] = []

    for medicine in medicines:
        medicine_name_normalized = _normalize_text(medicine.name)
        exact_row = knowledge_repository.get_exact(
            context_signature=context.context_signature,
            medicine_name_normalized=medicine_name_normalized,
        )

        ranked_candidates: list[KnowledgeCandidate] = []
        if exact_row is not None:
            ranked_candidates.append(_build_history_candidate(exact_row, context))
        else:
            candidate_rows = knowledge_repository.list_by_medicine(
                medicine_name_normalized=medicine_name_normalized,
                limit=DEFAULT_RETRIEVAL_LIMIT,
            )
            ranked_candidates = sorted(
                (_build_history_candidate(row, context) for row in candidate_rows),
                key=lambda item: (
                    item.similarity,
                    item.support_count,
                    item.dominant_ratio,
                    item.avg_confidence,
                ),
                reverse=True,
            )

        top_candidate = ranked_candidates[0] if ranked_candidates else None
        history_decision: HistoryDecision | None = None
        if top_candidate and _history_is_strong(top_candidate):
            history_decision = HistoryDecision(
                relevance=top_candidate.dominant_label,
                confidence=max(0.0, min(top_candidate.avg_confidence, 1.0)),
                reason=(
                    "Used historical context match "
                    f"(similarity={top_candidate.similarity:.2f}, support={top_candidate.support_count})."
                ),
                matched_diagnosis=bool(context.diagnosis_terms & set(top_candidate.diagnosis_terms)),
                matched_anamnesis=bool(context.anamnesis_terms & set(top_candidate.anamnesis_terms)),
            )
        else:
            history_decision = _resolve_history_decision(ranked_candidates[:DEFAULT_EVIDENCE_LIMIT], context)

        if history_decision is not None:
            medicine_result = _build_medicine_result(
                medicine,
                relevance=history_decision.relevance,
                confidence=history_decision.confidence,
                reason=history_decision.reason,
                matched_diagnosis=history_decision.matched_diagnosis,
                matched_anamnesis=history_decision.matched_anamnesis,
                decision_source="history_strong",
            )
            decisions.append(
                EvaluatedMedicineDecision(
                    medicine=medicine_result,
                    decision_source="history_strong",
                    trust_weight=1.0,
                )
            )
            continue

        try:
            llm_result = await llm_judge.judge(
                medicine_name=medicine.name,
                diagnosis_text=diagnosis_text,
                anamnesis_text=anamnesis_text,
                evidence_rows=ranked_candidates[:DEFAULT_EVIDENCE_LIMIT],
                ai_call_context={
                    "user_id": user_id,
                    "scan_log_id": scan_log_id,
                    "request_context_payload": {
                        "claim_id": claim_id,
                        "medicine_name": medicine.name,
                    },
                },
            )
            medicine_result = _build_medicine_result(
                medicine,
                relevance=llm_result.relevance,
                confidence=llm_result.confidence,
                reason=llm_result.reason,
                matched_diagnosis=llm_result.matched_diagnosis,
                matched_anamnesis=llm_result.matched_anamnesis,
                decision_source="llm_judge",
            )
            decisions.append(
                EvaluatedMedicineDecision(
                    medicine=medicine_result,
                    decision_source="llm_judge",
                    trust_weight=1.0,
                )
            )
        except Exception:  # noqa: BLE001
            heuristic_medicine = _evaluate_with_heuristic(medicine, context)
            decisions.append(
                EvaluatedMedicineDecision(
                    medicine=heuristic_medicine,
                    decision_source="heuristic_fallback",
                    trust_weight=0.6,
                )
            )

    evaluated = [decision.medicine for decision in decisions]
    non_relevant_names = [
        item.name
        for item in evaluated
        if item.evaluation is not None and item.evaluation.relevance == "not_relevant"
    ]
    requires_additional_review = len(non_relevant_names) > 0
    if not requires_additional_review:
        review_note: str | None = None
    else:
        unique_names = [name for index, name in enumerate(non_relevant_names) if name and name not in non_relevant_names[:index]]
        preview = ", ".join(unique_names[:3])
        if len(unique_names) > 3:
            preview = f"{preview}, and {len(unique_names) - 3} more"
        review_note = (
            "One or more medicines appear not relevant to diagnosis/anamnesis and require manual review"
            + (f": {preview}." if preview else ".")
        )

    if persist:
        _persist_knowledge_rows(
            db=db,
            medicines=evaluated,
            diagnosis_text=diagnosis_text,
            anamnesis_text=anamnesis_text,
            claim_id=claim_id,
        )

    return evaluated, requires_additional_review, review_note


def _trust_weight_for_source(decision_source: DecisionSource | None) -> float:
    if decision_source == "history_strong":
        return 1.0
    if decision_source == "llm_judge":
        return 1.0
    return 0.6


def _persist_knowledge_rows(
    *,
    db: Session,
    medicines: list[ClaimMedicine],
    diagnosis_text: str,
    anamnesis_text: str,
    claim_id: int | None,
) -> None:
    context = _build_context_bundle(diagnosis_text, anamnesis_text)
    repository = ClaimMedicineKnowledgeRepository(db)
    for medicine in medicines:
        if medicine.evaluation is None:
            continue
        if medicine.evaluation.relevance == "not_relevant":
            continue
        repository.upsert_observation(
            ClaimMedicineKnowledgeObservation(
                context_signature=context.context_signature,
                medicine_name_normalized=_normalize_text(medicine.name),
                diagnosis_text_normalized=context.diagnosis_text_normalized,
                anamnesis_text_normalized=context.anamnesis_text_normalized,
                diagnosis_terms=sorted(context.diagnosis_terms),
                anamnesis_terms=sorted(context.anamnesis_terms),
                relevance=medicine.evaluation.relevance,
                confidence=medicine.evaluation.confidence,
                reason=medicine.evaluation.reason,
                decision_source=medicine.evaluation.decision_source or "heuristic_fallback",
                trust_weight=_trust_weight_for_source(medicine.evaluation.decision_source),
                claim_id=claim_id,
            )
        )


def persist_claim_medicine_knowledge(
    db: Session,
    medicines: list[ClaimMedicine],
    diagnosis_text: str,
    anamnesis_text: str,
    *,
    claim_id: int,
) -> None:
    _persist_knowledge_rows(
        db=db,
        medicines=medicines,
        diagnosis_text=diagnosis_text,
        anamnesis_text=anamnesis_text,
        claim_id=claim_id,
    )
