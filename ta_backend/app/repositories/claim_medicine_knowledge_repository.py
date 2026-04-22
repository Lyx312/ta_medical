from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.claim_medicine_knowledge import ClaimMedicineKnowledge


@dataclass(frozen=True)
class ClaimMedicineKnowledgeObservation:
    context_signature: str
    medicine_name_normalized: str
    diagnosis_text_normalized: str
    anamnesis_text_normalized: str
    diagnosis_terms: list[str]
    anamnesis_terms: list[str]
    relevance: str
    confidence: float
    reason: str
    decision_source: str
    trust_weight: float
    claim_id: int | None = None


class ClaimMedicineKnowledgeRepository:
    def __init__(self, db: Session) -> None:
        self._db = db

    def get_exact(self, *, context_signature: str, medicine_name_normalized: str) -> ClaimMedicineKnowledge | None:
        return self._db.execute(
            select(ClaimMedicineKnowledge).where(
                ClaimMedicineKnowledge.context_signature == context_signature,
                ClaimMedicineKnowledge.medicine_name_normalized == medicine_name_normalized,
            )
        ).scalar_one_or_none()

    def list_by_medicine(self, *, medicine_name_normalized: str, limit: int = 100) -> list[ClaimMedicineKnowledge]:
        return list(
            self._db.execute(
                select(ClaimMedicineKnowledge)
                .where(ClaimMedicineKnowledge.medicine_name_normalized == medicine_name_normalized)
                .order_by(
                    ClaimMedicineKnowledge.updated_at.desc(),
                    ClaimMedicineKnowledge.support_count.desc(),
                    ClaimMedicineKnowledge.id.desc(),
                )
                .limit(limit)
            ).scalars()
        )

    def list_recent(self, *, limit: int = 200) -> list[ClaimMedicineKnowledge]:
        return list(
            self._db.execute(
                select(ClaimMedicineKnowledge)
                .order_by(
                    ClaimMedicineKnowledge.updated_at.desc(),
                    ClaimMedicineKnowledge.support_count.desc(),
                    ClaimMedicineKnowledge.id.desc(),
                )
                .limit(limit)
            ).scalars()
        )

    def upsert_observation(self, observation: ClaimMedicineKnowledgeObservation) -> ClaimMedicineKnowledge:
        now = datetime.now(timezone.utc)
        row = self.get_exact(
            context_signature=observation.context_signature,
            medicine_name_normalized=observation.medicine_name_normalized,
        )

        weighted_relevant = observation.trust_weight if observation.relevance == "relevant" else 0.0
        weighted_uncertain = observation.trust_weight if observation.relevance == "uncertain" else 0.0
        weighted_not_relevant = observation.trust_weight if observation.relevance == "not_relevant" else 0.0

        if row is None:
            row = ClaimMedicineKnowledge(
                context_signature=observation.context_signature,
                medicine_name_normalized=observation.medicine_name_normalized,
                diagnosis_text_normalized=observation.diagnosis_text_normalized,
                anamnesis_text_normalized=observation.anamnesis_text_normalized,
                diagnosis_terms_json=observation.diagnosis_terms,
                anamnesis_terms_json=observation.anamnesis_terms,
                support_count=1,
                relevant_count=weighted_relevant,
                uncertain_count=weighted_uncertain,
                not_relevant_count=weighted_not_relevant,
                avg_confidence=observation.confidence,
                last_reason=observation.reason,
                last_decision_source=observation.decision_source,
                first_seen_claim_id=observation.claim_id,
                last_seen_claim_id=observation.claim_id,
                created_at=now,
                updated_at=now,
                last_used_at=now,
            )
            self._db.add(row)
            self._db.flush()
            return row

        previous_support = max(int(row.support_count), 0)
        new_support = previous_support + 1
        row.avg_confidence = ((float(row.avg_confidence) * previous_support) + observation.confidence) / new_support
        row.support_count = new_support
        row.relevant_count = float(row.relevant_count) + weighted_relevant
        row.uncertain_count = float(row.uncertain_count) + weighted_uncertain
        row.not_relevant_count = float(row.not_relevant_count) + weighted_not_relevant
        row.last_reason = observation.reason
        row.last_decision_source = observation.decision_source
        if observation.claim_id is not None:
            row.last_seen_claim_id = observation.claim_id
        row.last_used_at = now
        row.updated_at = now

        if not row.first_seen_claim_id and observation.claim_id is not None:
            row.first_seen_claim_id = observation.claim_id

        if observation.diagnosis_terms:
            row.diagnosis_terms_json = observation.diagnosis_terms
        if observation.anamnesis_terms:
            row.anamnesis_terms_json = observation.anamnesis_terms
        if observation.diagnosis_text_normalized:
            row.diagnosis_text_normalized = observation.diagnosis_text_normalized
        if observation.anamnesis_text_normalized:
            row.anamnesis_text_normalized = observation.anamnesis_text_normalized

        self._db.flush()
        return row

    def mark_used(self, row: ClaimMedicineKnowledge) -> None:
        row.last_used_at = datetime.now(timezone.utc)
        self._db.flush()
