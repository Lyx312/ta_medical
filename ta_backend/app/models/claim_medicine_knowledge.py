from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import DateTime, Float, Integer, JSON, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class ClaimMedicineKnowledge(Base):
    __tablename__ = "claim_medicine_knowledge"
    __table_args__ = (
        UniqueConstraint(
            "context_signature",
            "medicine_name_normalized",
            name="uq_claim_medicine_knowledge_context_medicine",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    context_signature: Mapped[str] = mapped_column(String(64), nullable=False)
    medicine_name_normalized: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    diagnosis_text_normalized: Mapped[str] = mapped_column(String, nullable=False)
    anamnesis_text_normalized: Mapped[str] = mapped_column(String, nullable=False)
    diagnosis_terms_json: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    anamnesis_terms_json: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    support_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    relevant_count: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    uncertain_count: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    not_relevant_count: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    avg_confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    last_reason: Mapped[str | None] = mapped_column(String, nullable=True)
    last_decision_source: Mapped[str] = mapped_column(String(32), nullable=False, default="heuristic_fallback")
    first_seen_claim_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    last_seen_claim_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    def counts(self) -> dict[str, float]:
        return {
            "relevant": float(self.relevant_count),
            "uncertain": float(self.uncertain_count),
            "not_relevant": float(self.not_relevant_count),
        }

    def terms_payload(self) -> dict[str, list[str]]:
        diagnosis_terms = self.diagnosis_terms_json if isinstance(self.diagnosis_terms_json, list) else []
        anamnesis_terms = self.anamnesis_terms_json if isinstance(self.anamnesis_terms_json, list) else []
        return {
            "diagnosis_terms": [str(term) for term in diagnosis_terms],
            "anamnesis_terms": [str(term) for term in anamnesis_terms],
        }

    def context_payload(self) -> dict[str, Any]:
        terms = self.terms_payload()
        return {
            "context_signature": self.context_signature,
            "medicine_name_normalized": self.medicine_name_normalized,
            "diagnosis_text_normalized": self.diagnosis_text_normalized,
            "anamnesis_text_normalized": self.anamnesis_text_normalized,
            **terms,
        }
