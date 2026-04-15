from __future__ import annotations

import re
import unicodedata

from app.schemas.claim import (
    ClaimMedicine,
    ClaimMedicineEvaluation,
    ClaimMedicineInput,
)

TOKEN_SPLIT_RE = re.compile(r"[^a-z0-9]+")

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


def _normalize_text(value: str | None) -> str:
    if not value:
        return ""

    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return normalized.casefold().strip()


def _tokenize(value: str | None) -> set[str]:
    normalized = _normalize_text(value)
    if not normalized:
        return set()

    tokens = {token for token in TOKEN_SPLIT_RE.split(normalized) if token}
    return {token for token in tokens if len(token) >= 3}


def _evaluate_single_medicine(
    medicine: ClaimMedicineInput,
    diagnosis_tokens: set[str],
    anamnesis_tokens: set[str],
) -> ClaimMedicine:
    medicine_name = medicine.name.strip()
    medicine_tokens = _tokenize(medicine_name)
    context_tokens = diagnosis_tokens | anamnesis_tokens

    if not medicine_name:
        return ClaimMedicine(
            name=medicine_name,
            quantity=medicine.quantity,
            unit=medicine.unit,
            dosage_instruction=medicine.dosage_instruction,
            source=medicine.source,
            evaluation=ClaimMedicineEvaluation(
                relevance="not_relevant",
                confidence=0.99,
                reason="Medicine name is empty and cannot be evaluated against diagnosis/anamnesis.",
                matched_diagnosis=False,
                matched_anamnesis=False,
            ),
        )

    diagnosis_match = bool(medicine_tokens & diagnosis_tokens)
    anamnesis_match = bool(medicine_tokens & anamnesis_tokens)

    hint_match = False
    for token in medicine_tokens:
        hint_keywords = MEDICINE_CONTEXT_HINTS.get(token)
        if hint_keywords and any(hint in " ".join(sorted(context_tokens)) for hint in hint_keywords):
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

        return ClaimMedicine(
            name=medicine_name,
            quantity=medicine.quantity,
            unit=medicine.unit,
            dosage_instruction=medicine.dosage_instruction,
            source=medicine.source,
            evaluation=ClaimMedicineEvaluation(
                relevance="relevant",
                confidence=0.86 if (diagnosis_match or anamnesis_match) else 0.72,
                reason="; ".join(reason_parts) if reason_parts else "Medicine appears relevant to provided clinical context.",
                matched_diagnosis=diagnosis_match,
                matched_anamnesis=anamnesis_match,
            ),
        )

    if not context_tokens:
        return ClaimMedicine(
            name=medicine_name,
            quantity=medicine.quantity,
            unit=medicine.unit,
            dosage_instruction=medicine.dosage_instruction,
            source=medicine.source,
            evaluation=ClaimMedicineEvaluation(
                relevance="uncertain",
                confidence=0.5,
                reason="Diagnosis and anamnesis context are insufficient for reliable relevance scoring.",
                matched_diagnosis=False,
                matched_anamnesis=False,
            ),
        )

    return ClaimMedicine(
        name=medicine_name,
        quantity=medicine.quantity,
        unit=medicine.unit,
        dosage_instruction=medicine.dosage_instruction,
        source=medicine.source,
        evaluation=ClaimMedicineEvaluation(
            relevance="not_relevant",
            confidence=0.83,
            reason="No strong relationship found between medicine terms and diagnosis/anamnesis context.",
            matched_diagnosis=False,
            matched_anamnesis=False,
        ),
    )


def evaluate_claim_medicines(
    medicines: list[ClaimMedicineInput],
    diagnosis_text: str,
    anamnesis_text: str,
) -> tuple[list[ClaimMedicine], bool, str | None]:
    diagnosis_tokens = _tokenize(diagnosis_text)
    anamnesis_tokens = _tokenize(anamnesis_text)

    evaluated = [_evaluate_single_medicine(item, diagnosis_tokens, anamnesis_tokens) for item in medicines]

    non_relevant_names = [
        item.name
        for item in evaluated
        if item.evaluation is not None and item.evaluation.relevance == "not_relevant"
    ]

    requires_additional_review = len(non_relevant_names) > 0
    if not requires_additional_review:
        return evaluated, False, None

    unique_names = [name for index, name in enumerate(non_relevant_names) if name and name not in non_relevant_names[:index]]
    preview = ", ".join(unique_names[:3])
    if len(unique_names) > 3:
        preview = f"{preview}, and {len(unique_names) - 3} more"

    reason = (
        "One or more medicines appear not relevant to diagnosis/anamnesis and require manual review"
        + (f": {preview}." if preview else ".")
    )

    return evaluated, True, reason
