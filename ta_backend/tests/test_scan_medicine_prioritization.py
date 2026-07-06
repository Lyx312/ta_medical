from typing import Any, cast

import pytest
from sqlalchemy.orm import Session

from app.schemas.ai import AIMedicalScanResponse
from app.services import scan_medicine_prioritization as prioritization_service
from app.services.ai_service import DocumentImageInput


def _build_scan_result() -> AIMedicalScanResponse:
    return AIMedicalScanResponse.model_validate(
        {
            "summary": {
                "detected_documents": ["HOSPITAL_INVOICE"],
                "medical_facility_name": "RS Prioritas",
                "service_date": "2026-03-31",
                "transaction_date": "2026-03-31",
                "currency": "IDR",
                "total_amount": 200000,
            },
            "patient_info": {
                "name": "Siti",
                "date_of_birth": "1990-01-01",
                "age": 36,
                "gender": "FEMALE",
                "address": "Jl. Test",
            },
            "medical_context": {
                "anamnesis": "Demam batuk tiga hari",
                "diagnoses": [{"description": "Influenza", "doctor_name": "dr. B"}],
                "therapy_note": "Istirahat",
            },
            "medicines": [
                {
                    "name": "Paracitamol",
                    "quantity": 10,
                    "unit": "TABLET",
                    "dosage_instruction": "3 x 1",
                    "unit_price": 1000,
                    "total_price": 10000,
                },
                {
                    "name": "Amoxicillin",
                    "quantity": 10,
                    "unit": "TABLET",
                    "dosage_instruction": "3 x 1",
                    "unit_price": 2000,
                    "total_price": 20000,
                },
            ],
            "other_charges": [],
            "financial_recap": {
                "total_medicines": 30000,
                "total_other_charges": 170000,
                "grand_total": 200000,
            },
        }
    )


def _build_extraction_meta(*, score_0: float, score_1: float) -> dict[str, Any]:
    return {
        "list_group_meta": {
            "medicines": {
                "selected_groups": [
                    {
                        "output_index": 0,
                        "key": "paracitamol-10000",
                        "score": score_0,
                        "support_count": 3,
                        "producer_count": 2,
                        "representative_value": {"name": "Paracitamol", "total_price": 10000},
                    },
                    {
                        "output_index": 1,
                        "key": "amoxicillin-20000",
                        "score": score_1,
                        "support_count": 3,
                        "producer_count": 2,
                        "representative_value": {"name": "Amoxicillin", "total_price": 20000},
                    },
                ],
                "groups": [
                    {
                        "key": "paracitamol-10000",
                        "score": score_0,
                        "support_count": 3,
                        "producer_count": 2,
                        "excluded_reason": None,
                        "representative_value": {"name": "Paracitamol", "total_price": 10000},
                    },
                    {
                        "key": "amoxicillin-20000",
                        "score": score_1,
                        "support_count": 3,
                        "producer_count": 2,
                        "excluded_reason": None,
                        "representative_value": {"name": "Amoxicillin", "total_price": 20000},
                    },
                    {
                        "key": "paracetamol-10000-candidate",
                        "score": 0.61,
                        "support_count": 1,
                        "producer_count": 1,
                        "excluded_reason": "not selected",
                        "representative_value": {"name": "Paracetamol", "total_price": 10000},
                    },
                ],
            }
        }
    }


def _build_images() -> list[DocumentImageInput]:
    return [
        DocumentImageInput(
            filename="scan.jpg",
            mime_type="image/jpeg",
            base64_data="cHJvY2Vzc2VkLWltYWdl",
        )
    ]


def _fake_db_hints(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
    del args, kwargs
    return [{"medicine_name": "paracetamol", "support_count": 2}]


class _FakeBatchJudge:
    def __init__(self, judgements: list[dict[str, Any]]) -> None:
        self._judgements = judgements

    async def judge_batch(self, **kwargs: Any):
        del kwargs
        return prioritization_service.BatchMedicineInterpretationResponse.model_validate(
            {"judgements": self._judgements}
        )


@pytest.mark.asyncio
async def test_prioritization_sends_extracted_name_to_batch_judge(monkeypatch: pytest.MonkeyPatch) -> None:
    scan_result = _build_scan_result()
    extraction_meta = _build_extraction_meta(score_0=0.91, score_1=0.86)
    captured_kwargs: dict[str, Any] = {}

    class CapturingBatchJudge:
        async def judge_batch(self, **kwargs: Any):
            captured_kwargs.update(kwargs)
            return prioritization_service.BatchMedicineInterpretationResponse(judgements=[])

    monkeypatch.setattr(prioritization_service, "_build_db_medicine_hints", _fake_db_hints)
    monkeypatch.setattr(prioritization_service, "ScanMedicineImageInterpretationJudge", CapturingBatchJudge)

    await prioritization_service.prioritize_scan_medicines(
        cast(Session, object()),
        scan_result,
        extraction_meta,
        images=_build_images(),
        interpretation_mode="suggest_only",
    )

    ambiguous_medicines = captured_kwargs["ambiguous_medicines"]
    assert ambiguous_medicines[0]["extracted_name"] == "Paracitamol"
    assert "name" not in ambiguous_medicines[0]


@pytest.mark.asyncio
async def test_prioritization_keeps_image_dominant_order(monkeypatch: pytest.MonkeyPatch) -> None:
    scan_result = _build_scan_result()
    extraction_meta = _build_extraction_meta(score_0=0.95, score_1=0.70)

    monkeypatch.setattr(prioritization_service, "_build_db_medicine_hints", _fake_db_hints)
    monkeypatch.setattr(
        prioritization_service,
        "ScanMedicineImageInterpretationJudge",
        lambda: _FakeBatchJudge(
            [
                {
                    "original_index": 0,
                    "interpreted_name": "Paracetamol",
                    "interpretation_confidence": 0.99,
                    "interpretation_reason": "visible text",
                    "relevance": "uncertain",
                    "relevance_confidence": 0.5,
                    "matched_diagnosis": True,
                    "matched_anamnesis": True,
                },
                {
                    "original_index": 1,
                    "interpreted_name": "Amoxicillin",
                    "interpretation_confidence": 0.99,
                    "interpretation_reason": "visible text",
                    "relevance": "relevant",
                    "relevance_confidence": 0.95,
                    "matched_diagnosis": True,
                    "matched_anamnesis": True,
                },
            ]
        ),
    )

    prioritized, log_payload = await prioritization_service.prioritize_scan_medicines(
        cast(Session, object()),
        scan_result,
        extraction_meta,
        images=_build_images(),
        interpretation_mode="suggest_only",
    )

    assert [medicine.name for medicine in prioritized.medicines] == ["Paracitamol", "Amoxicillin"]
    assert log_payload["applied"] is False
    assert log_payload["batch_summary"]["batch_item_count"] == 0
    assert log_payload["batch_summary"]["batch_success"] is False
    assert log_payload["strategy"]["interpretation_mode"] == "suggest_only"


@pytest.mark.asyncio
async def test_prioritization_uses_batch_judgement_for_ambiguous_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scan_result = _build_scan_result()
    extraction_meta = _build_extraction_meta(score_0=0.91, score_1=0.86)

    monkeypatch.setattr(prioritization_service, "_build_db_medicine_hints", _fake_db_hints)
    monkeypatch.setattr(
        prioritization_service,
        "ScanMedicineImageInterpretationJudge",
        lambda: _FakeBatchJudge(
            [
                {
                    "original_index": 0,
                    "interpreted_name": "Paracetamol",
                    "interpretation_confidence": 0.96,
                    "interpretation_reason": "close to handwritten note",
                    "relevance": "uncertain",
                    "relevance_confidence": 0.52,
                    "matched_diagnosis": True,
                    "matched_anamnesis": True,
                },
                {
                    "original_index": 1,
                    "interpreted_name": "Amoxicillin",
                    "interpretation_confidence": 0.88,
                    "interpretation_reason": "clear print",
                    "relevance": "relevant",
                    "relevance_confidence": 0.93,
                    "matched_diagnosis": True,
                    "matched_anamnesis": True,
                },
            ]
        ),
    )

    prioritized, log_payload = await prioritization_service.prioritize_scan_medicines(
        cast(Session, object()),
        scan_result,
        extraction_meta,
        images=_build_images(),
        interpretation_mode="suggest_only",
    )

    assert [medicine.name for medicine in prioritized.medicines] == ["Amoxicillin", "Paracitamol"]
    assert prioritized.medicines[0].total_price == 20000
    assert prioritized.medicines[1].total_price == 10000
    assert log_payload["applied"] is True
    assert log_payload["optimized_source_indices"] == [1, 0]


@pytest.mark.asyncio
async def test_prioritization_auto_correct_guarded_applies_only_when_rules_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scan_result = _build_scan_result()
    extraction_meta = _build_extraction_meta(score_0=0.91, score_1=0.86)

    monkeypatch.setattr(prioritization_service, "_build_db_medicine_hints", _fake_db_hints)
    monkeypatch.setattr(
        prioritization_service,
        "ScanMedicineImageInterpretationJudge",
        lambda: _FakeBatchJudge(
            [
                {
                    "original_index": 0,
                    "interpreted_name": "Paracetamol",
                    "interpretation_confidence": 0.98,
                    "interpretation_reason": "clear pack text",
                    "relevance": "relevant",
                    "relevance_confidence": 0.80,
                    "matched_diagnosis": True,
                    "matched_anamnesis": True,
                },
                {
                    "original_index": 1,
                    "interpreted_name": "Something Else",
                    "interpretation_confidence": 0.93,
                    "interpretation_reason": "unclear",
                    "relevance": "relevant",
                    "relevance_confidence": 0.90,
                    "matched_diagnosis": True,
                    "matched_anamnesis": True,
                },
            ]
        ),
    )

    prioritized, log_payload = await prioritization_service.prioritize_scan_medicines(
        cast(Session, object()),
        scan_result,
        extraction_meta,
        images=_build_images(),
        interpretation_mode="auto_correct_guarded",
    )

    assert [medicine.name for medicine in prioritized.medicines] == ["Amoxicillin", "Paracetamol"]
    judgement_logs = log_payload["judgement_logs"]
    first = next(item for item in judgement_logs if item["original_index"] == 0)
    second = next(item for item in judgement_logs if item["original_index"] == 1)
    assert first["status"] == "auto_correct_applied"
    assert second["status"] == "auto_correct_rejected"
