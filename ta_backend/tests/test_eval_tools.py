import asyncio
import json
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from eval import generate_template, run_evaluation
from eval.cli_utils import ensure_family_candidates, guess_mime_type, register_test_user
from eval.evaluator import EVALUATOR_SCHEMA_VERSION, PRIMARY_METRIC, MedicalClaimEvaluator
from eval.ground_truth import GroundTruthError, load_ground_truth
from eval.medicine_matcher import match_medicines
from eval.result_store import (
    ClaimFingerprintInput,
    append_run_record,
    build_claim_fingerprint,
    build_latest_snapshot,
    build_run_record,
    fingerprint_input_payload,
    load_latest_snapshot,
    resolve_run_record_path,
    save_latest_snapshot,
)


def _minimal_eval_payload() -> dict[str, object]:
    return {
        "summary": {
            "detected_documents": ["Kwitansi"],
            "medical_facility_name": "RS Sehat Sentosa",
            "service_date": "2024-08-10",
            "transaction_date": None,
            "currency": "IDR",
            "total_amount": 125000,
        },
        "patient_info": {"name": "Budi Saputra", "date_of_birth": None, "age": None, "gender": None, "address": None},
        "medical_context": {"anamnesis": None, "diagnoses": [], "therapy_note": None},
        "medicines": [],
        "other_charges": [],
        "financial_recap": {"total_medicines": 0, "total_other_charges": 125000, "grand_total": 125000},
    }


def build_stub_eval_app(scan_status: int = 200) -> FastAPI:
    app = FastAPI()
    family_members: list[dict[str, object]] = []

    @app.post("/api/auth/register")
    def register() -> JSONResponse:
        return JSONResponse(status_code=201, content={"ok": True})

    @app.post("/api/auth/login")
    def login() -> dict[str, str]:
        return {"access_token": "token"}

    @app.get("/api/family-members")
    def list_family_members() -> list[dict[str, object]]:
        return family_members

    @app.post("/api/family-members")
    def create_family_member(payload: dict[str, str]) -> JSONResponse:
        family_member = {"id": len(family_members) + 1, "name": payload["name"], "relation": payload["relation"]}
        family_members.append(family_member)
        return JSONResponse(status_code=201, content=family_member)

    @app.post("/api/ai/scan-documents")
    def scan_documents() -> JSONResponse:
        if scan_status != 200:
            return JSONResponse(status_code=scan_status, content={"message": "scan failed"})
        return JSONResponse(
            status_code=200,
            content={
                "scan_log_id": 1,
                "summary": {
                    "detected_documents": ["receipt"],
                    "medical_facility_name": "RS Sehat Sentosa",
                    "service_date": "2024-08-10",
                    "transaction_date": None,
                    "currency": "IDR",
                    "total_amount": 125000,
                },
                "patient_info": {
                    "name": "Budi Saputra",
                    "date_of_birth": "1985-05-15",
                    "age": 39,
                    "gender": "MALE",
                    "address": "JL. Pejuang No.1",
                },
                "medical_context": {
                    "anamnesis": "Demam dua hari",
                    "diagnoses": [{"description": "Influenza", "doctor_name": "dr. Budi"}],
                    "therapy_note": "Istirahat dan minum obat",
                },
                "medicines": [
                    {
                        "name": "Paracetamol",
                        "quantity": 10,
                        "unit": "TABLET",
                        "dosage_instruction": "3 x 1",
                        "unit_price": 1000,
                        "total_price": 10000,
                    }
                ],
                "other_charges": [
                    {
                        "category": "CONSULTATION",
                        "description": "Konsultasi dokter",
                        "amount": 115000,
                    }
                ],
                "financial_recap": {
                    "total_medicines": 10000,
                    "total_other_charges": 115000,
                    "grand_total": 125000,
                },
            },
        )

    return app


def write_ground_truth(claim_dir: Path) -> None:
    images_dir = claim_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    (images_dir / "page1.jpg").write_bytes(b"fake-image")
    ground_truth = {
        "images": ["images/page1.jpg"],
        "patient_name_candidates": ["Budi Saputra", "Siti Aminah"],
        "expected": {
            "summary": {
                "detected_documents": ["receipt"],
                "medical_facility_name": "RS Sehat Sentosa",
                "service_date": "2024-08-10",
                "transaction_date": None,
                "currency": "IDR",
                "total_amount": 125000,
            },
            "patient_info": {
                "name": "Budi Saputra",
                "date_of_birth": "1985-05-15",
                "age": 39,
                "gender": "MALE",
                "address": "JL. Pejuang No.1",
            },
            "medical_context": {
                "anamnesis": "Demam dua hari",
                "diagnoses": [{"description": "Influenza", "doctor_name": "dr. Budi"}],
                "therapy_note": "Istirahat dan minum obat",
            },
            "medicines": [
                {
                    "name": "Paracetamol",
                    "quantity": 10,
                    "unit": "TABLET",
                    "dosage_instruction": "3 x 1",
                    "unit_price": 1000,
                    "total_price": 10000,
                }
            ],
            "other_charges": [
                {
                    "category": "CONSULTATION",
                    "description": "Konsultasi dokter",
                    "amount": 115000,
                }
            ],
            "financial_recap": {
                "total_medicines": 10000,
                "total_other_charges": 115000,
                "grand_total": 125000,
            },
        },
    }
    (claim_dir / f"{claim_dir.name}.json").write_text(json.dumps(ground_truth), encoding="utf-8")


def test_match_medicines_tracks_unmatched_prediction_and_ground_truth() -> None:
    predictions = [
        {"name": "Paracetamol 500 mg", "quantity": 10},
        {"name": "Vitamin C", "quantity": 5},
    ]
    ground_truths = [
        {"name": "Paracetamol", "quantity": 10},
        {"name": "Amoxicillin", "quantity": 10},
    ]

    matched, unmatched_pred, unmatched_gt = match_medicines(predictions, ground_truths)

    assert len(matched) == 1
    assert unmatched_pred == [1]
    assert unmatched_gt == [1]


def test_load_ground_truth_supports_relative_image_paths(tmp_path: Path) -> None:
    claim_dir = tmp_path / "claim_001"
    images_dir = claim_dir / "images"
    images_dir.mkdir(parents=True)
    (images_dir / "page1.jpg").write_bytes(b"fake-image")

    ground_truth = {
        "images": ["images/page1.jpg"],
        "patient_name_candidates": ["Budi Saputra"],
        "expected": {
            "summary": {"service_date": "2024-08-10"},
            "patient_info": {"name": "Budi Saputra"},
            "medical_context": {"diagnoses": []},
            "medicines": [],
            "other_charges": [],
        },
    }
    (claim_dir / f"{claim_dir.name}.json").write_text(json.dumps(ground_truth), encoding="utf-8")

    loaded, image_paths = load_ground_truth(claim_dir)

    assert loaded["images"] == ["images/page1.jpg"]
    assert loaded["patient_name_candidates"] == ["Budi Saputra"]
    assert image_paths == [images_dir / "page1.jpg"]


def test_load_ground_truth_rejects_empty_image_list(tmp_path: Path) -> None:
    claim_dir = tmp_path / "claim_001"
    claim_dir.mkdir()
    (claim_dir / f"{claim_dir.name}.json").write_text(json.dumps({"images": [], "expected": {}}), encoding="utf-8")

    with pytest.raises(GroundTruthError):
        load_ground_truth(claim_dir)


def test_medical_claim_evaluator_scores_perfect_match() -> None:
    evaluator = MedicalClaimEvaluator()
    payload = {
        "summary": {
            "detected_documents": ["receipt"],
            "medical_facility_name": "RS Sehat Sentosa",
            "service_date": "2024-08-10",
            "currency": "IDR",
            "total_amount": 125000,
        },
        "patient_info": {
            "name": "Budi Saputra",
            "date_of_birth": "1985-05-15",
            "age": 39,
            "gender": "MALE",
            "address": "JL. Pejuang No.1",
        },
        "medical_context": {
            "anamnesis": "Demam dua hari",
            "diagnoses": [{"description": "Influenza", "doctor_name": "dr. Budi"}],
            "therapy_note": "Istirahat dan minum obat",
        },
        "medicines": [
            {
                "name": "Paracetamol",
                "quantity": 10,
                "unit": "TABLET",
                "dosage_instruction": "3 x 1",
                "unit_price": 1000,
                "total_price": 10000,
            }
        ],
        "other_charges": [
            {
                "category": "CONSULTATION",
                "description": "Konsultasi dokter",
                "amount": 115000,
            }
        ],
    }

    result = evaluator.evaluate(payload, payload)

    assert result["schema_version"] == EVALUATOR_SCHEMA_VERSION
    assert result["primary_metric"] == PRIMARY_METRIC
    assert result["strict_kie_micro"]["f1"] == 1.0
    assert result["per_section"]["summary"]["f1"] == 1.0
    assert result["per_section"]["patient_info"]["f1"] == 1.0
    assert result["per_section"]["medical_context"]["f1"] == 1.0
    assert result["per_section"]["medicines"]["f1"] == 1.0
    assert result["per_section"]["other_charges"]["f1"] == 1.0
    assert result["per_field"]["summary.service_date"]["f1"] == 1.0
    assert result["field_exact_accuracy"]["accuracy"] == 1.0
    assert result["diagnostics"]["usability"]["pass_verify_only"] is True


def test_medical_claim_evaluator_clinical_text_matches_punctuation_equivalent_anamnesis() -> None:
    evaluator = MedicalClaimEvaluator()
    predicted = {
        "summary": {
            "detected_documents": ["Kwitansi"],
            "medical_facility_name": "RS Sehat Sentosa",
            "service_date": "2024-08-10",
            "transaction_date": None,
            "currency": "IDR",
            "total_amount": 125000,
        },
        "patient_info": {"name": "Budi Saputra", "date_of_birth": None, "age": None, "gender": None, "address": None},
        "medical_context": {
            "anamnesis": "Demam + Pusing + Pegal + Nyeri telan + Batuk",
            "diagnoses": [],
            "therapy_note": None,
        },
        "medicines": [],
        "other_charges": [],
        "financial_recap": {"total_medicines": 0, "total_other_charges": 125000, "grand_total": 125000},
    }
    ground_truth = {
        **predicted,
        "medical_context": {
            "anamnesis": "Demam + Pusing + Pegal (+) Nyeri telan (+) Batuk (+)",
            "diagnoses": [],
            "therapy_note": None,
        },
    }

    result = evaluator.evaluate(predicted, ground_truth)

    assert result["per_field"]["medical_context.anamnesis"]["f1"] == 1.0


def test_medical_claim_evaluator_strict_medicine_allows_obvious_ocr_near_match() -> None:
    evaluator = MedicalClaimEvaluator()
    predicted = {
        "summary": {
            "detected_documents": ["Kwitansi Klinik Sehat"],
            "medical_facility_name": "RS Sehat Sentosa",
            "service_date": "2024-08-10",
            "transaction_date": None,
            "currency": "IDR",
            "total_amount": 125000,
        },
        "patient_info": {"name": "Budi Saputra", "date_of_birth": None, "age": None, "gender": None, "address": None},
        "medical_context": {"anamnesis": None, "diagnoses": [], "therapy_note": None},
        "medicines": [
            {"name": "Lapnon", "quantity": None, "unit": None, "dosage_instruction": None, "unit_price": None, "total_price": None}
        ],
        "other_charges": [],
        "financial_recap": {"total_medicines": 10000, "total_other_charges": 115000, "grand_total": 125000},
    }
    ground_truth = {
        **predicted,
        "medicines": [
            {"name": "Lapron", "quantity": None, "unit": None, "dosage_instruction": None, "unit_price": None, "total_price": None}
        ],
    }

    result = evaluator.evaluate(predicted, ground_truth)

    assert result["entity_metrics"]["medicines"]["f1"] == 1.0
    assert result["medicine_details"]["matches"][0]["similarity"] >= 0.78


def test_medical_claim_evaluator_strict_medicine_rejects_unrelated_names_even_with_null_quantity() -> None:
    evaluator = MedicalClaimEvaluator()
    predicted = {
        "summary": {
            "detected_documents": ["Kwitansi Klinik Sehat"],
            "medical_facility_name": "RS Sehat Sentosa",
            "service_date": "2024-08-10",
            "transaction_date": None,
            "currency": "IDR",
            "total_amount": 125000,
        },
        "patient_info": {"name": "Budi Saputra", "date_of_birth": None, "age": None, "gender": None, "address": None},
        "medical_context": {"anamnesis": None, "diagnoses": [], "therapy_note": None},
        "medicines": [
            {"name": "Vitamin C", "quantity": None, "unit": None, "dosage_instruction": None, "unit_price": None, "total_price": None}
        ],
        "other_charges": [],
        "financial_recap": {"total_medicines": 0, "total_other_charges": 125000, "grand_total": 125000},
    }
    ground_truth = {
        **predicted,
        "medicines": [
            {"name": "Amoxicillin", "quantity": None, "unit": None, "dosage_instruction": None, "unit_price": None, "total_price": None}
        ],
    }

    result = evaluator.evaluate(predicted, ground_truth)

    assert result["entity_metrics"]["medicines"]["f1"] == 0.0
    assert result["medicine_details"]["unmatched_predicted"][0]["name"] == "Vitamin C"
    assert result["medicine_details"]["unmatched_ground_truth"][0]["name"] == "Amoxicillin"


def test_medical_claim_evaluator_strict_other_charge_requires_amount_and_category() -> None:
    evaluator = MedicalClaimEvaluator()
    predicted = {
        "summary": {
            "detected_documents": ["Kwitansi"],
            "medical_facility_name": "RS Sehat Sentosa",
            "service_date": "2024-08-10",
            "transaction_date": None,
            "currency": "IDR",
            "total_amount": 125000,
        },
        "patient_info": {"name": "Budi Saputra", "date_of_birth": None, "age": None, "gender": None, "address": None},
        "medical_context": {"anamnesis": None, "diagnoses": [], "therapy_note": None},
        "medicines": [],
        "other_charges": [{"category": "Administrasi", "description": "Administrasi", "amount": 120000}],
        "financial_recap": {"total_medicines": 0, "total_other_charges": 120000, "grand_total": 125000},
    }
    ground_truth = {
        **predicted,
        "other_charges": [{"category": "CONSULTATION", "description": "Konsultasi dokter", "amount": 115000}],
        "financial_recap": {"total_medicines": 0, "total_other_charges": 115000, "grand_total": 125000},
    }

    result = evaluator.evaluate(predicted, ground_truth)

    usability = result["diagnostics"]["usability"]
    assert result["per_section"]["other_charges"]["strict_f1"] == 0.0
    assert result["entity_metrics"]["other_charges"]["f1"] == 0.0
    assert "other_charges KIE F1 below relaxed verify-only threshold" in usability["advisory_fail_reasons"]
    assert usability["blocking_fail_reasons"] == []
    assert usability["critical_fail_reasons"] == []
    assert usability["pass_verify_only"] is True


def test_medical_claim_evaluator_strict_detects_financial_mismatch_as_critical() -> None:
    evaluator = MedicalClaimEvaluator()
    payload = {
        "summary": {
            "detected_documents": ["Kwitansi"],
            "medical_facility_name": "RS Sehat Sentosa",
            "service_date": "2024-08-10",
            "transaction_date": None,
            "currency": "IDR",
            "total_amount": 125000,
        },
        "patient_info": {"name": "Budi Saputra", "date_of_birth": None, "age": None, "gender": None, "address": None},
        "medical_context": {"anamnesis": None, "diagnoses": [], "therapy_note": None},
        "medicines": [],
        "other_charges": [],
        "financial_recap": {"total_medicines": 10000, "total_other_charges": 115000, "grand_total": 160000},
    }
    ground_truth = {
        **payload,
        "financial_recap": {"total_medicines": 10000, "total_other_charges": 115000, "grand_total": 125000},
    }

    result = evaluator.evaluate(payload, ground_truth)

    usability = result["diagnostics"]["usability"]
    assert any("financial_recap.grand_total outside verify-only tolerance" == reason for reason in usability["critical_fail_reasons"])
    assert usability["blocking_fail_reasons"] == usability["critical_fail_reasons"]
    assert usability["edit_burden_score"] >= evaluator.t.EDIT_COST_FINANCIAL_MISMATCH
    assert usability["pass_verify_only"] is False
    assert result["financial_exact_accuracy"]["incorrect_fields"] == ["financial_recap.grand_total"]


def test_medical_claim_evaluator_verify_only_allows_small_financial_tolerance() -> None:
    evaluator = MedicalClaimEvaluator()
    predicted = _minimal_eval_payload()
    ground_truth = {
        **predicted,
        "financial_recap": {"total_medicines": 0, "total_other_charges": 125000, "grand_total": 125000},
    }
    predicted["financial_recap"] = {"total_medicines": 0, "total_other_charges": 125000, "grand_total": 130000}

    result = evaluator.evaluate(predicted, ground_truth)

    usability = result["diagnostics"]["usability"]
    assert result["per_field"]["financial_recap.grand_total"]["strict_f1"] == 0.0
    assert result["financial_exact_accuracy"]["incorrect_fields"] == ["financial_recap.grand_total"]
    assert usability["blocking_fail_reasons"] == []
    assert usability["critical_fail_reasons"] == []
    assert usability["edit_burden_score"] == evaluator.t.EDIT_COST_PARTIAL_MATCH
    assert usability["pass_verify_only"] is True


def test_medical_claim_evaluator_verify_only_blocks_large_financial_difference() -> None:
    evaluator = MedicalClaimEvaluator()
    predicted = _minimal_eval_payload()
    ground_truth = {
        **predicted,
        "financial_recap": {"total_medicines": 0, "total_other_charges": 125000, "grand_total": 125000},
    }
    predicted["financial_recap"] = {"total_medicines": 0, "total_other_charges": 125000, "grand_total": 151000}

    result = evaluator.evaluate(predicted, ground_truth)

    usability = result["diagnostics"]["usability"]
    assert "financial_recap.grand_total outside verify-only tolerance" in usability["blocking_fail_reasons"]
    assert usability["critical_fail_reasons"] == usability["blocking_fail_reasons"]
    assert usability["pass_verify_only"] is False


def test_medical_claim_evaluator_verify_only_blocks_patient_name_mismatch() -> None:
    evaluator = MedicalClaimEvaluator()
    predicted = _minimal_eval_payload()
    ground_truth = {
        **predicted,
        "patient_info": {**predicted["patient_info"], "name": "Siti Aminah"},
    }

    result = evaluator.evaluate(predicted, ground_truth)

    usability = result["diagnostics"]["usability"]
    assert result["per_field"]["patient_info.name"]["strict_f1"] == 0.0
    assert "patient_info.name exact mismatch" in usability["blocking_fail_reasons"]
    assert usability["critical_fail_reasons"] == usability["blocking_fail_reasons"]
    assert usability["pass_verify_only"] is False


def test_medical_claim_evaluator_verify_only_relaxes_high_similarity_facility_text() -> None:
    evaluator = MedicalClaimEvaluator()
    predicted = _minimal_eval_payload()
    ground_truth = {
        **predicted,
        "summary": {**predicted["summary"], "medical_facility_name": "RS Sehat Sentosa"},
    }
    predicted["summary"] = {**predicted["summary"], "medical_facility_name": "RS Sht Sentosa"}

    result = evaluator.evaluate(predicted, ground_truth)

    usability = result["diagnostics"]["usability"]
    assert result["per_field"]["summary.medical_facility_name"]["strict_f1"] == 0.0
    assert result["relaxed_text"]["by_field"]["summary.medical_facility_name"]["average_anls"] >= 0.8
    assert usability["critical_fail_reasons"] == []
    assert usability["advisory_fail_reasons"] == []
    assert usability["pass_verify_only"] is True


def test_medical_claim_evaluator_verify_only_treats_date_mismatches_as_advisory() -> None:
    evaluator = MedicalClaimEvaluator()
    predicted = _minimal_eval_payload()
    ground_truth = {
        **predicted,
        "summary": {
            **predicted["summary"],
            "service_date": "2024-08-09",
            "transaction_date": "2024-08-09",
        },
    }

    result = evaluator.evaluate(predicted, ground_truth)

    usability = result["diagnostics"]["usability"]
    assert result["per_field"]["summary.service_date"]["strict_f1"] == 0.0
    assert result["per_field"]["summary.transaction_date"]["strict_f1"] == 0.0
    assert "summary.service_date exact mismatch" in usability["advisory_fail_reasons"]
    assert "summary.transaction_date exact mismatch" in usability["advisory_fail_reasons"]
    assert usability["blocking_fail_reasons"] == []
    assert usability["critical_fail_reasons"] == []
    assert usability["pass_verify_only"] is True


def test_medical_claim_evaluator_detected_documents_preserves_order_and_duplicates() -> None:
    evaluator = MedicalClaimEvaluator()
    predicted = {
        "summary": {
            "detected_documents": ["Kwitansi", "Resep", "Kwitansi"],
            "medical_facility_name": "RS Sehat Sentosa",
            "service_date": "2024-08-10",
            "transaction_date": None,
            "currency": "IDR",
            "total_amount": 125000,
        },
        "patient_info": {"name": "Budi Saputra", "date_of_birth": None, "age": None, "gender": None, "address": None},
        "medical_context": {"anamnesis": None, "diagnoses": [], "therapy_note": None},
        "medicines": [],
        "other_charges": [],
        "financial_recap": {"total_medicines": 0, "total_other_charges": 0, "grand_total": 125000},
    }
    ground_truth = {
        **predicted,
        "summary": {
            **predicted["summary"],
            "detected_documents": ["Kwitansi", "Kwitansi", "Resep"],
        },
    }

    result = evaluator.evaluate(predicted, ground_truth)

    assert result["per_field"]["summary.detected_documents"]["f1"] == 0.6667
    assert result["per_field"]["summary.detected_documents"]["strict_f1"] == 0.6667
    assert result["per_field"]["summary.detected_documents"]["sequence_exact"] is False
    assert result["per_field"]["summary.detected_documents"]["matched_sequence"] == ["receipt", "receipt"]
    assert result["diagnostics"]["usability"]["pass_verify_only"] is True


def test_medical_claim_evaluator_edit_burden_counts_missing_list_item_once() -> None:
    evaluator = MedicalClaimEvaluator()
    predicted = _minimal_eval_payload()
    ground_truth = {
        **predicted,
        "medicines": [
            {
                "name": "Paracetamol",
                "quantity": None,
                "unit": None,
                "dosage_instruction": None,
                "unit_price": None,
                "total_price": None,
            }
        ],
    }

    result = evaluator.evaluate(predicted, ground_truth)

    assert result["entity_metrics"]["medicines"]["fn"] == 1
    assert result["per_section"]["medicines"]["fn"] == 1.0
    assert result["diagnostics"]["usability"]["edit_burden_score"] == evaluator.t.EDIT_COST_MISSING_ITEM


def test_medical_claim_evaluator_verify_only_blocks_high_edit_burden_without_hard_failures() -> None:
    evaluator = MedicalClaimEvaluator()
    predicted = _minimal_eval_payload()
    ground_truth = {
        **predicted,
        "medicines": [
            {
                "name": f"Medicine {index}",
                "quantity": None,
                "unit": None,
                "dosage_instruction": None,
                "unit_price": None,
                "total_price": None,
            }
            for index in range(20)
        ],
    }

    result = evaluator.evaluate(predicted, ground_truth)

    usability = result["diagnostics"]["usability"]
    assert usability["edit_burden_score"] > evaluator.t.VERIFY_ONLY_MAX_EDIT_BURDEN
    assert "medicines KIE F1 below relaxed verify-only threshold" in usability["advisory_fail_reasons"]
    assert usability["blocking_fail_reasons"] == []
    assert usability["critical_fail_reasons"] == []
    assert usability["pass_verify_only"] is False


def test_medical_claim_evaluator_detected_documents_normalizes_prompt_and_internal_labels() -> None:
    evaluator = MedicalClaimEvaluator()
    predicted = {
        "summary": {
            "detected_documents": ["Kwitansi", "Formulir Rawat Jalan", "Kwitansi Apotek Sehat"],
            "medical_facility_name": "RS Sehat Sentosa",
            "service_date": "2024-08-10",
            "transaction_date": None,
            "currency": "IDR",
            "total_amount": 125000,
        },
        "patient_info": {"name": "Budi Saputra", "date_of_birth": None, "age": None, "gender": None, "address": None},
        "medical_context": {"anamnesis": None, "diagnoses": [], "therapy_note": None},
        "medicines": [],
        "other_charges": [],
        "financial_recap": {"total_medicines": 0, "total_other_charges": 0, "grand_total": 125000},
    }
    ground_truth = {
        **predicted,
        "summary": {
            **predicted["summary"],
            "detected_documents": ["receipt", "medical_form", "pharmacy_receipt"],
        },
    }

    result = evaluator.evaluate(predicted, ground_truth)

    assert result["per_field"]["summary.detected_documents"]["f1"] == 1.0
    assert result["per_field"]["summary.detected_documents"]["strict_f1"] == 1.0
    assert result["per_field"]["summary.detected_documents"]["predicted_sequence"] == [
        "receipt",
        "medical_form",
        "pharmacy_receipt",
    ]
    assert result["per_field"]["summary.detected_documents"]["ground_truth_sequence"] == [
        "receipt",
        "medical_form",
        "pharmacy_receipt",
    ]


def test_medical_claim_evaluator_strict_detected_documents_rejects_ambiguous_other() -> None:
    evaluator = MedicalClaimEvaluator()
    predicted = {
        "summary": {
            "detected_documents": ["Pembayaran"],
            "medical_facility_name": "RS Sehat Sentosa",
            "service_date": "2024-08-10",
            "transaction_date": None,
            "currency": "IDR",
            "total_amount": 125000,
        },
        "patient_info": {"name": "Budi Saputra", "date_of_birth": None, "age": None, "gender": None, "address": None},
        "medical_context": {"anamnesis": None, "diagnoses": [], "therapy_note": None},
        "medicines": [],
        "other_charges": [],
        "financial_recap": {"total_medicines": 0, "total_other_charges": 0, "grand_total": 125000},
    }
    ground_truth = {
        **predicted,
        "summary": {**predicted["summary"], "detected_documents": ["Kwitansi"]},
    }

    result = evaluator.evaluate(predicted, ground_truth)

    assert result["per_field"]["summary.detected_documents"]["f1"] == 0.0
    assert result["per_field"]["summary.detected_documents"]["strict_f1"] == 0.0


def test_medical_claim_evaluator_strict_facility_rejects_wrong_entity() -> None:
    evaluator = MedicalClaimEvaluator()
    predicted = {
        "summary": {
            "detected_documents": ["Kwitansi"],
            "medical_facility_name": "RS Petrokimia Gresik",
            "service_date": "2024-08-10",
            "transaction_date": None,
            "currency": "IDR",
            "total_amount": 125000,
        },
        "patient_info": {"name": "Budi Saputra", "date_of_birth": None, "age": None, "gender": None, "address": None},
        "medical_context": {"anamnesis": None, "diagnoses": [], "therapy_note": None},
        "medicines": [],
        "other_charges": [],
        "financial_recap": {"total_medicines": 0, "total_other_charges": 0, "grand_total": 125000},
    }
    ground_truth = {
        **predicted,
        "summary": {**predicted["summary"], "medical_facility_name": "Rumah Sakit Semen Gresik"},
    }

    result = evaluator.evaluate(predicted, ground_truth)

    assert result["per_field"]["summary.medical_facility_name"]["strict_f1"] == 0.0


def test_medical_claim_evaluator_facility_matches_known_ubaya_alias() -> None:
    evaluator = MedicalClaimEvaluator()
    predicted = {
        "summary": {
            "detected_documents": ["Kwitansi"],
            "medical_facility_name": "klinik ubaya",
            "service_date": "2024-08-10",
            "transaction_date": None,
            "currency": "IDR",
            "total_amount": 125000,
        },
        "patient_info": {"name": "Budi Saputra", "date_of_birth": None, "age": None, "gender": None, "address": None},
        "medical_context": {"anamnesis": None, "diagnoses": [], "therapy_note": None},
        "medicines": [],
        "other_charges": [],
        "financial_recap": {"total_medicines": 0, "total_other_charges": 125000, "grand_total": 125000},
    }
    ground_truth = {
        **predicted,
        "summary": {
            **predicted["summary"],
            "medical_facility_name": "Klinik Pratama Universitas Surabaya",
        },
    }

    result = evaluator.evaluate(predicted, ground_truth)

    assert result["per_field"]["summary.medical_facility_name"]["strict_f1"] == 1.0


def test_medical_claim_evaluator_detected_documents_supports_doctor_lab_and_radiology_types() -> None:
    evaluator = MedicalClaimEvaluator()
    predicted = {
        "summary": {
            "detected_documents": ["doctor_note", "lab_result", "radiology_result"],
            "medical_facility_name": "RS Sehat Sentosa",
            "service_date": "2024-08-10",
            "transaction_date": None,
            "currency": "IDR",
            "total_amount": 125000,
        },
        "patient_info": {"name": "Budi Saputra", "date_of_birth": None, "age": None, "gender": None, "address": None},
        "medical_context": {"anamnesis": None, "diagnoses": [], "therapy_note": None},
        "medicines": [],
        "other_charges": [],
        "financial_recap": {"total_medicines": 0, "total_other_charges": 0, "grand_total": 125000},
    }
    ground_truth = {
        **predicted,
        "summary": {
            **predicted["summary"],
            "detected_documents": ["doctor note", "lab result", "radiology result"],
        },
    }

    result = evaluator.evaluate(predicted, ground_truth)

    assert result["per_field"]["summary.detected_documents"]["predicted_sequence"] == [
        "doctor_note",
        "lab_result",
        "radiology_result",
    ]
    assert result["per_field"]["summary.detected_documents"]["strict_f1"] == 1.0


def test_medical_claim_evaluator_other_charge_normalizes_legacy_labels_to_prompt_contract() -> None:
    evaluator = MedicalClaimEvaluator()
    predicted = {
        "summary": {
            "detected_documents": ["Kwitansi"],
            "medical_facility_name": "RS Sehat Sentosa",
            "service_date": "2024-08-10",
            "transaction_date": None,
            "currency": "IDR",
            "total_amount": 175000,
        },
        "patient_info": {"name": "Budi Saputra", "date_of_birth": None, "age": None, "gender": None, "address": None},
        "medical_context": {"anamnesis": None, "diagnoses": [], "therapy_note": None},
        "medicines": [],
        "other_charges": [
            {"category": "Jasa Medik Dokter", "description": "Konsultasi dokter", "amount": 115000},
            {"category": "Administrasi", "description": "Administrasi", "amount": 10000},
            {"category": "Perawatan", "description": "Tindakan medis", "amount": 50000},
        ],
        "financial_recap": {"total_medicines": 0, "total_other_charges": 175000, "grand_total": 175000},
    }
    ground_truth = {
        **predicted,
        "other_charges": [
            {"category": "CONSULTATION", "description": "Konsultasi dokter", "amount": 115000},
            {"category": "ADMINISTRATION", "description": "Administrasi", "amount": 10000},
            {"category": "SERVICE", "description": "Tindakan medis", "amount": 50000},
        ],
    }

    result = evaluator.evaluate(predicted, ground_truth)

    assert result["per_section"]["other_charges"]["f1"] == 1.0
    assert result["per_section"]["other_charges"]["strict_f1"] == 1.0


def test_medical_claim_evaluator_other_charge_uses_description_fallback_only_when_category_missing() -> None:
    evaluator = MedicalClaimEvaluator()
    predicted = {
        "summary": {
            "detected_documents": ["Kwitansi"],
            "medical_facility_name": "RS Sehat Sentosa",
            "service_date": "2024-08-10",
            "transaction_date": None,
            "currency": "IDR",
            "total_amount": 115000,
        },
        "patient_info": {"name": "Budi Saputra", "date_of_birth": None, "age": None, "gender": None, "address": None},
        "medical_context": {"anamnesis": None, "diagnoses": [], "therapy_note": None},
        "medicines": [],
        "other_charges": [{"category": None, "description": "Konsultasi dokter", "amount": 115000}],
        "financial_recap": {"total_medicines": 0, "total_other_charges": 115000, "grand_total": 115000},
    }
    ground_truth = {
        **predicted,
        "other_charges": [{"category": "CONSULTATION", "description": "Konsultasi dokter", "amount": 115000}],
    }

    result = evaluator.evaluate(predicted, ground_truth)

    assert result["per_section"]["other_charges"]["strict_f1"] == 1.0


def test_medical_claim_evaluator_ignores_amount_null_duplicate_other_charge() -> None:
    evaluator = MedicalClaimEvaluator()
    predicted = {
        "summary": {
            "detected_documents": ["Kwitansi"],
            "medical_facility_name": "RS Sehat Sentosa",
            "service_date": "2024-08-10",
            "transaction_date": None,
            "currency": "IDR",
            "total_amount": 70000,
        },
        "patient_info": {"name": "Budi Saputra", "date_of_birth": None, "age": None, "gender": None, "address": None},
        "medical_context": {"anamnesis": None, "diagnoses": [], "therapy_note": None},
        "medicines": [],
        "other_charges": [
            {"category": "ADMINISTRATION", "description": "Administrasi - Test Swab Antigen Covid-19", "amount": 70000},
            {"category": "ADMINISTRATION", "description": "Administrasi", "amount": None},
        ],
        "financial_recap": {"total_medicines": 0, "total_other_charges": 70000, "grand_total": 70000},
    }
    ground_truth = {
        **predicted,
        "other_charges": [{"category": "SERVICE", "description": "Administrasi; Test Swab Antigen Covid-19", "amount": 70000}],
    }

    result = evaluator.evaluate(predicted, ground_truth)

    assert result["entity_metrics"]["other_charges"]["f1"] == 1.0
    assert result["other_charges_details"]["unmatched_predicted"] == []


def test_medical_claim_evaluator_other_charge_strict_rejects_category_mismatch_with_similar_description() -> None:
    evaluator = MedicalClaimEvaluator()
    predicted = {
        "summary": {
            "detected_documents": ["Kwitansi"],
            "medical_facility_name": "RS Sehat Sentosa",
            "service_date": "2024-08-10",
            "transaction_date": None,
            "currency": "IDR",
            "total_amount": 115000,
        },
        "patient_info": {"name": "Budi Saputra", "date_of_birth": None, "age": None, "gender": None, "address": None},
        "medical_context": {"anamnesis": None, "diagnoses": [], "therapy_note": None},
        "medicines": [],
        "other_charges": [{"category": "ADMINISTRATION", "description": "Konsultasi dokter", "amount": 115000}],
        "financial_recap": {"total_medicines": 0, "total_other_charges": 115000, "grand_total": 115000},
    }
    ground_truth = {
        **predicted,
        "other_charges": [{"category": "CONSULTATION", "description": "Konsultasi dokter", "amount": 115000}],
    }

    result = evaluator.evaluate(predicted, ground_truth)

    assert result["per_section"]["other_charges"]["f1"] == 0.0
    assert result["per_section"]["other_charges"]["strict_f1"] == 0.0


def test_guess_mime_type_uses_file_extension() -> None:
    assert guess_mime_type(Path("scan.jpg")) == "image/jpeg"


def test_register_test_user_returns_access_token() -> None:
    with TestClient(build_stub_eval_app()) as client:
        token = register_test_user(client, "eval@example.com", "Eval User")

    assert token == "token"


def test_ensure_family_candidates_adds_only_missing_candidates() -> None:
    with TestClient(build_stub_eval_app()) as client:
        token = register_test_user(client, "eval@example.com", "Budi Saputra")
        ensure_family_candidates(client, token, ["Budi Saputra", "Siti Aminah", "Siti Aminah"], "Budi Saputra")
        response = client.get("/api/family-members", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 200
    assert response.json() == [{"id": 1, "name": "Siti Aminah", "relation": "Family"}]


def test_build_aggregate_summary_includes_per_field_scores() -> None:
    summary = run_evaluation.build_aggregate_summary(
        [
            {
                "scan_source": "cache",
                "score": {
                    "strict_kie_micro": {"f1": 0.8, "precision": 0.8, "recall": 0.8, "tp": 4, "fp": 1, "fn": 1, "applicable": True},
                    "strict_kie_section_macro_f1": 0.8,
                    "per_section": {"summary": {"f1": 0.8, "precision": 0.8, "recall": 0.8, "tp": 4, "fp": 1, "fn": 1, "applicable": True}},
                    "per_field": {"summary.service_date": {"f1": 1.0, "precision": 1.0, "recall": 1.0, "tp": 1, "fp": 0, "fn": 0, "applicable": True}},
                    "field_exact_accuracy": {"correct": 1, "total": 1, "incorrect_fields": [], "accuracy": 1.0},
                    "financial_exact_accuracy": {"correct": 0, "total": 0, "incorrect_fields": [], "accuracy": 0.0},
                    "relaxed_text": {"average_anls": 1.0, "average_cer": 0.0, "count": 1},
                    "diagnostics": {"usability": {"edit_burden_score": 0.0, "pass_verify_only": True}},
                },
            },
            {
                "scan_source": "fresh_scan",
                "score": {
                    "strict_kie_micro": {"f1": 0.6, "precision": 0.6, "recall": 0.6, "tp": 3, "fp": 2, "fn": 2, "applicable": True},
                    "strict_kie_section_macro_f1": 0.6,
                    "per_section": {"summary": {"f1": 0.6, "precision": 0.6, "recall": 0.6, "tp": 3, "fp": 2, "fn": 2, "applicable": True}},
                    "per_field": {"summary.service_date": {"f1": 0.5, "precision": 0.5, "recall": 0.5, "tp": 1, "fp": 1, "fn": 1, "applicable": True}},
                    "field_exact_accuracy": {"correct": 0, "total": 1, "incorrect_fields": ["summary.service_date"], "accuracy": 0.0},
                    "financial_exact_accuracy": {"correct": 0, "total": 0, "incorrect_fields": [], "accuracy": 0.0},
                    "relaxed_text": {"average_anls": 0.5, "average_cer": 0.5, "count": 1},
                    "diagnostics": {"usability": {"edit_burden_score": 1.0, "pass_verify_only": False}},
                },
            },
        ]
    )

    assert summary["schema_version"] == EVALUATOR_SCHEMA_VERSION
    assert summary["strict_kie_micro"]["f1"] == 0.7
    assert summary["strict_kie_macro_f1"] == 0.7
    assert summary["per_section_macro_f1"]["summary"] == 0.7
    assert summary["per_field_macro_f1"]["summary.service_date"] == 0.75
    assert summary["field_exact_accuracy"]["accuracy"] == 0.5
    assert summary["relaxed_text"]["average_anls"] == 0.75
    assert summary["claims_from_cache"] == 1
    assert summary["claims_from_fresh_scan"] == 1


def test_build_aggregate_summary_reports_token_usage_from_run_records() -> None:
    base_score = {
        "strict_kie_micro": {"f1": 1.0, "precision": 1.0, "recall": 1.0, "tp": 1, "fp": 0, "fn": 0, "applicable": True},
        "strict_kie_section_macro_f1": 1.0,
        "per_section": {},
        "per_field": {},
        "field_exact_accuracy": {"correct": 0, "total": 0, "incorrect_fields": [], "accuracy": 0.0},
        "financial_exact_accuracy": {"correct": 0, "total": 0, "incorrect_fields": [], "accuracy": 0.0},
        "relaxed_text": {"average_anls": 0.0, "average_cer": 0.0, "count": 0},
        "diagnostics": {"usability": {"edit_burden_score": 0.0, "pass_verify_only": True}},
    }
    summary = run_evaluation.build_aggregate_summary(
        [
            {
                "scan_source": "fresh_scan",
                "score": base_score,
                "run_record": {
                    "scan_log": {
                        "ai_api_calls": [
                            {
                                "success": True,
                                "prompt_tokens": 100,
                                "completion_tokens": 20,
                                "total_tokens": 120,
                                "latency_ms": 900,
                            },
                            {
                                "success": True,
                                "prompt_tokens": 60,
                                "completion_tokens": 10,
                                "total_tokens": 70,
                                "latency_ms": 600,
                            },
                        ]
                    }
                },
            },
            {
                "scan_source": "cache",
                "score": base_score,
                "run_record": {
                    "scan_log": {
                        "ai_api_calls": [
                            {
                                "success": False,
                                "prompt_tokens": 40,
                                "completion_tokens": 5,
                                "total_tokens": None,
                                "latency_ms": 300,
                            }
                        ]
                    }
                },
            },
            {
                "scan_source": "cache",
                "score": base_score,
                "run_record": {"scan_log": {"ai_api_calls": []}},
            },
        ]
    )

    token_usage = summary["token_usage"]
    assert token_usage["claims_with_usage"] == 2
    assert token_usage["api_call_count"] == 3
    assert token_usage["successful_api_call_count"] == 2
    assert token_usage["failed_api_call_count"] == 1
    assert token_usage["total_input_tokens"] == 200
    assert token_usage["total_output_tokens"] == 35
    assert token_usage["total_tokens"] == 235
    assert token_usage["average_input_tokens"] == 100.0
    assert token_usage["average_output_tokens"] == 17.5
    assert token_usage["average_total_tokens"] == 117.5
    assert token_usage["average_input_tokens_per_call"] == 66.6667
    assert token_usage["average_output_tokens_per_call"] == 11.6667
    assert token_usage["average_total_tokens_per_call"] == 78.3333
    assert token_usage["average_latency_ms_per_call"] == 600.0


def test_discover_image_files_prefers_images_subfolder(tmp_path: Path) -> None:
    claim_dir = tmp_path / "claim"
    images_dir = claim_dir / "images"
    images_dir.mkdir(parents=True)
    (images_dir / "a.jpg").write_bytes(b"a")
    (claim_dir / "ignore.jpg").write_bytes(b"b")

    paths = generate_template._discover_image_files(claim_dir)

    assert paths == [images_dir / "a.jpg"]


def test_blank_like_preserves_structure_and_nulls_values() -> None:
    result = generate_template._blank_like({"a": [1, {"b": "x"}]})

    assert result == {"a": [None, {"b": None}]}


def test_generate_template_for_folder_returns_blank_expected(tmp_path: Path) -> None:
    claim_dir = tmp_path / "claim"
    images_dir = claim_dir / "images"
    images_dir.mkdir(parents=True)
    (images_dir / "page1.jpg").write_bytes(b"img")

    with TestClient(build_stub_eval_app()) as client:
        template = asyncio.run(
            generate_template.generate_template_for_folder(claim_dir, client, "token", ["Budi Saputra"])
        )

    assert template["images"] == ["images/page1.jpg"]
    assert template["patient_name_candidates"] == ["Budi Saputra"]
    assert template["expected"]["summary"]["service_date"] is None
    assert template["_predicted"]["summary"]["service_date"] == "2024-08-10"


def test_run_single_evaluation_success_with_stubbed_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    claim_dir = tmp_path / "claim_001"
    write_ground_truth(claim_dir)
    results_dir = tmp_path / "results"
    monkeypatch.setattr(run_evaluation, "create_app", lambda: build_stub_eval_app())
    monkeypatch.setattr(run_evaluation, "export_scan_log", lambda scan_log_id: {"scan_log_id": scan_log_id, "ai_model_runs": []})

    result = asyncio.run(run_evaluation.run_single_evaluation(claim_dir, results_dir=results_dir))

    assert result["claim_id"] == "claim_001"
    assert result["score"]["strict_kie_micro"]["f1"] == 1.0
    assert result["scan_source"] == "fresh_scan"
    latest = load_latest_snapshot(results_dir, "claim_001")
    assert latest is not None
    assert latest["predicted"]["summary"]["service_date"] == "2024-08-10"


def test_run_single_evaluation_uses_cache_on_second_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    claim_dir = tmp_path / "claim_001"
    write_ground_truth(claim_dir)
    results_dir = tmp_path / "results"
    monkeypatch.setattr(run_evaluation, "create_app", lambda: build_stub_eval_app())
    monkeypatch.setattr(run_evaluation, "export_scan_log", lambda scan_log_id: {"scan_log_id": scan_log_id, "ai_model_runs": []})

    first = asyncio.run(run_evaluation.run_single_evaluation(claim_dir, results_dir=results_dir))
    assert first["scan_source"] == "fresh_scan"

    def fail_create_app() -> FastAPI:
        raise RuntimeError("scan should not be called when cache is valid")

    monkeypatch.setattr(run_evaluation, "create_app", fail_create_app)
    second = asyncio.run(run_evaluation.run_single_evaluation(claim_dir, results_dir=results_dir))
    assert second["scan_source"] == "cache"


def test_run_single_evaluation_returns_error_on_scan_failure_and_saves_run_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    claim_dir = tmp_path / "claim_001"
    write_ground_truth(claim_dir)
    results_dir = tmp_path / "results"
    monkeypatch.setattr(run_evaluation, "create_app", lambda: build_stub_eval_app(scan_status=500))
    monkeypatch.setattr(run_evaluation, "_get_latest_scan_log_for_user", lambda email: None)

    result = asyncio.run(run_evaluation.run_single_evaluation(claim_dir, results_dir=results_dir))

    assert result["claim_id"] == "claim_001"
    assert "error" in result
    run_record_path = result.get("run_record_path")
    assert isinstance(run_record_path, str)
    assert Path(run_record_path).exists()


def test_run_single_evaluation_eval_only_missing_cache_errors(tmp_path: Path) -> None:
    claim_dir = tmp_path / "claim_001"
    write_ground_truth(claim_dir)
    results_dir = tmp_path / "results"

    result = asyncio.run(
        run_evaluation.run_single_evaluation(
            claim_dir,
            results_dir=results_dir,
            mode="eval-only",
        )
    )
    assert "error" in result


def test_run_evaluation_main_writes_output_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset_dir = tmp_path / "dataset"
    (dataset_dir / "claim_001").mkdir(parents=True)
    output_path = tmp_path / "results.json"

    async def fake_run_single_evaluation(**kwargs: object) -> dict[str, object]:
        folder = kwargs["folder"]
        assert isinstance(folder, Path)
        return {
            "claim_id": folder.name,
            "elapsed_seconds": 0.1,
            "scan_source": "cache",
            "predicted": {},
            "score": {
                "strict_kie_micro": {"f1": 0.9, "precision": 0.9, "recall": 0.9, "tp": 9, "fp": 1, "fn": 1, "applicable": True},
                "strict_kie_section_macro_f1": 0.9,
                "per_section": {
                    "summary": {"f1": 0.9, "precision": 0.9, "recall": 0.9, "tp": 9, "fp": 1, "fn": 1, "applicable": True}
                },
                "per_field": {"summary.service_date": {"f1": 1.0, "precision": 1.0, "recall": 1.0, "tp": 1, "fp": 0, "fn": 0, "applicable": True}},
                "field_exact_accuracy": {"correct": 1, "total": 1, "incorrect_fields": [], "accuracy": 1.0},
                "financial_exact_accuracy": {"correct": 0, "total": 0, "incorrect_fields": [], "accuracy": 0.0},
                "relaxed_text": {"average_anls": 0.0, "average_cer": 0.0, "count": 0},
                "diagnostics": {"usability": {"edit_burden_score": 0.0, "pass_verify_only": True}},
            },
        }

    monkeypatch.setattr(run_evaluation, "initialize_database", lambda: None)
    monkeypatch.setattr(run_evaluation, "run_single_evaluation", fake_run_single_evaluation)
    monkeypatch.setattr(sys, "argv", ["run_evaluation.py", "--folder", str(dataset_dir), "--output", str(output_path)])

    run_evaluation.main()

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["evaluation_schema"] == EVALUATOR_SCHEMA_VERSION
    assert payload["aggregate_summary"]["strict_kie_micro"]["f1"] == 0.9
    assert payload["aggregate_summary"]["strict_kie_macro_f1"] == 0.9
    assert payload["aggregate_summary"]["per_field_macro_f1"]["summary.service_date"] == 1.0
    assert payload["aggregate_summary"]["claims_from_cache"] == 1


def test_generate_template_main_writes_ground_truth(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    claim_dir = tmp_path / "claim_001"
    claim_dir.mkdir(parents=True)
    output_path = claim_dir / f"{claim_dir.name}.json"

    async def fake_generate_template_for_folder(
        folder: Path,
        client: TestClient,
        token: str,
        patient_name_candidates: list[str],
    ) -> dict[str, object]:
        del folder, client, token
        return {
            "images": ["images/page1.jpg"],
            "patient_name_candidates": patient_name_candidates,
            "expected": {
                "summary": {"service_date": None},
                "patient_info": {},
                "medical_context": {},
                "medicines": [],
                "other_charges": [],
            },
            "_predicted": {"summary": {"service_date": "2024-08-10"}},
        }

    monkeypatch.setattr(generate_template, "initialize_database", lambda: None)
    monkeypatch.setattr(generate_template, "generate_template_for_folder", fake_generate_template_for_folder)
    monkeypatch.setattr(generate_template, "register_test_user", lambda client, email, full_name: "token")
    monkeypatch.setattr(generate_template, "ensure_family_candidates", lambda client, token, candidates, primary_name: None)
    monkeypatch.setattr(generate_template, "create_app", lambda: FastAPI())
    monkeypatch.setattr(sys, "argv", ["generate_template.py", "--folder", str(claim_dir), "--candidates", "Budi Saputra"])

    generate_template.main()

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["patient_name_candidates"] == ["Budi Saputra"]
    assert payload["_predicted"]["summary"]["service_date"] == "2024-08-10"


def test_result_store_roundtrip(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    claim_id = "claim_001"
    image_entries = [("images/page1.jpg", "abc123")]
    fp = build_claim_fingerprint(
        ClaimFingerprintInput(
            claim_id=claim_id,
            image_entries=image_entries,
            patient_name_candidates=["Budi Saputra"],
        )
    )
    payload = fingerprint_input_payload(
        claim_id=claim_id,
        image_entries=image_entries,
        patient_name_candidates=["Budi Saputra"],
    )

    run_record = build_run_record(
        claim_id=claim_id,
        fingerprint=fp,
        fingerprint_input=payload,
        scan_log_export={"scan_log_id": 1, "ai_model_runs": []},
        predicted={"summary": {"service_date": "2024-08-10"}},
        elapsed_seconds=1.2,
        status="succeeded",
        source="fresh_scan",
    )
    run_path = append_run_record(results_dir, claim_id, run_record)
    latest = build_latest_snapshot(
        claim_id=claim_id,
        fingerprint=fp,
        fingerprint_input=payload,
        run_record_path=run_path,
        predicted={"summary": {"service_date": "2024-08-10"}},
        scan_log_id=1,
        source="fresh_scan",
    )
    save_latest_snapshot(results_dir, claim_id, latest)

    loaded_latest = load_latest_snapshot(results_dir, claim_id)
    assert loaded_latest is not None
    resolved_run = resolve_run_record_path(results_dir, claim_id, loaded_latest)
    assert resolved_run is not None
    assert resolved_run.exists()
