import json
from pathlib import Path
from typing import Any


class GroundTruthError(Exception):
    pass


def resolve_ground_truth_path(folder_path: str | Path) -> Path:
    folder = Path(folder_path)
    preferred_path = folder / f"{folder.name}.json"
    if preferred_path.exists():
        return preferred_path

    legacy_path = folder / f"{folder.name}_ground_truth.json"
    if legacy_path.exists():
        return legacy_path

    raise GroundTruthError(
        f"No {folder.name}.json found in {folder}"
        f" (legacy fallback {folder.name}_ground_truth.json also not found)"
    )


def load_ground_truth(folder_path: str | Path) -> tuple[dict[str, Any], list[Path]]:
    folder = Path(folder_path)
    gt_path = resolve_ground_truth_path(folder)

    try:
        with open(gt_path, "r", encoding="utf-8") as f:
            gt = json.load(f)
    except json.JSONDecodeError as e:
        raise GroundTruthError(f"Invalid JSON in {gt_path.name}: {e}")

    if "expected" not in gt:
        raise GroundTruthError(f"{gt_path.name} must contain 'expected' key")
    if "images" not in gt:
        raise GroundTruthError(f"{gt_path.name} must contain 'images' key")
    if not isinstance(gt.get("images"), list) or len(gt.get("images")) == 0:
        raise GroundTruthError(f"{gt_path.name} must contain a non-empty 'images' list")

    expected = gt["expected"]
    images_list = gt.get("images", [])
    patient_name_candidates = gt.get("patient_name_candidates", [])

    image_paths: list[Path] = []
    for img_rel in images_list:
        img_path = folder / img_rel
        if not img_path.exists():
            raise GroundTruthError(
                f"Image file not found: {img_rel} (resolved to {img_path}). "
                f"Check that 'images' paths in {gt_path.name} are relative to the claim folder."
            )
        image_paths.append(img_path)

    normalized = _normalize_structure(expected)

    return {
        "images": images_list,
        "patient_name_candidates": patient_name_candidates,
        "expected": normalized,
    }, image_paths


def _normalize_structure(data: dict[str, Any]) -> dict[str, Any]:
    summary = data.get("summary", {})
    patient_info = data.get("patient_info", {})
    medical_context = data.get("medical_context", {})
    medicines = data.get("medicines", [])
    other_charges = data.get("other_charges", [])

    normalized_summary = {
        "detected_documents": summary.get("detected_documents", []),
        "medical_facility_name": _str_or_none(summary.get("medical_facility_name")),
        "service_date": _str_or_none(summary.get("service_date")),
        "transaction_date": _str_or_none(summary.get("transaction_date")),
        "currency": _str_or_none(summary.get("currency")),
        "total_amount": _num_or_none(summary.get("total_amount")),
    }

    normalized_patient_info = {
        "name": _str_or_none(patient_info.get("name")),
        "date_of_birth": _str_or_none(patient_info.get("date_of_birth")),
        "age": _num_or_none(patient_info.get("age")),
        "gender": _str_or_none(patient_info.get("gender")),
        "address": _str_or_none(patient_info.get("address")),
    }

    diagnoses = medical_context.get("diagnoses", [])
    normalized_diagnoses = [
        {
            "description": _str_or_none(d.get("description")),
            "doctor_name": _str_or_none(d.get("doctor_name")),
        }
        for d in diagnoses
    ]

    normalized_medical_context = {
        "anamnesis": _str_or_none(medical_context.get("anamnesis")),
        "diagnoses": normalized_diagnoses,
        "therapy_note": _str_or_none(medical_context.get("therapy_note")),
    }

    normalized_medicines = [
        {
            "name": _str_or_none(m.get("name")),
            "quantity": _num_or_none(m.get("quantity")),
            "unit": _str_or_none(m.get("unit")),
            "dosage_instruction": _str_or_none(m.get("dosage_instruction")),
            "unit_price": _num_or_none(m.get("unit_price")),
            "total_price": _num_or_none(m.get("total_price")),
        }
        for m in medicines
    ]

    normalized_other_charges = [
        {
            "category": _str_or_none(c.get("category")),
            "description": _str_or_none(c.get("description")),
            "amount": _num_or_none(c.get("amount")),
        }
        for c in other_charges
    ]

    financial_recap = data.get("financial_recap", {})
    normalized_financial_recap = {
        "total_medicines": _num_or_none(financial_recap.get("total_medicines")),
        "total_other_charges": _num_or_none(financial_recap.get("total_other_charges")),
        "grand_total": _num_or_none(financial_recap.get("grand_total")),
    }

    return {
        "summary": normalized_summary,
        "patient_info": normalized_patient_info,
        "medical_context": normalized_medical_context,
        "medicines": normalized_medicines,
        "other_charges": normalized_other_charges,
        "financial_recap": normalized_financial_recap,
    }


def _str_or_none(v: Any) -> str | None:
    if v is None:
        return None
    return str(v).strip() if v else None


def _num_or_none(v: Any) -> float | int | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
