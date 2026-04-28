from dataclasses import dataclass, field


@dataclass
class EvalThresholds:
    MEDICINE_NAME_SIM_THRESHOLD: float = 0.30
    MEDICINE_QTY_PROXIMITY_WEIGHT: float = 0.30
    MEDICINE_NAME_WEIGHT: float = 0.70
    TEXT_TOKEN_F1_THRESHOLD: float = 0.50
    DIAGNOSIS_SIM_THRESHOLD: float = 0.60
    OTHER_CHARGE_SIM_THRESHOLD: float = 0.30
    STRICT_MEDICINE_SIM_THRESHOLD: float = 0.75
    STRICT_MEDICINE_QTY_THRESHOLD: float = 0.90
    STRICT_DIAGNOSIS_DESC_THRESHOLD: float = 0.85
    STRICT_DIAGNOSIS_DOCTOR_THRESHOLD: float = 0.60
    STRICT_OTHER_CHARGE_DESC_THRESHOLD: float = 0.85
    STRICT_FACILITY_TOKEN_THRESHOLD: float = 0.80
    VERIFY_ONLY_MAX_EDIT_BURDEN: float = 1.0
    EDIT_COST_CRITICAL_FIELD: float = 2.0
    EDIT_COST_MISSING_ITEM: float = 1.0
    EDIT_COST_EXTRA_ITEM: float = 1.0
    EDIT_COST_PARTIAL_MATCH: float = 0.5
    EDIT_COST_CATEGORY_FIX: float = 0.5
    EDIT_COST_FINANCIAL_MISMATCH: float = 3.0

    FIELD_WEIGHTS: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        self.FIELD_WEIGHTS = {
            "service_date": 1.0,
            "total_amount": 1.0,
            "medical_facility_name": 0.8,
            "detected_documents": 0.5,
            "patient_name": 1.5,
            "patient_gender": 1.0,
            "patient_date_of_birth": 0.8,
            "patient_age": 0.7,
            "patient_address": 0.5,
            "anamnesis": 1.5,
            "therapy_note": 1.5,
            "diagnoses": 1.5,
            "medicines": 2.0,
            "other_charges": 1.0,
        }


DEFAULT_THRESHOLDS = EvalThresholds()
