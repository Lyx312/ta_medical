from __future__ import annotations

# Execution and metadata constants shared by orchestration and fusion.
FUSION_VERSION = "ensemble-fusion-v3"
CALIBRATION_VERSION = "heuristic-v1"
RECONCILIATION_VERSION = "financial-reconciliation-v2"
AI_EXTRACTION_MAX_PRODUCERS = 3
AI_EXTRACTION_RUNS_PER_MODEL = 2
AI_EXTRACTION_MIN_SUCCESSFUL_RUNS = 2
AI_EXTRACTION_TEMPERATURE = 0.2

# Confidence and list-selection thresholds.
CRITICAL_FIELD_THRESHOLD = 0.80
NON_CRITICAL_FIELD_THRESHOLD = 0.70
SEMANTIC_FIELD_THRESHOLD = 0.65
HANDWRITTEN_DIVERGENCE_THRESHOLD = 0.84
MEDICINE_PRIMARY_COUNT_SOFT_CAP = 4
MEDICINE_PRIMARY_COUNT_BUFFER = 2
MEDICINE_SPLIT_COMPONENT_MIN = 2
GENDER_AMBIGUITY_MARGIN_THRESHOLD = 0.18
GENDER_NULL_SUPPORT_RATIO_THRESHOLD = 0.25
RECONCILIATION_LARGE_OVERRIDE_RATIO_THRESHOLD = 0.35
RECONCILIATION_WEAK_SOURCE_SUPPORT_THRESHOLD = 2.2
RECONCILIATION_HEAVY_DROP_COUNT_THRESHOLD = 1

# Field path categories used by normalization and confidence policies.
CRITICAL_FIELD_PATHS = {
    "summary.medical_facility_name",
    "summary.service_date",
    "summary.total_amount",
    "patient_info.name",
    "financial_recap.total_medicines",
    "financial_recap.total_other_charges",
    "financial_recap.grand_total",
}
SEMANTIC_FIELD_PATHS = {"medical_context.anamnesis", "medical_context.therapy_note"}
SEMANTIC_FIELD_SUFFIXES = (".dosage_instruction",)
NUMERIC_FIELD_PATHS = {
    "summary.total_amount",
    "patient_info.age",
    "financial_recap.total_medicines",
    "financial_recap.total_other_charges",
    "financial_recap.grand_total",
    "medicines.quantity",
    "medicines.unit_price",
    "medicines.total_price",
    "other_charges.amount",
}
DATE_FIELD_SUFFIXES = {"service_date", "transaction_date", "date_of_birth"}
FINANCIAL_FIELD_PATHS = {
    "summary.total_amount",
    "financial_recap.total_medicines",
    "financial_recap.total_other_charges",
    "financial_recap.grand_total",
}

# Domain keyword and lexicon configuration.
OTHER_CHARGE_DESCRIPTOR_STOPWORDS = {
    "biaya",
    "consult",
    "consultation",
    "fee",
    "jasa",
    "medical",
    "pembayaran",
    "pemeriksaan",
    "pengobatan",
    "penunjang",
    "perawatan",
    "procedure",
    "rawat",
    "service",
    "therapy",
    "tindakan",
    "treatment",
    "visit",
}

GENERIC_MEDICINE_KEYWORDS = ("obat", "medicine", "medication", "drug", "pharmacy", "apotek")
PACKAGE_LEVEL_MEDICINE_KEYWORDS = (
    "obat-obatan",
    "obat obatan",
    "apotek",
    "pharmacy",
    "receipt",
    "kwitansi",
    "struk",
    "pembelian obat",
)
CONSULTATION_KEYWORDS = ("consult", "doctor", "dokter", "visit", "consultation", "jasa dokter")
SERVICE_KEYWORDS = (
    "pemeriksaan",
    "pengobatan",
    "perawatan",
    "tindakan",
    "rawat",
    "service",
    "penunjang",
    "procedure",
    "treatment",
    "therapy",
)

OTHER_CHARGE_BUCKET_ORDER = ("consultation", "service", "other", "medicine")

STRUCTURED_TERM_LEXICON = {
    "diagnoses.description": (
        "Pertusis",
        "Influenza",
        "Pharyngitis",
        "Asthma",
        "Asma",
    ),
    "medicines.name": (
        "Mucopret",
        "Microcet",
        "Tritone",
        "Equal",
        "Lasal",
        "Ibramax",
        "Fibramax",
        "Prover",
    ),
}
