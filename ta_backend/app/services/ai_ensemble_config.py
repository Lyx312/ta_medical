from __future__ import annotations

CRITICAL_FIELD_THRESHOLD = 0.80
NON_CRITICAL_FIELD_THRESHOLD = 0.70
SEMANTIC_FIELD_THRESHOLD = 0.65
HANDWRITTEN_DIVERGENCE_THRESHOLD = 0.84
MEDICINE_PRIMARY_COUNT_SOFT_CAP = 4
MEDICINE_PRIMARY_COUNT_BUFFER = 2
MEDICINE_SPLIT_COMPONENT_MIN = 2

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
