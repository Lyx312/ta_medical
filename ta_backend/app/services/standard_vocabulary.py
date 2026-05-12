from __future__ import annotations
import re
from typing import Any

CANONICAL_DOCUMENT_TYPE_DISPLAY_NAMES = {
    "receipt": "Kwitansi",
    "prescription": "Resep",
    "sales_note": "Nota Penjualan",
    "payment_receipt": "Tanda Penerimaan Pembayaran",
    "medical_form": "Formulir Rawat Jalan",
    "doctor_note": "Doctor Note",
    "lab_result": "Lab Result",
    "radiology_result": "Radiology Result",
    "pharmacy_receipt": "Pharmacy Receipt",
    "other": "Other",
}

CANONICAL_OTHER_CHARGE_DISPLAY_NAMES = {
    "CONSULTATION": "Jasa Konsultasi Dokter",
    "ADMINISTRATION": "Biaya Administrasi",
    "SERVICE": "Pelayanan Medis",
    "LABORATORY": "Pemeriksaan Laboratorium",
    "ROOM": "Kamar/Ruangan",
    "OTHER": "Lain-lain",
}

DOCUMENT_TYPE_KEYWORDS = {
    "RECEIPT": ["kwitansi", "struk", "receipt", "pembayaran", "penjualan"],
    "PRESCRIPTION": ["resep", "salinan resep", "resep obat", "prescription"],
    "SALES_NOTE": ["nota penjualan", "nota", "faktur", "sales note", "nota penjualan resep"],
    "PAYMENT_RECEIPT": ["tanda terima", "penerimaan", "pembayaran", "payment"],
    "MEDICAL_RECORD": ["formulir rawat", "formulir", "medical record", "medical form", "formulir rawat jalan"],
    "PHARMACY_RECEIPT": ["apotek", "pharmacy", "apotik", "farmasi"],
}

STRICT_DOCUMENT_TYPE_KEYWORDS = {
    "RECEIPT": ["kwitansi", "struk", "receipt"],
    "PRESCRIPTION": ["salinan resep", "resep obat", "prescription", "resep"],
    "SALES_NOTE": ["nota penjualan resep", "nota penjualan", "sales note", "faktur"],
    "PAYMENT_RECEIPT": ["tanda terima", "tanda penerimaan pembayaran", "payment receipt"],
    "MEDICAL_RECORD": ["formulir rawat jalan", "medical form", "medical record", "formulir rawat"],
    "PHARMACY_RECEIPT": ["kwitansi apotek", "struk apotek", "pharmacy receipt", "receipt apotek", "apotik"],
}

STANDARD_DOCUMENT_TYPES = {
    doc_type: {"keywords": keywords, "display": doc_type.capitalize()}
    for doc_type, keywords in DOCUMENT_TYPE_KEYWORDS.items()
}

FACILITY_TYPE_ABBREVIATIONS = {
    "rs ": "Rumah Sakit ",
    "rsu ": "Rumah Sakit Umum ",
    "rsud ": "Rumah Sakit Umum Daerah ",
    "klinik ": "Klinik ",
    "apotek ": "Apotek ",
    "praktek ": "Praktek ",
    "dokter ": "Dokter ",
}

OWNERSHIP_PREFIXES = {
    "umum": "Umum",
    "um": "Umum",
    "daerah": "Daerah",
    "dr": "Daerah",
    "swasta": "Swasta",
    "sw": "Swasta",
    "negri": "Negeri",
    "n": "Negeri",
    "yogyakarta": "Yogyakarta",
    "jogja": "Yogyakarta",
    "jogja": "Yogyakarta",
}

OTHER_CHARGE_CATEGORIES = {
    "CONSULTATION": {
        "keywords": ["konsultasi", "consult", "jasa dokter", "visit", "dokter", "consultation", "fee dokter"],
        "description_templates": ["Jasa Konsultasi Dokter", "Biaya Konsultasi"],
    },
    "ADMINISTRATION": {
        "keywords": ["administrasi", "admin", "administrasi umum", "administrasi"],
        "description_templates": ["Biaya Administrasi", "Administrasi"],
    },
    "SERVICE": {
        "keywords": ["pelayanan", "service", "tindakan", "procedure", "treatment", "medis", "jasa medis", "pelayanan medis", "med", "meds"],
        "description_templates": ["Pelayanan Medis", "Biaya Pelayanan"],
    },
    "MEDICINE_ADMINISTRATION": {
        "keywords": ["injek", "injeksi", "infuse", "suntik", "pemberian obat", " suntik"],
        "description_templates": ["Pemberian Obat", "Biaya Injeksi"],
    },
    "ROOM": {
        "keywords": ["kamar", "ruangan", "room", "akomodasi", "rawat inap", "inap"],
        "description_templates": ["Kamar/Ruangan", "Akomodasi"],
    },
    "LABORATORY": {
        "keywords": ["lab", "laboratorium", "pemeriksaan", "test", "pemeriksaan lab"],
        "description_templates": ["Pemeriksaan Laboratorium", "Biaya Lab"],
    },
    "EQUIPMENT": {
        "keywords": ["alat", "device", "perangkat", "alat kesehatan"],
        "description_templates": ["Alat Medis", "Biaya Alat"],
    },
    "OTHER": {
        "keywords": ["lain", "other", "diverse", "lainnya", "berbagai"],
        "description_templates": ["Lain-lain", "Biaya Lain"],
    },
}

STRICT_OTHER_CHARGE_CATEGORIES = {
    "CONSULTATION": ["konsultasi", "consultation", "jasa dokter"],
    "ADMINISTRATION": ["administrasi", "admin"],
    "SERVICE": ["tindakan", "pelayanan medis", "jasa medis", "procedure", "treatment"],
    "MEDICINE_ADMINISTRATION": ["injeksi", "suntik", "pemberian obat", "infuse"],
    "ROOM": ["rawat inap", "kamar", "ruangan", "akomodasi"],
    "LABORATORY": ["laboratorium", "pemeriksaan lab", "lab test"],
    "EQUIPMENT": ["alat medis", "alat kesehatan"],
}


def _normalize_lower_text(value: str | None) -> str:
    return value.strip().lower() if value else ""


def _normalize_spaced_text(value: str | None) -> str:
    return re.sub(r"\s+", " ", _normalize_lower_text(value)).strip()


def _match_keyword_groups(value: str, mapping: dict[str, list[str]]) -> str | None:
    hits: list[str] = []
    for key, keywords in mapping.items():
        if any(keyword in value for keyword in keywords):
            hits.append(key)
    if len(hits) == 1:
        return hits[0]
    return None


def normalize_to_standard_document_type(value: str | None) -> str | None:
    if not value:
        return None
    normalized = _normalize_lower_text(value)
    for doc_type, config in STANDARD_DOCUMENT_TYPES.items():
        for keyword in config["keywords"]:
            if keyword in normalized:
                return doc_type
    return "OTHER"


def normalize_to_standard_document_type_strict(value: str | None) -> str | None:
    if not value:
        return None
    return _match_keyword_groups(_normalize_lower_text(value), STRICT_DOCUMENT_TYPE_KEYWORDS)


def normalize_to_evaluation_document_type(value: str | None) -> str | None:
    if not value:
        return None

    normalized = _normalize_spaced_text(value)
    compact = normalized.replace("-", "_").replace(" ", "_")
    if compact in CANONICAL_DOCUMENT_TYPE_DISPLAY_NAMES:
        return compact

    alias_map = {
        "salesnote": "sales_note",
        "paymentreceipt": "payment_receipt",
        "medicalrecord": "medical_form",
        "medicalform": "medical_form",
        "doctornote": "doctor_note",
        "labresult": "lab_result",
        "radiologyresult": "radiology_result",
        "pharmacyreceipt": "pharmacy_receipt",
    }
    if compact in alias_map:
        return alias_map[compact]

    if any(keyword in normalized for keyword in ("apotek", "apotik", "pharmacy", "farmasi")) and any(
        keyword in normalized for keyword in ("kwitansi", "struk", "receipt")
    ):
        return "pharmacy_receipt"
    if any(keyword in normalized for keyword in ("tanda penerimaan pembayaran", "payment receipt", "tanda terima pembayaran")):
        return "payment_receipt"
    if any(keyword in normalized for keyword in ("nota penjualan resep", "nota penjualan", "sales note", "faktur")):
        return "sales_note"
    if any(keyword in normalized for keyword in ("salinan resep", "resep obat", "prescription", "resep")):
        return "prescription"
    if any(keyword in normalized for keyword in ("formulir rawat jalan", "medical form", "medical record", "formulir rawat")):
        return "medical_form"
    if any(keyword in normalized for keyword in ("doctor note", "catatan dokter", "surat dokter")):
        return "doctor_note"
    if any(keyword in normalized for keyword in ("lab result", "hasil lab", "hasil laboratorium", "laboratorium")):
        return "lab_result"
    if any(keyword in normalized for keyword in ("radiology result", "hasil radiologi", "radiologi", "x ray", "xray", "rontgen")):
        return "radiology_result"
    if any(keyword in normalized for keyword in ("kwitansi", "struk", "receipt")):
        return "receipt"
    return "other"


def normalize_facility_name(value: str | None) -> str | None:
    if not value:
        return None
    normalized = value.strip()
    normalized_lower = normalized.lower()
    for abbrev, full_form in FACILITY_TYPE_ABBREVIATIONS.items():
        if normalized_lower.startswith(abbrev.strip().lower()):
            normalized = full_form + normalized[len(abbrev):]
            break
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized.lower()


def map_category_to_standard(value: str | None) -> str | None:
    if not value:
        return None
    normalized = _normalize_lower_text(value)
    for cat_key, config in OTHER_CHARGE_CATEGORIES.items():
        for keyword in config["keywords"]:
            if keyword in normalized:
                return cat_key
    return "OTHER"


def map_category_to_standard_strict(value: str | None) -> str | None:
    if not value:
        return None
    return _match_keyword_groups(_normalize_lower_text(value), STRICT_OTHER_CHARGE_CATEGORIES)


def normalize_to_evaluation_other_charge_category(value: str | None) -> str | None:
    if not value:
        return None

    normalized = _normalize_spaced_text(value)
    compact = normalized.replace("-", "_").replace(" ", "_")
    if compact in CANONICAL_OTHER_CHARGE_DISPLAY_NAMES:
        return compact
    if compact == "medicine_administration":
        return "SERVICE"
    if compact == "equipment":
        return "OTHER"

    if any(keyword in normalized for keyword in ("konsultasi", "consultation", "consult", "jasa dokter", "fee dokter", "jasa medik dokter", "visit dokter")):
        return "CONSULTATION"
    if any(keyword in normalized for keyword in ("administrasi", "admin")):
        return "ADMINISTRATION"
    if any(keyword in normalized for keyword in ("laboratorium", "lab test", "pemeriksaan lab")):
        return "LABORATORY"
    if any(keyword in normalized for keyword in ("rawat inap", "kamar", "ruangan", "akomodasi")):
        return "ROOM"
    if any(keyword in normalized for keyword in ("pelayanan", "tindakan", "procedure", "treatment", "perawatan", "medical service", "jasa medis", "pelayanan medis", "medis")):
        return "SERVICE"
    if any(keyword in normalized for keyword in ("lain", "lainnya", "other", "misc")):
        return "OTHER"
    return None


def map_description_to_category(description: str | None) -> str | None:
    if not description:
        return None
    return map_category_to_standard(description)


def map_description_to_category_strict(description: str | None) -> str | None:
    if not description:
        return None
    return map_category_to_standard_strict(description)


def get_category_display_name(category: str | None) -> str | None:
    if not category:
        return None
    if category in CANONICAL_OTHER_CHARGE_DISPLAY_NAMES:
        return CANONICAL_OTHER_CHARGE_DISPLAY_NAMES[category]
    if category in OTHER_CHARGE_CATEGORIES:
        return OTHER_CHARGE_CATEGORIES[category]["description_templates"][0]
    return category


def get_document_type_display_name(doc_type: str | None) -> str | None:
    if not doc_type:
        return None
    if doc_type in CANONICAL_DOCUMENT_TYPE_DISPLAY_NAMES:
        return CANONICAL_DOCUMENT_TYPE_DISPLAY_NAMES[doc_type]
    if doc_type in STANDARD_DOCUMENT_TYPES:
        return STANDARD_DOCUMENT_TYPES[doc_type]["display"]
    return doc_type
