from __future__ import annotations
import re
from typing import Any

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
    if category in OTHER_CHARGE_CATEGORIES:
        return OTHER_CHARGE_CATEGORIES[category]["description_templates"][0]
    return category


def get_document_type_display_name(doc_type: str | None) -> str | None:
    if not doc_type:
        return None
    if doc_type in STANDARD_DOCUMENT_TYPES:
        return STANDARD_DOCUMENT_TYPES[doc_type]["display"]
    return doc_type
