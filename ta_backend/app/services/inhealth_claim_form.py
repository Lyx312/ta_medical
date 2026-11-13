from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from io import BytesIO
from pathlib import Path

TEMPLATE_PATH = Path(__file__).resolve().parents[1] / "templates" / "claim_forms" / "formulir_klaim_inhealth-1.pdf"
SIGNATURE_DATE_MONTH_RIGHT_X = 165
SIGNATURE_YEAR_SUFFIX_X = 178
SIGNATURE_DATE_Y = 204
SIGNATURE_NAME_CENTER_X = 134
SIGNATURE_NAME_Y = 157
CHECKMARK_WIDTH = 10
CHECKMARK_HEIGHT = 8
CHECKMARK_LINE_WIDTH = 1.4


@dataclass(frozen=True)
class InHealthClaimFormData:
    employee_name: str
    nip: str
    company_name: str
    patient_name: str
    patient_relation: str
    patient_card_number: str
    patient_phone_number: str
    patient_mobile_number: str
    patient_email: str
    bank_account_number: str
    bank_account_holder_name: str
    bank_name: str
    claim_amount: float
    receipt_date: date
    export_place: str
    export_date: date
    signature_name: str


def build_inhealth_claim_form_pdf(
    data: InHealthClaimFormData,
    *,
    template_path: Path = TEMPLATE_PATH,
) -> bytes:
    if not template_path.exists():
        raise FileNotFoundError(f"InHealth claim form template not found: {template_path}")

    try:
        from pypdf import PdfReader, PdfWriter
        from reportlab.pdfgen import canvas
    except ImportError as exc:
        raise RuntimeError("PDF export dependencies are not installed. Install pypdf and reportlab.") from exc

    reader = PdfReader(str(template_path))
    if not reader.pages:
        raise ValueError("InHealth claim form template has no pages.")

    first_page = reader.pages[0]
    width = float(first_page.mediabox.width)
    height = float(first_page.mediabox.height)

    overlay_buffer = BytesIO()
    overlay = canvas.Canvas(overlay_buffer, pagesize=(width, height))
    overlay.setFillColorRGB(0, 0, 0)
    overlay.setFont("Helvetica", 9)

    def draw_text(x: float, y: float, value: object, *, font_size: int = 9, max_chars: int | None = None) -> None:
        text = _truncate_text(value, max_chars)
        overlay.setFont("Helvetica", font_size)
        overlay.drawString(x, y, text)

    def draw_right_text(
        x: float,
        y: float,
        value: object,
        *,
        font_size: int = 9,
        max_chars: int | None = None,
    ) -> None:
        text = _truncate_text(value, max_chars)
        overlay.setFont("Helvetica", font_size)
        overlay.drawRightString(x, y, text)

    def draw_center_text(
        x: float,
        y: float,
        value: object,
        *,
        font_size: int = 9,
        max_chars: int | None = None,
    ) -> None:
        text = _truncate_text(value, max_chars)
        overlay.setFont("Helvetica", font_size)
        overlay.drawCentredString(x, y, text)

    def draw_check(x: float, y: float) -> None:
        overlay.saveState()
        overlay.setStrokeColorRGB(0, 0, 0)
        overlay.setLineWidth(CHECKMARK_LINE_WIDTH)
        path = overlay.beginPath()
        path.moveTo(x, y + CHECKMARK_HEIGHT * 0.45)
        path.lineTo(x + CHECKMARK_WIDTH * 0.35, y)
        path.lineTo(x + CHECKMARK_WIDTH, y + CHECKMARK_HEIGHT)
        overlay.drawPath(path, stroke=1, fill=0)
        overlay.restoreState()

    draw_text(170, 708, data.employee_name, max_chars=46)
    draw_text(390, 708, data.nip, max_chars=20)
    draw_text(175, 690, data.company_name, max_chars=58)
    draw_text(155, 673, data.patient_name, max_chars=58)
    relation_checks = {
        "self": (128, 654),
        "husband": (198, 654),
        "wife": (271, 654),
        "child": (344, 654),
    }
    if data.patient_relation in relation_checks:
        draw_check(*relation_checks[data.patient_relation]) 
    draw_text(135, 639, data.patient_card_number, max_chars=64)
    draw_text(147, 622, data.patient_phone_number, max_chars=20)
    draw_text(245, 622, data.patient_mobile_number, max_chars=20)
    draw_text(365, 622, data.patient_email, max_chars=28)
    draw_text(154, 605, data.bank_account_number, max_chars=50)
    draw_text(145, 588, data.bank_account_holder_name, max_chars=50)
    draw_text(120, 570, data.bank_name, max_chars=50)
    draw_text(171, 552, _format_idr(data.claim_amount), max_chars=38)
    draw_text(170, 535, _format_date(data.receipt_date), max_chars=30)

    for y in (475, 450, 415, 377, 339):
        draw_check(388, y)

    draw_right_text(
        SIGNATURE_DATE_MONTH_RIGHT_X,
        SIGNATURE_DATE_Y,
        _format_place_day_month(data.export_place, data.export_date),
        max_chars=32,
    )
    draw_center_text(136, 245, _format_date(data.export_date), max_chars=50)
    draw_text(SIGNATURE_YEAR_SUFFIX_X, SIGNATURE_DATE_Y, _format_year_suffix(data.export_date), max_chars=2)
    draw_center_text(SIGNATURE_NAME_CENTER_X, SIGNATURE_NAME_Y, data.signature_name, max_chars=38)

    overlay.save()
    overlay_buffer.seek(0)

    overlay_reader = PdfReader(overlay_buffer)
    first_page.merge_page(overlay_reader.pages[0])

    writer = PdfWriter()
    writer.add_page(first_page)
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


def _format_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _truncate_text(value: object, max_chars: int | None = None) -> str:
    text = _format_text(value)
    if max_chars is not None and len(text) > max_chars:
        return text[:max_chars]
    return text


def _format_idr(value: float) -> str:
    return f"{value:,.0f}".replace(",", ".")


def _format_date(value: date) -> str:
    return value.strftime("%d-%m-%Y")


def _format_long_date(value: date) -> str:
    return f"{value.day} {_month_name(value)} {value.year}"


def _format_place_day_month(place: str, value: date) -> str:
    formatted_place = _format_text(place)
    day_month = f"{value.day} {_month_name(value)}"
    if not formatted_place:
        return day_month
    return f"{formatted_place}, {day_month}"


def _format_year_suffix(value: date) -> str:
    return f"{value.year % 100:02d}"


def _month_name(value: date) -> str:
    month_names = [
        "Januari",
        "Februari",
        "Maret",
        "April",
        "Mei",
        "Juni",
        "Juli",
        "Agustus",
        "September",
        "Oktober",
        "November",
        "Desember",
    ]
    return month_names[value.month - 1]
