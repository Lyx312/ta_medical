import json
from pathlib import Path
from typing import Generator

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.api.routes import ai as ai_route
from app.api.routes import claims as claims_route
from app.api.routes.ai import processor as ai_processor
from app.core.config import get_settings
from app.db.session import clear_database_caches, get_session_factory
from app.main import create_app
from app.models.claim import Claim
from app.models.claim_medicine_knowledge import ClaimMedicineKnowledge
from app.schemas.ai import AIMedicalScanResponse, DocumentValidationResponse
from app.services.ai_service import DocumentImageInput
from app.services import claim_medicine_relevance as medicine_relevance_service
from app.services.image_preprocessing import ProcessedDocumentImage

SCAN_RESULT = {
    "summary": {
        "detected_documents": ["HOSPITAL_INVOICE"],
        "medical_facility_name": "RS Sehat Sentosa",
        "service_date": "2024-08-10",
        "transaction_date": "2024-08-10",
        "currency": "IDR",
        "total_amount": 125000,
    },
    "patient_info": {
        "name": "Claim User",
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
    "medicines": [],
    "other_charges": [],
    "financial_recap": {
        "total_medicines": 0,
        "total_other_charges": 125000,
        "grand_total": 125000,
    },
}


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient, None, None]:
    database_path = tmp_path / "claims-test.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{database_path.as_posix()}")
    monkeypatch.setenv("APP_ENV", "test")
    get_settings.cache_clear()
    clear_database_caches()

    app = create_app()
    with TestClient(app) as test_client:
        yield test_client

    clear_database_caches()
    get_settings.cache_clear()


def register_and_login(client: TestClient, email: str, *, full_name: str = "Claim User") -> str:
    register = client.post(
        "/api/auth/register",
        json={
            "full_name": full_name,
            "email": email,
            "password": "password123",
            "confirm_password": "password123",
        },
    )
    assert register.status_code == 201

    login = client.post(
        "/api/auth/login",
        json={"email": email, "password": "password123"},
    )
    assert login.status_code == 200
    return login.json()["access_token"]


def build_valid_validation_response(filename: str) -> DocumentValidationResponse:
    return DocumentValidationResponse.model_validate(
        {
            "packet_patient_name": "Claim User",
            "patient_name_match_status": "matched",
            "patient_name_match_reason": None,
            "review_required": False,
            "review_reasons": [],
            "images": [
                {
                    "image_index": 0,
                    "filename": filename,
                    "document_count_detected": 1,
                    "documents": [
                        {
                            "document_index": 0,
                            "document_type": "receipt",
                            "is_medical_document": True,
                            "readability_ok": True,
                            "facility_identity_visible": True,
                            "stamp_detected": True,
                            "signature_detected": False,
                            "decision": "valid",
                            "reasons": ["Visible facility stamp detected."],
                        }
                    ],
                }
            ]
        }
    )


def get_all_claims() -> list[Claim]:
    session_factory = get_session_factory()
    with session_factory() as db:
        return list(db.execute(select(Claim).order_by(Claim.id.asc())).scalars())


def get_all_knowledge_rows() -> list[ClaimMedicineKnowledge]:
    session_factory = get_session_factory()
    with session_factory() as db:
        return list(
            db.execute(
                select(ClaimMedicineKnowledge).order_by(ClaimMedicineKnowledge.id.asc())
            ).scalars()
        )


def test_create_and_list_claims_include_nama_pasien_and_persist_after_restart(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = register_and_login(client, "claims@example.com")

    def fake_preprocess_document_image(filename: str, content: bytes) -> ProcessedDocumentImage:
        assert filename == "receipt.png"
        assert content == b"fake-image"
        return ProcessedDocumentImage(
            filename="receipt.jpg",
            mime_type="image/jpeg",
            content=b"processed-image",
        )

    async def fake_validation_process(images: list[DocumentImageInput], **_: object) -> DocumentValidationResponse:
        assert len(images) == 1
        assert images[0].filename == "receipt.jpg"
        return build_valid_validation_response(images[0].filename)

    monkeypatch.setattr(claims_route, "preprocess_document_image", fake_preprocess_document_image)
    monkeypatch.setattr(claims_route.validation_processor, "process", fake_validation_process)

    create_response = client.post(
        "/api/claims",
        headers={"Authorization": f"Bearer {token}"},
        data={
            "nama_pasien": "Budi Santoso",
            "tanggal_pelayanan": "31-03-2026",
            "fasilitas_kesehatan": "RS Sehat Selalu",
            "jenis_manfaat": "rawat_jalan",
            "nominal_klaim": "150000",
            "keterangan_penyakit": "Demam",
            "kronologi": "Pemeriksaan rawat jalan",
        },
        files=[("files", ("receipt.png", b"fake-image", "image/png"))],
    )

    assert create_response.status_code == 201
    created_payload = create_response.json()
    assert created_payload["nama_pasien"] == "Budi Santoso"
    assert created_payload["tanggal_pelayanan"] == "31-03-2026"

    list_response = client.get(
        "/api/claims",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert list_response.status_code == 200
    claims = list_response.json()
    assert len(claims) == 1
    assert claims[0]["nama_pasien"] == "Budi Santoso"
    assert claims[0]["tanggal_pelayanan"] == "31-03-2026"
    assert "source_scan_extraction_payload" not in claims[0]

    restarted_app = create_app()
    with TestClient(restarted_app) as restarted_client:
        restarted_response = restarted_client.get(
            "/api/claims",
            headers={"Authorization": f"Bearer {token}"},
        )

    assert restarted_response.status_code == 200
    restarted_claims = restarted_response.json()
    assert len(restarted_claims) == 1
    assert restarted_claims[0]["nama_pasien"] == "Budi Santoso"


def test_create_claim_rejects_non_dmy_date_format(client: TestClient) -> None:
    token = register_and_login(client, "claims-invalid-date@example.com")

    response = client.post(
        "/api/claims",
        headers={"Authorization": f"Bearer {token}"},
        data={
            "nama_pasien": "Budi Santoso",
            "tanggal_pelayanan": "2026-03-31",
            "fasilitas_kesehatan": "RS Sehat Selalu",
            "jenis_manfaat": "rawat_jalan",
            "nominal_klaim": "150000",
            "keterangan_penyakit": "Demam",
            "kronologi": "Pemeriksaan rawat jalan",
        },
    )

    assert response.status_code == 422
    payload = response.json()
    assert payload["message"] == "Input validation failed."
    assert payload["errors"] == [
        {"field": "non_field", "message": "Invalid tanggal_pelayanan. Expected format: dd-mm-yyyy"}
    ]


def test_create_claim_allows_submit_but_flags_manual_review_for_non_relevant_medicines(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = register_and_login(client, "claims-medicine-review@example.com")

    def fake_preprocess_document_image(filename: str, content: bytes) -> ProcessedDocumentImage:
        assert filename == "receipt.png"
        assert content == b"fake-image"
        return ProcessedDocumentImage(
            filename="receipt.jpg",
            mime_type="image/jpeg",
            content=b"processed-image",
        )

    async def fake_validation_process(images: list[DocumentImageInput], **_: object) -> DocumentValidationResponse:
        assert len(images) == 1
        return build_valid_validation_response(images[0].filename)

    monkeypatch.setattr(claims_route, "preprocess_document_image", fake_preprocess_document_image)
    monkeypatch.setattr(claims_route.validation_processor, "process", fake_validation_process)

    response = client.post(
        "/api/claims",
        headers={"Authorization": f"Bearer {token}"},
        data={
            "nama_pasien": "Budi Santoso",
            "tanggal_pelayanan": "31-03-2026",
            "fasilitas_kesehatan": "RS Sehat Selalu",
            "jenis_manfaat": "rawat_jalan",
            "nominal_klaim": "150000",
            "keterangan_penyakit": "Demam",
            "kronologi": "Pemeriksaan rawat jalan",
            "medicines_payload": json.dumps(
                [
                    {
                        "name": "Engine Cleaner",
                        "quantity": 1,
                        "unit": "botol",
                        "dosage_instruction": "Bukan obat",
                        "source": "user_input",
                    }
                ]
            ),
        },
        files=[("files", ("receipt.png", b"fake-image", "image/png"))],
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["requires_additional_review"] is True
    assert payload["review_note"]
    assert len(payload["medicines"]) == 1
    assert payload["medicines"][0]["evaluation"]["relevance"] == "not_relevant"
    knowledge_rows = get_all_knowledge_rows()
    assert knowledge_rows == []


def test_create_claim_exposes_linked_scan_extraction_payload(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = register_and_login(client, "claims-extraction-payload@example.com")

    def fake_preprocess_document_image(filename: str, content: bytes) -> ProcessedDocumentImage:
        assert filename == "receipt.png"
        assert content in {b"fake-image", b"processed-image"}
        return ProcessedDocumentImage(
            filename="receipt.jpg",
            mime_type="image/jpeg",
            content=b"processed-image",
        )

    async def fake_validation_process(images: list[DocumentImageInput], **_: object) -> DocumentValidationResponse:
        assert len(images) == 1
        return build_valid_validation_response(images[0].filename)

    async def fake_scan_process(
        images: list[DocumentImageInput],
        patient_name_candidates: list[str] | None = None,
        **_: object,
    ) -> AIMedicalScanResponse:
        assert patient_name_candidates == ["Claim User"]
        return AIMedicalScanResponse.model_validate(SCAN_RESULT)

    monkeypatch.setattr(ai_route, "preprocess_document_image", fake_preprocess_document_image)
    monkeypatch.setattr(ai_route.validation_processor, "process", fake_validation_process)
    monkeypatch.setattr(ai_processor, "process", fake_scan_process)
    monkeypatch.setattr(claims_route, "preprocess_document_image", fake_preprocess_document_image)
    monkeypatch.setattr(claims_route.validation_processor, "process", fake_validation_process)

    scan_response = client.post(
        "/api/ai/scan-documents",
        headers={"Authorization": f"Bearer {token}"},
        files=[("files", ("receipt.png", b"fake-image", "image/png"))],
    )
    assert scan_response.status_code == 200
    scan_log_id = scan_response.json()["scan_log_id"]

    claim_response = client.post(
        "/api/claims",
        headers={"Authorization": f"Bearer {token}"},
        data={
            "nama_pasien": "Claim User",
            "tanggal_pelayanan": "31-03-2026",
            "fasilitas_kesehatan": "RS Sehat Selalu",
            "jenis_manfaat": "rawat_jalan",
            "nominal_klaim": "150000",
            "keterangan_penyakit": "Demam",
            "kronologi": "Pemeriksaan rawat jalan",
            "source_scan_log_id": str(scan_log_id),
        },
        files=[("files", ("receipt.png", b"fake-image", "image/png"))],
    )
    assert claim_response.status_code == 201
    payload = claim_response.json()
    assert payload["source_scan_log_id"] == scan_log_id
    assert payload["source_scan_extraction_payload"] == SCAN_RESULT

    list_response = client.get(
        "/api/claims",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert list_response.status_code == 200
    listed = list_response.json()
    assert len(listed) == 1
    assert "source_scan_extraction_payload" not in listed[0]

    detail_response = client.get(
        f"/api/claims/{payload['id']}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert detail_response.status_code == 200
    assert detail_response.json()["source_scan_extraction_payload"] == SCAN_RESULT


def test_create_claim_uses_balanced_history_threshold_before_llm(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = register_and_login(client, "claims-history-threshold@example.com")

    def fake_preprocess_document_image(filename: str, content: bytes) -> ProcessedDocumentImage:
        assert filename == "receipt.png"
        assert content == b"fake-image"
        return ProcessedDocumentImage(
            filename="receipt.jpg",
            mime_type="image/jpeg",
            content=b"processed-image",
        )

    async def fake_validation_process(images: list[DocumentImageInput], **_: object) -> DocumentValidationResponse:
        assert len(images) == 1
        return build_valid_validation_response(images[0].filename)

    llm_call_count = {"count": 0}

    async def fake_llm_judge(*args: object, **kwargs: object):
        llm_call_count["count"] += 1
        raise RuntimeError("Simulated LLM unavailable during seeding.")

    monkeypatch.setattr(claims_route, "preprocess_document_image", fake_preprocess_document_image)
    monkeypatch.setattr(claims_route.validation_processor, "process", fake_validation_process)
    monkeypatch.setattr(medicine_relevance_service.MedicineRelevanceLLMJudge, "judge", fake_llm_judge)

    first_response = client.post(
        "/api/claims",
        headers={"Authorization": f"Bearer {token}"},
        data={
            "nama_pasien": "Budi Santoso",
            "tanggal_pelayanan": "31-03-2026",
            "fasilitas_kesehatan": "RS Sehat Selalu",
            "jenis_manfaat": "rawat_jalan",
            "nominal_klaim": "150000",
            "keterangan_penyakit": "Demam influenza",
            "kronologi": "Batuk pilek demam 2 hari",
            "medicines_payload": json.dumps(
                [
                    {
                        "name": "Paracetamol",
                        "source": "user_input",
                    }
                ]
            ),
        },
        files=[("files", ("receipt.png", b"fake-image", "image/png"))],
    )
    assert first_response.status_code == 201

    second_response = client.post(
        "/api/claims",
        headers={"Authorization": f"Bearer {token}"},
        data={
            "nama_pasien": "Budi Santoso",
            "tanggal_pelayanan": "01-04-2026",
            "fasilitas_kesehatan": "RS Sehat Selalu",
            "jenis_manfaat": "rawat_jalan",
            "nominal_klaim": "150000",
            "keterangan_penyakit": "Demam influenza",
            "kronologi": "Batuk pilek demam sejak kemarin",
            "medicines_payload": json.dumps(
                [
                    {
                        "name": "Paracetamol",
                        "source": "user_input",
                    }
                ]
            ),
        },
        files=[("files", ("receipt.png", b"fake-image", "image/png"))],
    )

    assert second_response.status_code == 201
    third_response = client.post(
        "/api/claims",
        headers={"Authorization": f"Bearer {token}"},
        data={
            "nama_pasien": "Budi Santoso",
            "tanggal_pelayanan": "02-04-2026",
            "fasilitas_kesehatan": "RS Sehat Selalu",
            "jenis_manfaat": "rawat_jalan",
            "nominal_klaim": "150000",
            "keterangan_penyakit": "Demam influenza",
            "kronologi": "Demam batuk pilek berlanjut",
            "medicines_payload": json.dumps(
                [
                    {
                        "name": "Paracetamol",
                        "source": "user_input",
                    }
                ]
            ),
        },
        files=[("files", ("receipt.png", b"fake-image", "image/png"))],
    )

    assert third_response.status_code == 201
    payload = third_response.json()
    assert payload["medicines"][0]["evaluation"]["decision_source"] == "history_strong"
    assert llm_call_count["count"] == 2

    knowledge_rows = get_all_knowledge_rows()
    assert len(knowledge_rows) == 3
    assert all(row.medicine_name_normalized == "paracetamol" for row in knowledge_rows)
    assert sum(row.support_count for row in knowledge_rows) == 3


def test_create_claim_falls_back_to_heuristic_when_llm_fails(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = register_and_login(client, "claims-llm-fallback@example.com")

    def fake_preprocess_document_image(filename: str, content: bytes) -> ProcessedDocumentImage:
        assert filename == "receipt.png"
        assert content == b"fake-image"
        return ProcessedDocumentImage(
            filename="receipt.jpg",
            mime_type="image/jpeg",
            content=b"processed-image",
        )

    async def fake_validation_process(images: list[DocumentImageInput], **_: object) -> DocumentValidationResponse:
        assert len(images) == 1
        return build_valid_validation_response(images[0].filename)

    async def fake_llm_judge(*args: object, **kwargs: object):
        raise RuntimeError("Simulated LLM failure")

    monkeypatch.setattr(claims_route, "preprocess_document_image", fake_preprocess_document_image)
    monkeypatch.setattr(claims_route.validation_processor, "process", fake_validation_process)
    monkeypatch.setattr(medicine_relevance_service.MedicineRelevanceLLMJudge, "judge", fake_llm_judge)

    response = client.post(
        "/api/claims",
        headers={"Authorization": f"Bearer {token}"},
        data={
            "nama_pasien": "Budi Santoso",
            "tanggal_pelayanan": "31-03-2026",
            "fasilitas_kesehatan": "RS Sehat Selalu",
            "jenis_manfaat": "rawat_jalan",
            "nominal_klaim": "150000",
            "keterangan_penyakit": "Demam",
            "kronologi": "Pemeriksaan rawat jalan",
            "medicines_payload": json.dumps(
                [
                    {
                        "name": "Paracetamol",
                        "source": "user_input",
                    }
                ]
            ),
        },
        files=[("files", ("receipt.png", b"fake-image", "image/png"))],
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["medicines"][0]["evaluation"]["decision_source"] == "heuristic_fallback"

    knowledge_rows = get_all_knowledge_rows()
    assert len(knowledge_rows) == 1
    assert knowledge_rows[0].support_count == 1


def test_create_claim_rejects_document_validation_failure(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = register_and_login(client, "claims-invalid-doc@example.com")

    def fake_preprocess_document_image(filename: str, content: bytes) -> ProcessedDocumentImage:
        assert filename == "receipt.png"
        assert content == b"fake-image"
        return ProcessedDocumentImage(
            filename="receipt.jpg",
            mime_type="image/jpeg",
            content=b"processed-image",
        )

    async def fake_validation_process(images: list[DocumentImageInput], **_: object) -> DocumentValidationResponse:
        assert len(images) == 1
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Document validation failed.",
                "errors": [
                    {
                        "field": "non_field",
                        "message": "Image 1 document 1: a visible facility stamp or doctor signature is required.",
                    }
                ],
            },
        )

    monkeypatch.setattr(claims_route, "preprocess_document_image", fake_preprocess_document_image)
    monkeypatch.setattr(claims_route.validation_processor, "process", fake_validation_process)

    response = client.post(
        "/api/claims",
        headers={"Authorization": f"Bearer {token}"},
        data={
            "nama_pasien": "Budi Santoso",
            "tanggal_pelayanan": "31-03-2026",
            "fasilitas_kesehatan": "RS Sehat Selalu",
            "jenis_manfaat": "rawat_jalan",
            "nominal_klaim": "150000",
            "keterangan_penyakit": "Demam",
            "kronologi": "Pemeriksaan rawat jalan",
        },
        files=[("files", ("receipt.png", b"fake-image", "image/png"))],
    )

    assert response.status_code == 422
    payload = response.json()
    assert payload["message"] == "Document validation failed."
    assert payload["errors"] == [
        {"field": "non_field", "message": "Image 1 document 1: a visible facility stamp or doctor signature is required."}
    ]
    assert get_all_claims() == []


def test_create_claim_accepts_soft_review_validation_result(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = register_and_login(client, "claims-soft-review@example.com")

    def fake_preprocess_document_image(filename: str, content: bytes) -> ProcessedDocumentImage:
        assert filename == "receipt.png"
        assert content == b"fake-image"
        return ProcessedDocumentImage(
            filename="receipt.jpg",
            mime_type="image/jpeg",
            content=b"processed-image",
        )

    async def fake_validation_process(images: list[DocumentImageInput], **_: object) -> DocumentValidationResponse:
        assert len(images) == 1
        return DocumentValidationResponse.model_validate(
            {
                "packet_patient_name": None,
                "patient_name_match_status": "not_found",
                "patient_name_match_reason": "Patient name is not visible clearly enough in the uploaded documents.",
                "review_required": True,
                "review_reasons": [
                    "Image 1: multiple documents were present but could not be validated clearly.",
                ],
                "images": [
                    {
                        "image_index": 0,
                        "filename": images[0].filename,
                        "document_count_detected": 2,
                        "documents": [
                            {
                                "document_index": 0,
                                "document_type": "receipt",
                                "is_medical_document": True,
                                "readability_ok": True,
                                "facility_identity_visible": True,
                                "stamp_detected": False,
                                "signature_detected": False,
                                "decision": "valid",
                                "reasons": ["Facility identity is visible."],
                            }
                        ],
                    }
                ],
            }
        )

    monkeypatch.setattr(claims_route, "preprocess_document_image", fake_preprocess_document_image)
    monkeypatch.setattr(claims_route.validation_processor, "process", fake_validation_process)

    response = client.post(
        "/api/claims",
        headers={"Authorization": f"Bearer {token}"},
        data={
            "nama_pasien": "Budi Santoso",
            "tanggal_pelayanan": "31-03-2026",
            "fasilitas_kesehatan": "RS Sehat Selalu",
            "jenis_manfaat": "rawat_jalan",
            "nominal_klaim": "150000",
            "keterangan_penyakit": "Demam",
            "kronologi": "Pemeriksaan rawat jalan",
        },
        files=[("files", ("receipt.png", b"fake-image", "image/png"))],
    )

    assert response.status_code == 201
    claims = get_all_claims()
    assert len(claims) == 1


def test_create_claim_links_matching_scan_log(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = register_and_login(client, "claims-link@example.com")

    def fake_preprocess_document_image(filename: str, content: bytes) -> ProcessedDocumentImage:
        assert filename == "receipt.png"
        assert content == b"fake-image"
        return ProcessedDocumentImage(
            filename="receipt.jpg",
            mime_type="image/jpeg",
            content=b"processed-image",
        )

    async def fake_validation_process(images: list[DocumentImageInput], **_: object) -> DocumentValidationResponse:
        assert len(images) == 1
        return build_valid_validation_response(images[0].filename)

    async def fake_scan_process(
        images: list[DocumentImageInput],
        patient_name_candidates: list[str] | None = None,
        **_: object,
    ) -> AIMedicalScanResponse:
        assert patient_name_candidates == ["Claim User"]
        return AIMedicalScanResponse.model_validate(SCAN_RESULT)

    monkeypatch.setattr(ai_route, "preprocess_document_image", fake_preprocess_document_image)
    monkeypatch.setattr(ai_route.validation_processor, "process", fake_validation_process)
    monkeypatch.setattr(ai_processor, "process", fake_scan_process)
    monkeypatch.setattr(claims_route, "preprocess_document_image", fake_preprocess_document_image)
    monkeypatch.setattr(claims_route.validation_processor, "process", fake_validation_process)

    scan_response = client.post(
        "/api/ai/scan-documents",
        headers={"Authorization": f"Bearer {token}"},
        files=[("files", ("receipt.png", b"fake-image", "image/png"))],
    )

    assert scan_response.status_code == 200
    scan_log_id = scan_response.json()["scan_log_id"]

    claim_response = client.post(
        "/api/claims",
        headers={"Authorization": f"Bearer {token}"},
        data={
            "nama_pasien": "Claim User",
            "tanggal_pelayanan": "31-03-2026",
            "fasilitas_kesehatan": "RS Sehat Selalu",
            "jenis_manfaat": "rawat_jalan",
            "nominal_klaim": "150000",
            "keterangan_penyakit": "Demam",
            "kronologi": "Pemeriksaan rawat jalan",
            "source_scan_log_id": str(scan_log_id),
        },
        files=[("files", ("receipt.png", b"fake-image", "image/png"))],
    )

    assert claim_response.status_code == 201
    claims = get_all_claims()
    assert len(claims) == 1
    assert claims[0].source_scan_log_id == scan_log_id


def test_create_claim_reuses_successful_scan_validation_without_revalidating(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = register_and_login(client, "claims-reuse-validation@example.com")

    def fake_preprocess_document_image(filename: str, content: bytes) -> ProcessedDocumentImage:
        assert filename == "receipt.png"
        assert content == b"fake-image"
        return ProcessedDocumentImage(
            filename="receipt.jpg",
            mime_type="image/jpeg",
            content=b"processed-image",
        )

    async def fake_validation_process(images: list[DocumentImageInput], **_: object) -> DocumentValidationResponse:
        assert len(images) == 1
        return build_valid_validation_response(images[0].filename)

    async def fake_scan_process(
        images: list[DocumentImageInput],
        patient_name_candidates: list[str] | None = None,
        **_: object,
    ) -> AIMedicalScanResponse:
        assert patient_name_candidates == ["Claim User"]
        return AIMedicalScanResponse.model_validate(SCAN_RESULT)

    async def fail_if_claim_revalidates(images: list[DocumentImageInput], **_: object) -> DocumentValidationResponse:
        raise AssertionError("Claim submission should reuse prior validation and skip revalidation.")

    monkeypatch.setattr(ai_route, "preprocess_document_image", fake_preprocess_document_image)
    monkeypatch.setattr(ai_route.validation_processor, "process", fake_validation_process)
    monkeypatch.setattr(ai_processor, "process", fake_scan_process)
    monkeypatch.setattr(claims_route, "preprocess_document_image", fake_preprocess_document_image)
    monkeypatch.setattr(claims_route.validation_processor, "process", fail_if_claim_revalidates)

    scan_response = client.post(
        "/api/ai/scan-documents",
        headers={"Authorization": f"Bearer {token}"},
        files=[("files", ("receipt.png", b"fake-image", "image/png"))],
    )

    assert scan_response.status_code == 200
    scan_log_id = scan_response.json()["scan_log_id"]

    claim_response = client.post(
        "/api/claims",
        headers={"Authorization": f"Bearer {token}"},
        data={
            "nama_pasien": "Claim User",
            "tanggal_pelayanan": "31-03-2026",
            "fasilitas_kesehatan": "RS Sehat Selalu",
            "jenis_manfaat": "rawat_jalan",
            "nominal_klaim": "150000",
            "keterangan_penyakit": "Demam",
            "kronologi": "Pemeriksaan rawat jalan",
            "source_scan_log_id": str(scan_log_id),
        },
        files=[("files", ("receipt.png", b"fake-image", "image/png"))],
    )

    assert claim_response.status_code == 201
    claims = get_all_claims()
    assert len(claims) == 1
    assert claims[0].source_scan_log_id == scan_log_id


def test_create_claim_rejects_documents_that_do_not_match_selected_patient(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = register_and_login(client, "claims-selected-mismatch@example.com")

    def fake_preprocess_document_image(filename: str, content: bytes) -> ProcessedDocumentImage:
        assert filename == "receipt.png"
        assert content == b"fake-image"
        return ProcessedDocumentImage(
            filename="receipt.jpg",
            mime_type="image/jpeg",
            content=b"processed-image",
        )

    async def fake_validation_process(images: list[DocumentImageInput], expected_patient_name: str | None = None, **_: object) -> DocumentValidationResponse:
        assert len(images) == 1
        assert expected_patient_name == "Budi Santoso"
        return DocumentValidationResponse.model_validate(
            {
                "packet_patient_name": "Claim User",
                "patient_name_match_status": "mismatched",
                "patient_name_match_reason": "Document patient name 'Claim User' does not match the selected patient.",
                "review_required": False,
                "review_reasons": [],
                "images": [
                    {
                        "image_index": 0,
                        "filename": images[0].filename,
                        "document_count_detected": 1,
                        "documents": [
                            {
                                "document_index": 0,
                                "document_type": "receipt",
                                "is_medical_document": True,
                                "readability_ok": True,
                                "facility_identity_visible": True,
                                "stamp_detected": True,
                                "signature_detected": False,
                                "decision": "valid",
                                "reasons": ["Visible facility stamp detected."],
                            }
                        ],
                    }
                ],
            }
        )

    monkeypatch.setattr(claims_route, "preprocess_document_image", fake_preprocess_document_image)
    monkeypatch.setattr(claims_route.validation_processor, "process", fake_validation_process)

    response = client.post(
        "/api/claims",
        headers={"Authorization": f"Bearer {token}"},
        data={
            "nama_pasien": "Budi Santoso",
            "tanggal_pelayanan": "31-03-2026",
            "fasilitas_kesehatan": "RS Sehat Selalu",
            "jenis_manfaat": "rawat_jalan",
            "nominal_klaim": "150000",
            "keterangan_penyakit": "Demam",
            "kronologi": "Pemeriksaan rawat jalan",
        },
        files=[("files", ("receipt.png", b"fake-image", "image/png"))],
    )

    assert response.status_code == 422
    assert response.json()["message"] == "Patient name verification failed."
    assert response.json()["errors"] == [
        {
            "field": "non_field",
            "message": "Uploaded documents do not match the selected patient. Please choose the correct patient or upload the correct documents.",
        }
    ]


def test_create_claim_reruns_validation_for_legacy_scan_log_without_patient_name_evidence(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = register_and_login(client, "claims-legacy-scan@example.com")

    def fake_preprocess_document_image(filename: str, content: bytes) -> ProcessedDocumentImage:
        assert filename == "receipt.png"
        assert content == b"fake-image"
        return ProcessedDocumentImage(
            filename="receipt.jpg",
            mime_type="image/jpeg",
            content=b"processed-image",
        )

    async def fake_scan_validation_process(images: list[DocumentImageInput], **_: object) -> DocumentValidationResponse:
        assert len(images) == 1
        return DocumentValidationResponse.model_validate(
            {
                "packet_patient_name": None,
                "patient_name_match_status": "not_checked",
                "patient_name_match_reason": None,
                "review_required": False,
                "review_reasons": [],
                "images": [
                    {
                        "image_index": 0,
                        "filename": images[0].filename,
                        "document_count_detected": 1,
                        "documents": [
                            {
                                "document_index": 0,
                                "document_type": "receipt",
                                "is_medical_document": True,
                                "readability_ok": True,
                                "facility_identity_visible": True,
                                "stamp_detected": True,
                                "signature_detected": False,
                                "decision": "valid",
                                "reasons": ["Visible facility stamp detected."],
                            }
                        ],
                    }
                ],
            }
        )

    async def fake_claim_validation_process(
        images: list[DocumentImageInput],
        expected_patient_name: str | None = None,
        **_: object,
    ) -> DocumentValidationResponse:
        assert len(images) == 1
        assert expected_patient_name == "Claim User"
        return build_valid_validation_response(images[0].filename)

    async def fake_scan_process(
        images: list[DocumentImageInput],
        patient_name_candidates: list[str] | None = None,
        **_: object,
    ) -> AIMedicalScanResponse:
        assert patient_name_candidates == ["Claim User"]
        return AIMedicalScanResponse.model_validate(SCAN_RESULT)

    monkeypatch.setattr(ai_route, "preprocess_document_image", fake_preprocess_document_image)
    monkeypatch.setattr(ai_route.validation_processor, "process", fake_scan_validation_process)
    monkeypatch.setattr(ai_processor, "process", fake_scan_process)
    monkeypatch.setattr(claims_route, "preprocess_document_image", fake_preprocess_document_image)
    monkeypatch.setattr(claims_route.validation_processor, "process", fake_claim_validation_process)

    scan_response = client.post(
        "/api/ai/scan-documents",
        headers={"Authorization": f"Bearer {token}"},
        files=[("files", ("receipt.png", b"fake-image", "image/png"))],
    )

    assert scan_response.status_code == 200
    scan_log_id = scan_response.json()["scan_log_id"]

    claim_response = client.post(
        "/api/claims",
        headers={"Authorization": f"Bearer {token}"},
        data={
            "nama_pasien": "Claim User",
            "tanggal_pelayanan": "31-03-2026",
            "fasilitas_kesehatan": "RS Sehat Selalu",
            "jenis_manfaat": "rawat_jalan",
            "nominal_klaim": "150000",
            "keterangan_penyakit": "Demam",
            "kronologi": "Pemeriksaan rawat jalan",
            "source_scan_log_id": str(scan_log_id),
        },
        files=[("files", ("receipt.png", b"fake-image", "image/png"))],
    )

    assert claim_response.status_code == 201


def test_create_claim_rejects_linked_scan_log_when_selected_patient_conflicts_with_stored_scan_name(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = register_and_login(client, "claims-linked-name-mismatch@example.com")

    def fake_preprocess_document_image(filename: str, content: bytes) -> ProcessedDocumentImage:
        assert filename == "receipt.png"
        assert content == b"fake-image"
        return ProcessedDocumentImage(
            filename="receipt.jpg",
            mime_type="image/jpeg",
            content=b"processed-image",
        )

    async def fake_validation_process(images: list[DocumentImageInput], **_: object) -> DocumentValidationResponse:
        assert len(images) == 1
        return build_valid_validation_response(images[0].filename)

    async def fake_scan_process(
        images: list[DocumentImageInput],
        patient_name_candidates: list[str] | None = None,
        **_: object,
    ) -> AIMedicalScanResponse:
        assert patient_name_candidates == ["Claim User"]
        return AIMedicalScanResponse.model_validate(SCAN_RESULT)

    monkeypatch.setattr(ai_route, "preprocess_document_image", fake_preprocess_document_image)
    monkeypatch.setattr(ai_route.validation_processor, "process", fake_validation_process)
    monkeypatch.setattr(ai_processor, "process", fake_scan_process)
    monkeypatch.setattr(claims_route, "preprocess_document_image", fake_preprocess_document_image)
    monkeypatch.setattr(claims_route.validation_processor, "process", fake_validation_process)

    scan_response = client.post(
        "/api/ai/scan-documents",
        headers={"Authorization": f"Bearer {token}"},
        files=[("files", ("receipt.png", b"fake-image", "image/png"))],
    )

    assert scan_response.status_code == 200
    scan_log_id = scan_response.json()["scan_log_id"]

    claim_response = client.post(
        "/api/claims",
        headers={"Authorization": f"Bearer {token}"},
        data={
            "nama_pasien": "Budi Santoso",
            "tanggal_pelayanan": "31-03-2026",
            "fasilitas_kesehatan": "RS Sehat Selalu",
            "jenis_manfaat": "rawat_jalan",
            "nominal_klaim": "150000",
            "keterangan_penyakit": "Demam",
            "kronologi": "Pemeriksaan rawat jalan",
            "source_scan_log_id": str(scan_log_id),
        },
        files=[("files", ("receipt.png", b"fake-image", "image/png"))],
    )

    assert claim_response.status_code == 422
    assert claim_response.json()["message"] == "Patient name verification failed."
    assert claim_response.json()["errors"] == [
        {
            "field": "non_field",
            "message": "Uploaded documents do not match the selected patient. Please choose the correct patient or upload the correct documents.",
        }
    ]


def test_create_claim_rejects_mismatched_scan_log(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = register_and_login(client, "claims-mismatch@example.com")

    def fake_preprocess_document_image(filename: str, content: bytes) -> ProcessedDocumentImage:
        return ProcessedDocumentImage(
            filename="receipt.jpg",
            mime_type="image/jpeg",
            content=b"processed-image",
        )

    async def fake_validation_process(images: list[DocumentImageInput], **_: object) -> DocumentValidationResponse:
        return build_valid_validation_response(images[0].filename)

    async def fake_scan_process(
        images: list[DocumentImageInput],
        patient_name_candidates: list[str] | None = None,
        **_: object,
    ) -> AIMedicalScanResponse:
        return AIMedicalScanResponse.model_validate(SCAN_RESULT)

    monkeypatch.setattr(ai_route, "preprocess_document_image", fake_preprocess_document_image)
    monkeypatch.setattr(ai_route.validation_processor, "process", fake_validation_process)
    monkeypatch.setattr(ai_processor, "process", fake_scan_process)
    monkeypatch.setattr(claims_route, "preprocess_document_image", fake_preprocess_document_image)
    monkeypatch.setattr(claims_route.validation_processor, "process", fake_validation_process)

    scan_response = client.post(
        "/api/ai/scan-documents",
        headers={"Authorization": f"Bearer {token}"},
        files=[("files", ("receipt.png", b"fake-image", "image/png"))],
    )
    assert scan_response.status_code == 200
    scan_log_id = scan_response.json()["scan_log_id"]

    claim_response = client.post(
        "/api/claims",
        headers={"Authorization": f"Bearer {token}"},
        data={
            "nama_pasien": "Budi Santoso",
            "tanggal_pelayanan": "31-03-2026",
            "fasilitas_kesehatan": "RS Sehat Selalu",
            "jenis_manfaat": "rawat_jalan",
            "nominal_klaim": "150000",
            "keterangan_penyakit": "Demam",
            "kronologi": "Pemeriksaan rawat jalan",
            "source_scan_log_id": str(scan_log_id),
        },
        files=[("files", ("receipt.png", b"different-image", "image/png"))],
    )

    assert claim_response.status_code == 422
    assert claim_response.json()["errors"] == [
        {"field": "source_scan_log_id", "message": "The selected scan log cannot be linked to this claim."}
    ]
    assert get_all_claims() == []
