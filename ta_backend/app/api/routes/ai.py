import base64

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.api.routes.auth import get_current_user_dependency
from app.core.config import get_settings
from app.db.session import get_db_session
from app.repositories.family_member_repository import FamilyMemberRepository
from app.repositories.user_repository import UserRepository
from app.schemas.auth import TokenData
from app.schemas.ai import AIMedicalScanResponse
from app.services.ai_service import DocumentImageInput, MedicalDocumentProcessor
from app.services.image_preprocessing import preprocess_document_image

router = APIRouter(prefix="/ai", tags=["ai"])
processor = MedicalDocumentProcessor(settings=get_settings())


def build_patient_name_candidates(user_full_name: str, family_member_names: list[str]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()

    for candidate in [user_full_name, *family_member_names]:
        normalized_candidate = candidate.strip()
        if not normalized_candidate:
            continue

        dedupe_key = normalized_candidate.casefold()
        if dedupe_key in seen:
            continue

        seen.add(dedupe_key)
        names.append(normalized_candidate)

    return names


@router.post("/scan-documents", response_model=AIMedicalScanResponse)
async def scan_documents(
    token_data: TokenData = Depends(get_current_user_dependency),
    db: Session = Depends(get_db_session),
    files: list[UploadFile] = File(...),
) -> AIMedicalScanResponse:
    user_repository = UserRepository(db)
    family_member_repository = FamilyMemberRepository(db)
    user = user_repository.get_by_id(token_data.user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )

    family_members = family_member_repository.get_by_user_id(token_data.user_id)
    images: list[DocumentImageInput] = []
    for file in files:
        content = await file.read()
        processed_image = preprocess_document_image(
            filename=file.filename or "uploaded_file",
            content=content,
        )
        images.append(
            DocumentImageInput(
                filename=processed_image.filename,
                mime_type=processed_image.mime_type,
                base64_data=base64.b64encode(processed_image.content).decode("utf-8"),
            )
        )

    patient_name_candidates = build_patient_name_candidates(
        user_full_name=user.full_name,
        family_member_names=[family_member.name for family_member in family_members],
    )
    return await processor.process(images, patient_name_candidates=patient_name_candidates)
