from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, Response, UploadFile, status
from sqlalchemy.orm import Session

from app.api.routes.auth import get_current_user_dependency
from app.db.session import get_db_session
from app.repositories.family_member_repository import FamilyMemberRepository
from app.schemas.auth import TokenData
from app.schemas.family_member import FamilyMemberCreate, FamilyMemberResponse, FamilyMemberUpdate
from app.services.profile_completion import (
    build_data_url,
    get_missing_family_profile_fields,
    read_insurance_card_upload,
)

router = APIRouter(prefix="/family-members", tags=["family-members"])


def build_family_member_response(family_member: Any) -> FamilyMemberResponse:
    missing_fields = get_missing_family_profile_fields(family_member)
    return FamilyMemberResponse(
        id=family_member.id,
        name=family_member.name,
        relation=family_member.relation,
        insurance_card_number=family_member.insurance_card_number,
        phone_number=family_member.phone_number,
        mobile_number=family_member.mobile_number,
        email=family_member.email,
        insurance_card_filename=family_member.insurance_card_filename,
        insurance_card_mime_type=family_member.insurance_card_mime_type,
        insurance_card_byte_size=family_member.insurance_card_byte_size,
        insurance_card_sha256_hash=family_member.insurance_card_sha256_hash,
        insurance_card_data_url=build_data_url(family_member.insurance_card_mime_type, family_member.insurance_card_content),
        profile_complete=len(missing_fields) == 0,
        missing_profile_fields=missing_fields,
    )


@router.get("", response_model=list[FamilyMemberResponse])
async def get_family_members(
    token_data: TokenData = Depends(get_current_user_dependency),
    db: Session = Depends(get_db_session),
) -> list[FamilyMemberResponse]:
    family_member_repository = FamilyMemberRepository(db)
    family_members = family_member_repository.get_by_user_id(token_data.user_id)

    return [build_family_member_response(family_member) for family_member in family_members]


@router.post("", response_model=FamilyMemberResponse, status_code=status.HTTP_201_CREATED)
async def create_family_member(
    request: FamilyMemberCreate,
    token_data: TokenData = Depends(get_current_user_dependency),
    db: Session = Depends(get_db_session),
) -> FamilyMemberResponse:
    family_member_repository = FamilyMemberRepository(db)
    family_member = family_member_repository.create(
        user_id=token_data.user_id,
        name=request.name,
        relation=request.relation,
        insurance_card_number=request.insurance_card_number,
        phone_number=request.phone_number,
        mobile_number=request.mobile_number,
        email=request.email,
    )

    return build_family_member_response(family_member)


@router.put("/{family_member_id}", response_model=FamilyMemberResponse)
async def update_family_member(
    family_member_id: int,
    request: FamilyMemberUpdate,
    token_data: TokenData = Depends(get_current_user_dependency),
    db: Session = Depends(get_db_session),
) -> FamilyMemberResponse:
    family_member_repository = FamilyMemberRepository(db)
    family_member = family_member_repository.get_by_id_and_user_id(
        family_member_id=family_member_id,
        user_id=token_data.user_id,
    )
    if family_member is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Family member not found.",
        )

    updated_family_member = family_member_repository.update(
        family_member,
        name=request.name,
        relation=request.relation,
        insurance_card_number=request.insurance_card_number,
        phone_number=request.phone_number,
        mobile_number=request.mobile_number,
        email=request.email,
    )
    return build_family_member_response(updated_family_member)


@router.post("/{family_member_id}/insurance-card", response_model=FamilyMemberResponse)
async def upload_family_member_insurance_card(
    family_member_id: int,
    file: UploadFile = File(...),
    token_data: TokenData = Depends(get_current_user_dependency),
    db: Session = Depends(get_db_session),
) -> FamilyMemberResponse:
    family_member_repository = FamilyMemberRepository(db)
    family_member = family_member_repository.get_by_id_and_user_id(
        family_member_id=family_member_id,
        user_id=token_data.user_id,
    )
    if family_member is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Family member not found.",
        )

    upload = await read_insurance_card_upload(file)
    updated_family_member = family_member_repository.update_insurance_card(
        family_member,
        filename=upload.filename,
        mime_type=upload.mime_type,
        byte_size=upload.byte_size,
        sha256_hash=upload.sha256_hash,
        content=upload.content,
    )
    return build_family_member_response(updated_family_member)


@router.delete("/{family_member_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_family_member(
    family_member_id: int,
    token_data: TokenData = Depends(get_current_user_dependency),
    db: Session = Depends(get_db_session),
) -> Response:
    family_member_repository = FamilyMemberRepository(db)
    family_member = family_member_repository.get_by_id_and_user_id(
        family_member_id=family_member_id,
        user_id=token_data.user_id,
    )
    if family_member is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Family member not found.",
        )

    family_member_repository.delete(family_member)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
