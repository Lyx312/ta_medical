from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.orm import Session

from app.api.routes.auth import get_current_user_dependency
from app.db.session import get_db_session
from app.repositories.family_member_repository import FamilyMemberRepository
from app.schemas.auth import TokenData
from app.schemas.family_member import FamilyMemberCreate, FamilyMemberResponse, FamilyMemberUpdate

router = APIRouter(prefix="/family-members", tags=["family-members"])


@router.get("", response_model=list[FamilyMemberResponse])
async def get_family_members(
    token_data: TokenData = Depends(get_current_user_dependency),
    db: Session = Depends(get_db_session),
) -> list[FamilyMemberResponse]:
    family_member_repository = FamilyMemberRepository(db)
    family_members = family_member_repository.get_by_user_id(token_data.user_id)

    return [
        FamilyMemberResponse(
            id=family_member.id,
            name=family_member.name,
            relation=family_member.relation,
        )
        for family_member in family_members
    ]


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
    )

    return FamilyMemberResponse(
        id=family_member.id,
        name=family_member.name,
        relation=family_member.relation,
    )


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
    )
    return FamilyMemberResponse(
        id=updated_family_member.id,
        name=updated_family_member.name,
        relation=updated_family_member.relation,
    )


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
