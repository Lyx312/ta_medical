from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.family_member import FamilyMember


class FamilyMemberRepository:
    def __init__(self, db: Session) -> None:
        self._db = db

    def get_by_user_id(self, user_id: int) -> list[FamilyMember]:
        return list(
            self._db.execute(
                select(FamilyMember)
                .where(FamilyMember.user_id == user_id)
                .order_by(FamilyMember.id.asc())
            ).scalars()
        )

    def get_by_id_and_user_id(self, family_member_id: int, user_id: int) -> FamilyMember | None:
        return self._db.execute(
            select(FamilyMember).where(
                FamilyMember.id == family_member_id,
                FamilyMember.user_id == user_id,
            )
        ).scalar_one_or_none()

    def create(
        self,
        *,
        user_id: int,
        name: str,
        relation: str,
    ) -> FamilyMember:
        family_member = FamilyMember(
            user_id=user_id,
            name=name.strip(),
            relation=relation.strip(),
        )
        self._db.add(family_member)
        self._db.commit()
        self._db.refresh(family_member)
        return family_member

    def update(
        self,
        family_member: FamilyMember,
        *,
        name: str,
        relation: str,
    ) -> FamilyMember:
        family_member.name = name.strip()
        family_member.relation = relation.strip()
        self._db.commit()
        self._db.refresh(family_member)
        return family_member

    def delete(self, family_member: FamilyMember) -> None:
        self._db.delete(family_member)
        self._db.commit()
