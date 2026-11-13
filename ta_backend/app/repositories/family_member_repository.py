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
        insurance_card_number: str | None = None,
        phone_number: str | None = None,
        mobile_number: str | None = None,
        email: str | None = None,
    ) -> FamilyMember:
        family_member = FamilyMember(
            user_id=user_id,
            name=name.strip(),
            relation=relation.strip(),
            insurance_card_number=insurance_card_number.strip() if insurance_card_number else None,
            phone_number=phone_number.strip() if phone_number else None,
            mobile_number=mobile_number.strip() if mobile_number else None,
            email=email.strip().lower() if email else None,
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
        insurance_card_number: str | None = None,
        phone_number: str | None = None,
        mobile_number: str | None = None,
        email: str | None = None,
    ) -> FamilyMember:
        family_member.name = name.strip()
        family_member.relation = relation.strip()
        family_member.insurance_card_number = insurance_card_number.strip() if insurance_card_number else None
        family_member.phone_number = phone_number.strip() if phone_number else None
        family_member.mobile_number = mobile_number.strip() if mobile_number else None
        family_member.email = email.strip().lower() if email else None
        self._db.commit()
        self._db.refresh(family_member)
        return family_member

    def update_insurance_card(
        self,
        family_member: FamilyMember,
        *,
        filename: str,
        mime_type: str,
        byte_size: int,
        sha256_hash: str,
        content: bytes,
    ) -> FamilyMember:
        family_member.insurance_card_filename = filename
        family_member.insurance_card_mime_type = mime_type
        family_member.insurance_card_byte_size = byte_size
        family_member.insurance_card_sha256_hash = sha256_hash
        family_member.insurance_card_content = content
        self._db.commit()
        self._db.refresh(family_member)
        return family_member

    def delete(self, family_member: FamilyMember) -> None:
        self._db.delete(family_member)
        self._db.commit()
