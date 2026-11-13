from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.user import User, UserRole


class UserRepository:
    def __init__(self, db: Session) -> None:
        self._db = db

    def get_by_email(self, email: str) -> User | None:
        normalized_email = email.strip().lower()
        return self._db.execute(
            select(User).where(User.email == normalized_email)
        ).scalar_one_or_none()

    def get_by_id(self, user_id: int) -> User | None:
        return self._db.get(User, user_id)

    def create_user(
        self,
        *,
        full_name: str,
        email: str,
        password_hash: str,
        role: UserRole = "user",
    ) -> User:
        user = User(
            full_name=full_name.strip(),
            email=email.strip().lower(),
            password_hash=password_hash,
            role=role,
        )
        self._db.add(user)
        self._db.commit()
        self._db.refresh(user)
        return user

    def update_profile(
        self,
        user: User,
        *,
        nip: str | None,
        company_name: str | None,
        insurance_card_number: str | None,
        phone_number: str | None,
        mobile_number: str | None,
        bank_account_number: str | None,
        bank_account_holder_name: str | None,
        bank_name: str | None,
    ) -> User:
        user.nip = nip
        user.company_name = company_name
        user.insurance_card_number = insurance_card_number
        user.phone_number = phone_number
        user.mobile_number = mobile_number
        user.bank_account_number = bank_account_number
        user.bank_account_holder_name = bank_account_holder_name
        user.bank_name = bank_name
        self._db.commit()
        self._db.refresh(user)
        return user

    def update_insurance_card(
        self,
        user: User,
        *,
        filename: str,
        mime_type: str,
        byte_size: int,
        sha256_hash: str,
        content: bytes,
    ) -> User:
        user.insurance_card_filename = filename
        user.insurance_card_mime_type = mime_type
        user.insurance_card_byte_size = byte_size
        user.insurance_card_sha256_hash = sha256_hash
        user.insurance_card_content = content
        self._db.commit()
        self._db.refresh(user)
        return user
