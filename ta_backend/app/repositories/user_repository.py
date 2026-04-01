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
