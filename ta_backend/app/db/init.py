from sqlalchemy import select

from app.db.base import Base
from app.models.family_member import FamilyMember
from app.db.session import get_engine, get_session_factory
from app.models.user import User, hash_password


def initialize_database() -> None:
    Base.metadata.create_all(bind=get_engine())
    seed_default_admin()


def seed_default_admin() -> None:
    session_factory = get_session_factory()
    with session_factory() as db:
        existing_admin = db.execute(
            select(User).where(User.email == "admin@mail.com")
        ).scalar_one_or_none()
        if existing_admin is not None:
            return

        db.add(
            User(
                full_name="Admin User",
                email="admin@mail.com",
                password_hash=hash_password("admin123"),
                role="admin",
            )
        )
        db.commit()
