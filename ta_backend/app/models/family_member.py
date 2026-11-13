from sqlalchemy import ForeignKey, Integer, LargeBinary, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class FamilyMember(Base):
    __tablename__ = "family_members"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True, nullable=False)
    name: Mapped[str] = mapped_column(String(150), nullable=False)
    relation: Mapped[str] = mapped_column(String(100), nullable=False)
    insurance_card_number: Mapped[str | None] = mapped_column(String(100), nullable=True)
    phone_number: Mapped[str | None] = mapped_column(String(50), nullable=True)
    mobile_number: Mapped[str | None] = mapped_column(String(50), nullable=True)
    email: Mapped[str | None] = mapped_column(String(255), nullable=True)
    insurance_card_filename: Mapped[str | None] = mapped_column(String(255), nullable=True)
    insurance_card_mime_type: Mapped[str | None] = mapped_column(String(100), nullable=True)
    insurance_card_byte_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    insurance_card_sha256_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    insurance_card_content: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
