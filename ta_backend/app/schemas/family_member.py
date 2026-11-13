from pydantic import BaseModel, Field, field_validator


class FamilyMemberWrite(BaseModel):
    name: str = Field(max_length=150)
    relation: str = Field(max_length=100)
    insurance_card_number: str | None = Field(default=None, max_length=100)
    phone_number: str | None = Field(default=None, max_length=50)
    mobile_number: str | None = Field(default=None, max_length=50)
    email: str | None = Field(default=None, max_length=255)

    @field_validator("name", "relation", "insurance_card_number", "phone_number", "mobile_number", "email")
    @classmethod
    def validate_non_empty_trimmed(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized_value = value.strip()
        if not normalized_value:
            raise ValueError("Field cannot be empty.")
        return normalized_value


class FamilyMemberCreate(FamilyMemberWrite):
    pass


class FamilyMemberUpdate(FamilyMemberWrite):
    pass


class FamilyMemberResponse(BaseModel):
    id: int
    name: str
    relation: str
    insurance_card_number: str | None = None
    phone_number: str | None = None
    mobile_number: str | None = None
    email: str | None = None
    insurance_card_filename: str | None = None
    insurance_card_mime_type: str | None = None
    insurance_card_byte_size: int | None = None
    insurance_card_sha256_hash: str | None = None
    insurance_card_data_url: str | None = None
    profile_complete: bool = False
    missing_profile_fields: list[str] = Field(default_factory=list)
