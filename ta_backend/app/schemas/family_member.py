from pydantic import BaseModel, Field, field_validator


class FamilyMemberWrite(BaseModel):
    name: str = Field(max_length=150)
    relation: str = Field(max_length=100)

    @field_validator("name", "relation")
    @classmethod
    def validate_non_empty_trimmed(cls, value: str) -> str:
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
