import re
from datetime import datetime, timedelta, timezone
from typing import Literal

from jose import JWTError, jwt
from pydantic import BaseModel, Field, field_validator, model_validator

from app.core.config import get_settings

EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _validate_email(value: str) -> str:
    email = value.strip().lower()
    if len(email) > 255:
        raise ValueError("Email must be at most 255 characters long.")
    if not EMAIL_PATTERN.fullmatch(email):
        raise ValueError("Please enter a valid email address.")
    return email


UserRole = Literal["user", "admin"]


class RegisterRequest(BaseModel):
    full_name: str = Field(min_length=1, max_length=150)
    email: str = Field(max_length=255)
    password: str = Field(max_length=128)
    confirm_password: str = Field(max_length=128)

    @field_validator("full_name")
    @classmethod
    def validate_full_name(cls, value: str) -> str:
        full_name = value.strip()
        if not full_name:
            raise ValueError("Full name cannot be empty.")
        return full_name

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str) -> str:
        return _validate_email(value)

    @field_validator("password", "confirm_password")
    @classmethod
    def validate_password_fields(cls, value: str) -> str:
        password = value.strip()
        if len(password) < 6:
            raise ValueError("Password must be at least 6 characters long.")
        if len(password) > 128:
            raise ValueError("Password must be at most 128 characters long.")
        return password

    @model_validator(mode="after")
    def validate_password_match(self) -> "RegisterRequest":
        if self.password != self.confirm_password:
            raise ValueError("Password and confirm password do not match.")
        return self


class UserPublic(BaseModel):
    id: int
    full_name: str
    email: str
    role: UserRole


class RegisterResponse(BaseModel):
    message: str
    user: UserPublic


class LoginRequest(BaseModel):
    email: str = Field(max_length=255)
    password: str = Field(max_length=128)

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str) -> str:
        return _validate_email(value)

    @field_validator("password")
    @classmethod
    def validate_password(cls, value: str) -> str:
        password = value.strip()
        if len(password) > 128:
            raise ValueError("Password must be at most 128 characters long.")
        return password


class TokenPayload(BaseModel):
    sub: int
    email: str
    role: UserRole
    exp: datetime


class LoginResponse(BaseModel):
    message: str
    access_token: str
    token_type: str = "bearer"
    user: UserPublic


class TokenData(BaseModel):
    user_id: int
    email: str
    role: UserRole
