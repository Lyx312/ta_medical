from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Depends, File, Header, HTTPException, UploadFile, status
from jose import JWTError, jwt
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.session import get_db_session
from app.models.user import hash_password, verify_password
from app.repositories.user_repository import UserRepository
from app.schemas.auth import (
    LoginRequest,
    LoginResponse,
    RegisterRequest,
    RegisterResponse,
    TokenData,
    UserRole,
    UserPublic,
    UserProfileUpdate,
)
from app.services.profile_completion import (
    build_data_url,
    get_missing_user_profile_fields,
    read_insurance_card_upload,
)

router = APIRouter(prefix="/auth", tags=["auth"])


def build_user_public(user: Any) -> UserPublic:
    missing_fields = get_missing_user_profile_fields(user)
    return UserPublic(
        id=user.id,
        full_name=user.full_name,
        email=user.email,
        role=user.role,
        nip=user.nip,
        company_name=user.company_name,
        insurance_card_number=user.insurance_card_number,
        phone_number=user.phone_number,
        mobile_number=user.mobile_number,
        bank_account_number=user.bank_account_number,
        bank_account_holder_name=user.bank_account_holder_name,
        bank_name=user.bank_name,
        insurance_card_filename=user.insurance_card_filename,
        insurance_card_mime_type=user.insurance_card_mime_type,
        insurance_card_byte_size=user.insurance_card_byte_size,
        insurance_card_sha256_hash=user.insurance_card_sha256_hash,
        insurance_card_data_url=build_data_url(user.insurance_card_mime_type, user.insurance_card_content),
        profile_complete=len(missing_fields) == 0,
        missing_profile_fields=missing_fields,
    )


def parse_user_role(value: object) -> UserRole:
    if value in ("user", "admin"):
        return value
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials.",
        headers={"WWW-Authenticate": "Bearer"},
    )


def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    settings = get_settings()
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=settings.access_token_expire_minutes)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode, settings.jwt_secret, algorithm="HS256"
    )
    return encoded_jwt


def get_current_user(token: str) -> TokenData:
    settings = get_settings()
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
        user_id = payload.get("sub")
        email = payload.get("email")
        role = parse_user_role(payload.get("role"))
        if user_id is None or email is None or role is None:
            raise credentials_exception
        return TokenData(user_id=int(user_id), email=str(email), role=role)
    except JWTError:
        raise credentials_exception


@router.post("/register", response_model=RegisterResponse, status_code=status.HTTP_201_CREATED)
async def register_user(
    request: RegisterRequest,
    db: Session = Depends(get_db_session),
) -> RegisterResponse:
    user_repository = UserRepository(db)
    existing_user = user_repository.get_by_email(request.email)
    if existing_user is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "message": "Input validation failed.",
                "errors": [{"field": "email", "message": "Email is already registered."}],
            },
        )

    try:
        new_user = user_repository.create_user(
            full_name=request.full_name,
            email=request.email,
            password_hash=hash_password(request.password),
            role="user",
        )
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "message": "Input validation failed.",
                "errors": [{"field": "email", "message": "Email is already registered."}],
            },
        )

    access_token = create_access_token(
        data={"sub": str(new_user.id), "email": new_user.email, "role": new_user.role}
    )

    return RegisterResponse(
        message="Registration successful.",
        access_token=access_token,
        user=build_user_public(new_user),
    )


@router.post("/login", response_model=LoginResponse)
async def login_user(
    request: LoginRequest,
    db: Session = Depends(get_db_session),
) -> LoginResponse:
    user_repository = UserRepository(db)
    matched_user = user_repository.get_by_email(request.email)
    if matched_user is None or not verify_password(request.password, matched_user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "message": "Input validation failed.",
                "errors": [{"field": "credentials", "message": "Wrong email or password."}],
            },
        )

    access_token = create_access_token(
        data={"sub": str(matched_user.id), "email": matched_user.email, "role": matched_user.role}
    )

    return LoginResponse(
        message="Login successful.",
        access_token=access_token,
        user=build_user_public(matched_user),
    )


async def get_current_user_dependency(authorization: str | None = Header(None)) -> TokenData:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = authorization[7:]
    return get_current_user(token)


async def get_current_admin_dependency(
    token_data: TokenData = Depends(get_current_user_dependency),
) -> TokenData:
    if token_data.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to access this resource.",
        )
    return token_data


@router.get("/me", response_model=UserPublic)
async def get_me(
    token_data: TokenData = Depends(get_current_user_dependency),
    db: Session = Depends(get_db_session),
) -> UserPublic:
    user_repository = UserRepository(db)
    user = user_repository.get_by_id(token_data.user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )
    return build_user_public(user)


@router.put("/me/profile", response_model=UserPublic)
async def update_me_profile(
    request: UserProfileUpdate,
    token_data: TokenData = Depends(get_current_user_dependency),
    db: Session = Depends(get_db_session),
) -> UserPublic:
    user_repository = UserRepository(db)
    user = user_repository.get_by_id(token_data.user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )

    updated_user = user_repository.update_profile(
        user,
        nip=request.nip,
        company_name=request.company_name,
        insurance_card_number=request.insurance_card_number,
        phone_number=request.phone_number,
        mobile_number=request.mobile_number,
        bank_account_number=request.bank_account_number,
        bank_account_holder_name=request.bank_account_holder_name,
        bank_name=request.bank_name,
    )
    return build_user_public(updated_user)


@router.post("/me/insurance-card", response_model=UserPublic)
async def upload_me_insurance_card(
    file: UploadFile = File(...),
    token_data: TokenData = Depends(get_current_user_dependency),
    db: Session = Depends(get_db_session),
) -> UserPublic:
    user_repository = UserRepository(db)
    user = user_repository.get_by_id(token_data.user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )

    upload = await read_insurance_card_upload(file)
    updated_user = user_repository.update_insurance_card(
        user,
        filename=upload.filename,
        mime_type=upload.mime_type,
        byte_size=upload.byte_size,
        sha256_hash=upload.sha256_hash,
        content=upload.content,
    )
    return build_user_public(updated_user)
