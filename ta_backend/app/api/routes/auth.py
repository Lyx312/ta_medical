from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Header, HTTPException, status
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
)

router = APIRouter(prefix="/auth", tags=["auth"])


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

    return RegisterResponse(
        message="Registration successful.",
        user=UserPublic(
            id=new_user.id,
            full_name=new_user.full_name,
            email=new_user.email,
            role=new_user.role,
        ),
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
        user=UserPublic(
            id=matched_user.id,
            full_name=matched_user.full_name,
            email=matched_user.email,
            role=matched_user.role,
        ),
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
    return UserPublic(
        id=user.id,
        full_name=user.full_name,
        email=user.email,
        role=user.role,
    )
