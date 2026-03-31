from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Header, HTTPException, status
from jose import JWTError, jwt

from app.core.config import get_settings
from app.models.user import UserInDB, UserRepository, hash_password, verify_password
from app.schemas.auth import (
    LoginRequest,
    LoginResponse,
    RegisterRequest,
    RegisterResponse,
    TokenData,
    UserPublic,
)

router = APIRouter(prefix="/auth", tags=["auth"])
user_repository = UserRepository()
settings = get_settings()


def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
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
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
        user_id = payload.get("sub")
        email = payload.get("email")
        role = payload.get("role")
        if user_id is None or email is None or role is None:
            raise credentials_exception
        return TokenData(user_id=int(user_id), email=str(email), role=str(role))
    except JWTError:
        raise credentials_exception


@router.post("/register", response_model=RegisterResponse, status_code=status.HTTP_201_CREATED)
async def register_user(
    request: RegisterRequest,
) -> RegisterResponse:
    existing_user = user_repository.get_by_email(request.email)
    if existing_user is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "message": "Input validation failed.",
                "errors": [{"field": "email", "message": "Email is already registered."}],
            },
        )

    new_user = user_repository.add(UserInDB(
        id=user_repository._users_by_email.__len__() + 1,
        full_name=request.full_name,
        email=request.email,
        password_hash=hash_password(request.password),
        role="user",
    ))

    access_token = create_access_token(
        data={"sub": new_user.id, "email": new_user.email, "role": new_user.role}
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
) -> LoginResponse:
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
        data={"sub": matched_user.id, "email": matched_user.email, "role": matched_user.role}
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


@router.get("/me", response_model=UserPublic)
async def get_me(token_data: TokenData = Depends(get_current_user_dependency)) -> UserPublic:
    print(f"Token data: {token_data}")
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
