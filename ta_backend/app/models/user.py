from dataclasses import dataclass
import hashlib
import hmac
import os
from typing import Literal

_counter = 0


def _next_id() -> int:
    global _counter
    _counter += 1
    return _counter


@dataclass(frozen=True)
class UserInDB:
    id: int
    full_name: str
    email: str
    password_hash: str
    role: Literal["user", "admin"]


class UserRepository:
    def __init__(self) -> None:
        self._users_by_email: dict[str, UserInDB] = {}
        self._users_by_id: dict[int, UserInDB] = {}

        admin = UserInDB(
            id=_next_id(),
            full_name="Admin User",
            email="admin@mail.com",
            password_hash=hash_password("admin123"),
            role="admin",
        )
        self.add(admin)

        user = UserInDB(
            id=_next_id(),
            full_name="John Doe",
            email="john@mail.com",
            password_hash=hash_password("pass123"),
            role="user",
        )
        self.add(user)

    def get_by_email(self, email: str) -> UserInDB | None:
        return self._users_by_email.get(email.lower())

    def get_by_id(self, user_id: int) -> UserInDB | None:
        return self._users_by_id.get(user_id)

    def add(self, user: UserInDB) -> UserInDB:
        self._users_by_email[user.email.lower()] = user
        self._users_by_id[user.id] = user
        return user

    def clear(self) -> None:
        self._users_by_email.clear()
        self._users_by_id.clear()


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    iterations = 120_000
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"pbkdf2_sha256${iterations}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        method, iterations_str, salt_hex, digest_hex = stored_hash.split("$")
        algorithm = method.replace("pbkdf2_", "", 1)
        iterations = int(iterations_str)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(digest_hex)
    except (ValueError, TypeError):
        return False

    calculated = hashlib.pbkdf2_hmac(
        algorithm,
        password.encode("utf-8"),
        salt,
        iterations,
    )
    return hmac.compare_digest(calculated, expected)
