from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal

_counter = 0


def _next_id() -> int:
    global _counter
    _counter += 1
    return _counter


@dataclass(frozen=True)
class Claim:
    id: int
    user_id: int
    user_email: str
    tanggal_pelayanan: date
    fasilitas_kesehatan: str
    jenis_manfaat: Literal["rawat_jalan", "rawat_inap", "emergency", "maternity"]
    nominal_klaim: float
    keterangan_penyakit: str
    kronologi: str
    files: list[str]
    status: str
    created_at: datetime


class ClaimRepository:
    def __init__(self) -> None:
        self._claims: dict[int, Claim] = {}

    def add(self, claim: Claim) -> Claim:
        self._claims[claim.id] = claim
        return claim

    def get_by_id(self, claim_id: int) -> Claim | None:
        return self._claims.get(claim_id)

    def get_by_user_id(self, user_id: int) -> list[Claim]:
        return [c for c in self._claims.values() if c.user_id == user_id]

    def get_all(self) -> list[Claim]:
        return list(self._claims.values())

    def clear(self) -> None:
        self._claims.clear()
