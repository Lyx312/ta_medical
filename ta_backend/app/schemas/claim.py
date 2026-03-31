from datetime import datetime, date
from typing import Literal

from pydantic import BaseModel, Field


class ClaimCreate(BaseModel):
    tanggal_pelayanan: date
    fasilitas_kesehatan: str = Field(min_length=1, max_length=255)
    jenis_manfaat: Literal["rawat_jalan", "rawat_inap", "emergency", "maternity"]
    nominal_klaim: float = Field(ge=0)
    keterangan_penyakit: str = Field(min_length=1)
    kronologi: str = Field(min_length=1)


class ClaimResponse(BaseModel):
    id: int
    user_id: int
    user_email: str
    tanggal_pelayanan: date
    fasilitas_kesehatan: str
    jenis_manfaat: str
    nominal_klaim: float
    keterangan_penyakit: str
    kronologi: str
    file_count: int
    status: str
    created_at: datetime
