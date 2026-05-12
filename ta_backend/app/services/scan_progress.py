from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session

from app.models.scan_log import ScanLog
from app.repositories.scan_log_repository import ScanLogRepository
from app.schemas.ai import ScanJobStage, ScanJobStatus, ScanProgressPayload


@dataclass(frozen=True)
class ScanStageProgress:
    stage: ScanJobStage
    title: str
    percent: int
    status: ScanJobStatus = "running"
    detail: str | None = None
    image_count: int | None = None
    total_runs: int | None = None
    completed_runs: int | None = None
    successful_runs: int | None = None
    medicine_count: int | None = None

    def to_payload(self) -> dict[str, Any]:
        return ScanProgressPayload(
            stage=self.stage,
            title=self.title,
            detail=self.detail,
            percent=self.percent,
            status=self.status,
            image_count=self.image_count,
            total_runs=self.total_runs,
            completed_runs=self.completed_runs,
            successful_runs=self.successful_runs,
            medicine_count=self.medicine_count,
        ).model_dump()


class ScanProgressReporter:
    def __init__(self, db: Session, scan_log: ScanLog) -> None:
        self._db = db
        self._repository = ScanLogRepository(db)
        self._scan_log = scan_log

    @property
    def scan_log(self) -> ScanLog:
        return self._scan_log

    def replace_scan_log(self, scan_log: ScanLog) -> None:
        self._scan_log = scan_log

    def set(self, progress: ScanStageProgress) -> ScanLog:
        self._scan_log = self._repository.update_progress(
            self._scan_log,
            progress_payload=progress.to_payload(),
            status=progress.status,
        )
        return self._scan_log
