from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, AsyncGenerator, AsyncIterator, Awaitable, Callable, Literal

from sqlalchemy.orm import Session

from app.models.scan_log import ScanLog
from app.repositories.scan_log_repository import ScanLogRepository
from app.schemas.ai import ScanJobStage, ScanJobStatus, ScanProgressPayload


ScanProgressEventName = Literal["progress", "completed", "failed", "cancelled", "heartbeat"]


@dataclass(frozen=True)
class ScanProgressStreamEvent:
    event: ScanProgressEventName
    payload: dict[str, Any]
    terminal: bool = False


@dataclass(frozen=True)
class ScanProgressSubscriber:
    queue: asyncio.Queue[ScanProgressStreamEvent]
    loop: asyncio.AbstractEventLoop


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


class ScanProgressEventHub:
    def __init__(self) -> None:
        self._subscribers: dict[int, set[ScanProgressSubscriber]] = {}

    def subscribe(self, scan_log_id: int) -> asyncio.Queue[ScanProgressStreamEvent]:
        queue: asyncio.Queue[ScanProgressStreamEvent] = asyncio.Queue(maxsize=10)
        subscriber = ScanProgressSubscriber(
            queue=queue,
            loop=asyncio.get_running_loop(),
        )
        self._subscribers.setdefault(scan_log_id, set()).add(subscriber)
        return queue

    def unsubscribe(self, scan_log_id: int, queue: asyncio.Queue[ScanProgressStreamEvent]) -> None:
        subscribers = self._subscribers.get(scan_log_id)
        if not subscribers:
            return
        subscribers = {
            subscriber
            for subscriber in subscribers
            if subscriber.queue is not queue
        }
        if subscribers:
            self._subscribers[scan_log_id] = subscribers
            return
        self._subscribers.pop(scan_log_id, None)

    def publish(
        self,
        scan_log_id: int,
        *,
        event: ScanProgressEventName,
        payload: dict[str, Any],
        terminal: bool = False,
    ) -> None:
        stream_event = ScanProgressStreamEvent(
            event=event,
            payload=payload,
            terminal=terminal,
        )
        for subscriber in list(self._subscribers.get(scan_log_id, ())):
            try:
                subscriber.loop.call_soon_threadsafe(
                    self._publish_to_subscriber,
                    subscriber.queue,
                    stream_event,
                )
            except RuntimeError:
                self.unsubscribe(scan_log_id, subscriber.queue)

    def subscriber_count(self, scan_log_id: int) -> int:
        return len(self._subscribers.get(scan_log_id, ()))

    def _publish_to_subscriber(
        self,
        queue: asyncio.Queue[ScanProgressStreamEvent],
        stream_event: ScanProgressStreamEvent,
    ) -> None:
        if queue.full():
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        try:
            queue.put_nowait(stream_event)
        except asyncio.QueueFull:
            return


def format_sse_event(event: ScanProgressEventName, payload: dict[str, Any] | None = None) -> str:
    lines = [f"event: {event}"]
    if payload is not None:
        lines.append(f"data: {json.dumps(payload, ensure_ascii=False)}")
    return "\n".join(lines) + "\n\n"


scan_progress_event_hub = ScanProgressEventHub()


async def generate_scan_progress_sse(
    *,
    scan_log_id: int,
    initial_event: ScanProgressEventName,
    initial_payload: dict[str, Any],
    initial_terminal: bool,
    is_disconnected: Callable[[], Awaitable[bool]],
    event_hub: ScanProgressEventHub = scan_progress_event_hub,
    heartbeat_interval: float = 15.0,
    poll_interval: float = 1.0,
) -> AsyncGenerator[str, None]:
    queue = event_hub.subscribe(scan_log_id)
    try:
        yield format_sse_event(initial_event, initial_payload)

        if initial_terminal:
            return

        loop = asyncio.get_running_loop()
        next_heartbeat_at = loop.time() + heartbeat_interval

        while True:
            if await is_disconnected():
                break

            try:
                stream_event = await asyncio.wait_for(queue.get(), timeout=poll_interval)
            except asyncio.TimeoutError:
                if await is_disconnected():
                    break
                if heartbeat_interval <= 0:
                    continue
                now = loop.time()
                if now < next_heartbeat_at:
                    continue
                next_heartbeat_at = now + heartbeat_interval
                yield format_sse_event("heartbeat", {"scan_log_id": scan_log_id})
                continue

            yield format_sse_event(stream_event.event, stream_event.payload)
            if stream_event.terminal:
                break
    finally:
        event_hub.unsubscribe(scan_log_id, queue)
