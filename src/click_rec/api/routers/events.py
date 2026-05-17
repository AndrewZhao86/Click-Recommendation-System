"""`/events/*` ingestion router — HTTP → validate → dedupe → Kafka → 202."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime

from aiokafka.errors import KafkaConnectionError, KafkaTimeoutError
from fastapi import APIRouter, HTTPException, status
from uuid6 import uuid7

from click_rec.cache.redis_client import dedupe_event_id, release_event_id
from click_rec.kafka.producer import publish_event
from click_rec.kafka.topics import USER_CLICKS, USER_IMPRESSIONS, USER_SEARCHES
from click_rec.models.requests import (
    ClickEventRequest,
    EventBatchRequest,
    ImpressionEventRequest,
    SearchEventRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/events", tags=["events"])


EventRequest = ClickEventRequest | ImpressionEventRequest | SearchEventRequest

_TOPIC_BY_TYPE: dict[str, str] = {
    "click": USER_CLICKS.name,
    "impression": USER_IMPRESSIONS.name,
    "search": USER_SEARCHES.name,
}


async def _ingest_one(req: EventRequest, topic: str) -> dict[str, str]:
    event_id = req.event_id or uuid7()
    server_ts = datetime.now(tz=UTC).isoformat().replace("+00:00", "Z")
    payload = req.model_dump(mode="json")
    payload["event_id"] = str(event_id)
    payload["server_ts"] = server_ts

    if not await dedupe_event_id(event_id):
        return {"event_id": str(event_id), "status": "duplicate"}

    try:
        await publish_event(topic, key=req.user_id, event=payload)
    except (KafkaConnectionError, KafkaTimeoutError) as exc:
        # Release the claim so the client's retry can actually publish —
        # otherwise the key lives for `dedupe_ttl_seconds` and every retry
        # returns `duplicate` without reaching Kafka.
        await release_event_id(event_id)
        logger.exception("kafka publish failed", extra={"event_id": str(event_id), "topic": topic})
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="kafka unavailable",
            headers={"Retry-After": "1"},
        ) from exc

    return {"event_id": str(event_id), "status": "accepted"}


@router.post("/click", status_code=status.HTTP_202_ACCEPTED)
async def ingest_click(req: ClickEventRequest) -> dict[str, str]:
    return await _ingest_one(req, USER_CLICKS.name)


@router.post("/impression", status_code=status.HTTP_202_ACCEPTED)
async def ingest_impression(req: ImpressionEventRequest) -> dict[str, str]:
    return await _ingest_one(req, USER_IMPRESSIONS.name)


@router.post("/search", status_code=status.HTTP_202_ACCEPTED)
async def ingest_search(req: SearchEventRequest) -> dict[str, str]:
    return await _ingest_one(req, USER_SEARCHES.name)


def _batch_error(ev: EventRequest, detail: str) -> dict[str, str]:
    return {
        "event_id": str(ev.event_id) if ev.event_id else "",
        "status": "error",
        "detail": detail,
    }


async def _ingest_group(
    items: list[tuple[int, EventRequest]],
) -> list[tuple[int, dict[str, str]]]:
    """Process all events for a single `user_id` sequentially.

    Serialising within a key preserves the per-user ordering contract
    (plan §6 / §9) — two concurrent `send_and_wait` calls on the same key
    can otherwise race at the broker.
    """
    results: list[tuple[int, dict[str, str]]] = []
    for idx, ev in items:
        topic = _TOPIC_BY_TYPE[ev.event_type]
        try:
            res = await _ingest_one(ev, topic)
        except HTTPException as exc:
            res = _batch_error(ev, str(exc.detail))
        except Exception as exc:
            logger.exception(
                "batch item failed unexpectedly",
                extra={"event_type": ev.event_type, "user_id": ev.user_id},
            )
            res = _batch_error(ev, f"internal error: {type(exc).__name__}")
        results.append((idx, res))
    return results


@router.post("/batch", status_code=status.HTTP_202_ACCEPTED)
async def ingest_batch(batch: EventBatchRequest) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[tuple[int, EventRequest]]] = {}
    for idx, ev in enumerate(batch.events):
        groups.setdefault(ev.user_id, []).append((idx, ev))

    group_results = await asyncio.gather(
        *(_ingest_group(items) for items in groups.values()),
        return_exceptions=True,
    )

    flat: list[tuple[int, dict[str, str]]] = []
    for group, (user_id, items) in zip(group_results, groups.items(), strict=True):
        if isinstance(group, BaseException):
            logger.exception(
                "batch group failed unexpectedly",
                exc_info=group,
                extra={"user_id": user_id},
            )
            for idx, ev in items:
                flat.append((idx, _batch_error(ev, f"internal error: {type(group).__name__}")))
        else:
            flat.extend(group)

    flat.sort(key=lambda p: p[0])
    return {"results": [res for _, res in flat]}
