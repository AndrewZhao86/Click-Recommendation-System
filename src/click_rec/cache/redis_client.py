"""Async Redis singleton + `event_id` idempotency helper.

The dedupe helper is fail-open: on Redis outage we return True (publish
anyway). Phase 3's dedup exists to absorb client retries, not to guarantee
correctness — the enrichment consumer (Phase 4a/4b) is idempotent on
`event_id`, so a few extra messages on the topic do not break downstream
state.
"""

from __future__ import annotations

import logging
from uuid import UUID

import redis.asyncio as redis
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import RedisError
from redis.exceptions import TimeoutError as RedisTimeoutError

from click_rec.config import get_settings

logger = logging.getLogger(__name__)

_redis: redis.Redis | None = None


async def start_redis() -> redis.Redis:
    global _redis
    if _redis is not None:
        return _redis
    settings = get_settings()
    client: redis.Redis = redis.from_url(settings.redis_url, decode_responses=False)
    await client.ping()  # type: ignore[misc]
    _redis = client
    logger.info("redis client started")
    return client


async def stop_redis() -> None:
    global _redis
    if _redis is None:
        return
    try:
        await _redis.aclose()
    finally:
        _redis = None
        logger.info("redis client stopped")


def get_redis() -> redis.Redis:
    if _redis is None:
        raise RuntimeError("redis not started — ensure FastAPI lifespan is wired")
    return _redis


async def dedupe_event_id(event_id: UUID) -> bool:
    """Claim an event_id for publication.

    Returns True when we hold the claim (key didn't exist — caller should
    publish), False when the key already existed (caller should skip and
    return `duplicate`). On any Redis error we fail-open and return True.
    """
    client = get_redis()
    ttl = get_settings().dedupe_ttl_seconds
    key = f"dedupe:event:{event_id}"
    try:
        result = await client.set(key, b"1", nx=True, ex=ttl)
    except (RedisConnectionError, RedisTimeoutError, RedisError) as exc:
        logger.warning(
            "cache_unavailable: redis dedupe failed, failing open",
            extra={"event_id": str(event_id), "error": str(exc)},
        )
        return True
    return bool(result)


async def release_event_id(event_id: UUID) -> None:
    """Release a dedupe claim so a client retry can re-publish.

    Called when publish fails after a successful claim — without this the
    key sits for `dedupe_ttl_seconds` and the retry returns `duplicate`
    without anything ever reaching Kafka. Fail-open: if Redis is down we
    swallow the error (the dedupe key is best-effort anyway).
    """
    client = get_redis()
    key = f"dedupe:event:{event_id}"
    try:
        await client.delete(key)
    except (RedisConnectionError, RedisTimeoutError, RedisError) as exc:
        logger.warning(
            "cache_unavailable: redis dedupe release failed",
            extra={"event_id": str(event_id), "error": str(exc)},
        )
