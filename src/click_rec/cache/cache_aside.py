"""Stampede-protected cache-aside for item lookups.

Flow (plan §8 Phase 5, step 1):

    GET item:{id}  ──► HIT  → deserialize → cache_hit_total++
                      │
                      └── MISS → SET lock:item:{id} <caller> NX PX 2000
                                 │
                                 ├── won: SELECT from Postgres
                                 │        SETEX item:{id}  (or null sentinel)
                                 │        DEL-if-owner lock:item:{id}  (Lua)
                                 │
                                 └── lost: poll GET item:{id} (20ms up to 1.5s)
                                           └── timeout → DB fallback (no cache write)

On any Redis error (`RedisConnectionError`, `RedisTimeoutError`, `RedisError`)
we log `cache_unavailable`, increment `cache_unavailable_total`, and fall
through to Postgres. Matches the fail-open pattern in
`redis_client.dedupe_event_id`.

Negative caching uses a `b"null"` sentinel with `cache_negative_ttl_seconds`
so a flood of requests for a deleted/unknown item doesn't pound Postgres
(same motivation as the enrichment-side negative cache in
`kafka/enrichment.py`, but API-side and bounded by TTL instead of LRU).
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any

import orjson
import redis.asyncio as redis
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import RedisError
from redis.exceptions import TimeoutError as RedisTimeoutError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from click_rec.cache.redis_client import get_redis
from click_rec.config import get_settings
from click_rec.models.item import Item
from click_rec.models.schemas import ItemDTO
from click_rec.telemetry.metrics import (
    cache_hit_total,
    cache_lock_wait_seconds,
    cache_miss_total,
    cache_unavailable_total,
)

logger = logging.getLogger(__name__)

_REDIS_ERRORS = (RedisConnectionError, RedisTimeoutError, RedisError)

# Lua-scripted release: delete the lock *only* if we still own it.
# Without this, a caller whose fill took longer than `cache_lock_ttl_ms`
# could expire, a second caller wins the lock, and the first caller's
# blind DEL would release someone else's lock — leaking the stampede
# protection to a third caller.
_RELEASE_LOCK_LUA = """
if redis.call("GET", KEYS[1]) == ARGV[1] then
    return redis.call("DEL", KEYS[1])
else
    return 0
end
"""

_NULL_SENTINEL = b"null"


def _item_key(item_id: str) -> str:
    return f"item:{item_id}"


def _lock_key(item_id: str) -> str:
    return f"lock:item:{item_id}"


def _deserialize_item(raw: bytes) -> ItemDTO | None:
    """Decode the cached JSON blob back into an ItemDTO.

    Returns None for the null sentinel (known-missing item).
    """
    if raw == _NULL_SENTINEL:
        return None
    return ItemDTO.model_validate(orjson.loads(raw))


def _serialize_item(dto: ItemDTO) -> bytes:
    return orjson.dumps(dto.model_dump(mode="json"))


async def _fetch_from_db(session: AsyncSession, item_id: str) -> ItemDTO | None:
    row = (await session.execute(select(Item).where(Item.id == item_id))).scalar_one_or_none()
    if row is None:
        return None
    return ItemDTO.model_validate(row)


async def _try_get_cached(client: redis.Redis, item_id: str) -> tuple[bool, ItemDTO | None]:
    """Return (present, dto_or_none).

    `present=True, dto=None` means the null sentinel was cached — a
    previously-observed miss that we must honour to protect Postgres.
    `present=False` means the key wasn't in Redis.
    """
    raw = await client.get(_item_key(item_id))
    if raw is None:
        return False, None
    return True, _deserialize_item(raw)


async def _wait_for_filler(
    client: redis.Redis, item_id: str, max_wait_ms: int, poll_ms: int
) -> tuple[bool, ItemDTO | None]:
    """Poll the cache key while another caller holds the fill lock.

    Returns `(True, dto)` once the filler has populated the key, or
    `(False, None)` after `max_wait_ms` — in which case the caller falls
    through to a direct DB read (no cache write) so traffic isn't
    wedged waiting on a crashed filler.
    """
    deadline = time.monotonic() + (max_wait_ms / 1000.0)
    poll_s = poll_ms / 1000.0
    while time.monotonic() < deadline:
        await asyncio.sleep(poll_s)
        present, dto = await _try_get_cached(client, item_id)
        if present:
            return True, dto
    return False, None


async def get_item_cached(item_id: str, session: AsyncSession) -> ItemDTO | None:
    """Return an item, using Redis cache-aside with stampede protection.

    Contract:
    - Hit (including a cached null sentinel) never touches Postgres.
    - Under N concurrent callers for a cold key, exactly one executes
      the Postgres SELECT; the rest wait for the cache fill.
    - Any Redis error falls back to a direct Postgres read and logs
      `cache_unavailable`. The caller never sees a 5xx from a Redis
      outage.
    """
    settings = get_settings()
    try:
        client = get_redis()
    except RuntimeError:
        # Redis singleton wasn't started (e.g. test harness). Serving
        # directly from Postgres is still correct; it just skips caching.
        return await _fetch_from_db(session, item_id)

    # 1. Fast path: a plain GET.
    try:
        present, dto = await _try_get_cached(client, item_id)
    except _REDIS_ERRORS as exc:
        logger.warning(
            "cache_unavailable: item GET failed, falling back to Postgres",
            extra={"item_id": item_id, "error": str(exc)},
        )
        cache_unavailable_total.labels(operation="get").inc()
        return await _fetch_from_db(session, item_id)

    if present:
        # `dto is None` means we hit the negative sentinel — still a DB
        # bypass, but worth splitting in the metric so 404-bypass rate is
        # separately observable from real value hits.
        status = "null" if dto is None else "value"
        cache_hit_total.labels(key_type="item", status=status).inc()
        return dto

    cache_miss_total.labels(key_type="item").inc()

    # 2. Miss path: try to acquire the stampede lock.
    caller_id = uuid.uuid4().hex.encode()
    try:
        acquired = await client.set(
            _lock_key(item_id), caller_id, nx=True, px=settings.cache_lock_ttl_ms
        )
    except _REDIS_ERRORS as exc:
        logger.warning(
            "cache_unavailable: lock SET failed, falling back to Postgres",
            extra={"item_id": item_id, "error": str(exc)},
        )
        cache_unavailable_total.labels(operation="lock").inc()
        return await _fetch_from_db(session, item_id)

    if not acquired:
        # Someone else is filling — wait for them.
        wait_start = time.monotonic()
        try:
            filled, dto = await _wait_for_filler(
                client,
                item_id,
                max_wait_ms=settings.cache_lock_wait_max_ms,
                poll_ms=settings.cache_lock_wait_poll_ms,
            )
        except _REDIS_ERRORS as exc:
            logger.warning(
                "cache_unavailable: lock wait failed, falling back to Postgres",
                extra={"item_id": item_id, "error": str(exc)},
            )
            cache_unavailable_total.labels(operation="lock_wait").inc()
            return await _fetch_from_db(session, item_id)
        cache_lock_wait_seconds.observe(time.monotonic() - wait_start)
        if filled:
            return dto
        # Timed out waiting: the original filler probably crashed. Read
        # through without populating — whoever wins the next lock cycle
        # will repopulate.
        logger.warning(
            "cache_stampede_timeout: fell through to DB",
            extra={"item_id": item_id},
        )
        return await _fetch_from_db(session, item_id)

    # 3. We hold the lock: read Postgres, populate cache, release lock.
    try:
        dto = await _fetch_from_db(session, item_id)
        await _populate_cache(client, item_id, dto, settings)
    finally:
        await _release_lock(client, item_id, caller_id)

    return dto


async def _populate_cache(
    client: redis.Redis,
    item_id: str,
    dto: ItemDTO | None,
    settings: Any,
) -> None:
    """Write the DTO (or null sentinel) into Redis with the right TTL.

    Fail-open: if Redis errors here, we still return the DB value to
    the caller — the cache just stays cold.
    """
    if dto is None:
        payload = _NULL_SENTINEL
        ttl = settings.cache_negative_ttl_seconds
    else:
        payload = _serialize_item(dto)
        ttl = settings.cache_item_ttl_seconds
    try:
        await client.set(_item_key(item_id), payload, ex=ttl)
    except _REDIS_ERRORS as exc:
        logger.warning(
            "cache_unavailable: item SET failed",
            extra={"item_id": item_id, "error": str(exc)},
        )
        cache_unavailable_total.labels(operation="set").inc()


async def _release_lock(client: redis.Redis, item_id: str, caller_id: bytes) -> None:
    """Release the fill lock iff we still own it.

    Uses a Lua script so the GET-and-DEL is atomic. See
    `_RELEASE_LOCK_LUA` above for the ownership-leak risk this guards.
    """
    try:
        await client.eval(_RELEASE_LOCK_LUA, 1, _lock_key(item_id), caller_id)  # type: ignore[misc]
    except _REDIS_ERRORS as exc:
        # Lock release is best-effort; the key has a PX TTL that will
        # reclaim it if we fail here.
        logger.warning(
            "cache_unavailable: lock release failed",
            extra={"item_id": item_id, "error": str(exc)},
        )
        cache_unavailable_total.labels(operation="release").inc()
