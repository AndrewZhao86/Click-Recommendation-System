"""Periodic refresh of `items:top:{category}` sorted sets.

Runs inside the consumer pool as a single `asyncio.Task`. Every
`cache_refresh_interval_seconds` ticks, for each `items:top:*` key:

1. Trim to the top `cache_top_max_members` (`ZREMRANGEBYRANK 0 -201`).
2. Apply exponential decay: multiply every score by
   `exp(-interval / half_life)`. Inactive categories drift toward zero
   and eventually hit the TTL; active ones stay above the noise floor.
3. Refresh the safety TTL so a category that stops receiving traffic
   gets cleaned up by Redis rather than sitting forever.

Fail-open: any Redis error logs `cache_unavailable` and the loop
continues on the next tick. No backoff — we'd rather try again in a
minute than accumulate retries that stall the task.

The decay is computed in Python and written back via `ZADD` (not
`ZADD GT`) — real Redis doesn't shrink scores with `GT`, and the whole
point of decay is to shrink. We accept the race: a concurrent
`ZINCRBY` between read and write may be partially overwritten, but the
next click's `ZINCRBY` immediately re-reflects. For a popularity
counter that's perfectly fine.
"""

from __future__ import annotations

import asyncio
import logging
import math
from typing import Any

import redis.asyncio as redis
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import RedisError
from redis.exceptions import TimeoutError as RedisTimeoutError

from click_rec.cache.redis_client import get_redis
from click_rec.config import get_settings

logger = logging.getLogger(__name__)

_REDIS_ERRORS = (RedisConnectionError, RedisTimeoutError, RedisError)
_TOP_KEY_PREFIX = "items:top:"


async def _refresh_one_category(client: redis.Redis, key: str, settings: Any) -> None:
    """Trim + decay + TTL-refresh a single `items:top:{category}` key."""
    decay = math.exp(
        -settings.cache_refresh_interval_seconds / settings.cache_refresh_half_life_seconds
    )
    trim_stop = -(settings.cache_top_max_members + 1)

    # Trim first — cheaper than decaying members that are about to be
    # dropped. `0 -201` removes everything except the top 200.
    await client.zremrangebyrank(key, 0, trim_stop)

    # Read remaining members with scores for decay.
    members = await client.zrange(key, 0, -1, withscores=True)
    if members:
        decayed: dict[bytes | str, float] = {}
        for raw_member, score in members:
            new_score = float(score) * decay
            # Drop members whose decayed score is effectively zero;
            # otherwise a long-idle category accumulates dead entries.
            if new_score > 1e-6:
                decayed[raw_member] = new_score
        if decayed:
            await client.zadd(key, decayed)
        # Members whose decayed score didn't clear the floor need to go.
        to_remove = [m for m, _ in members if m not in decayed]
        if to_remove:
            await client.zrem(key, *to_remove)

    await client.expire(key, settings.cache_top_category_ttl_seconds)


async def _refresh_tick(client: redis.Redis) -> None:
    settings = get_settings()
    async for raw_key in client.scan_iter(match=f"{_TOP_KEY_PREFIX}*", count=100):
        key = raw_key.decode() if isinstance(raw_key, bytes) else raw_key
        try:
            await _refresh_one_category(client, key, settings)
        except _REDIS_ERRORS as exc:
            # One bad key shouldn't kill the whole tick.
            logger.warning(
                "cache_unavailable: popularity refresh failed for key",
                extra={"key": key, "error": str(exc)},
            )


async def run_popularity_refresher(stop_event: asyncio.Event) -> None:
    """Run the refresh tick every `cache_refresh_interval_seconds` until stop.

    Waits on the stop event instead of a plain `asyncio.sleep` so
    shutdown from `run_consumer_pool` wakes us up immediately rather
    than leaving the task idle for up to a minute.
    """
    settings = get_settings()
    interval = settings.cache_refresh_interval_seconds
    logger.info("popularity refresher started (interval=%ds)", interval)

    while not stop_event.is_set():
        try:
            client = get_redis()
            await _refresh_tick(client)
        except _REDIS_ERRORS as exc:
            logger.warning(
                "cache_unavailable: popularity refresh tick failed",
                extra={"error": str(exc)},
            )
        except RuntimeError:
            # Singleton not started — possible during racy shutdown.
            logger.warning("popularity refresher: redis singleton missing, skipping tick")

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
        except TimeoutError:
            continue

    logger.info("popularity refresher stopped")
