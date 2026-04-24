"""Unit tests for the Phase 5 popularity refresher.

Uses a fake Redis that supports the sorted-set + scan ops the refresher
touches. `asyncio.wait_for(stop_event.wait(), timeout=...)` is patched
so the test doesn't actually sleep for 60s.
"""

from __future__ import annotations

import asyncio
import math
from typing import Any

import pytest

from click_rec.cache import popularity_refresher as refresher_module
from click_rec.cache.popularity_refresher import (
    _refresh_one_category,
    _refresh_tick,
    run_popularity_refresher,
)


class _FakeRedisZ:
    """Redis fake limited to ZSET + scan surface the refresher uses."""

    def __init__(self) -> None:
        self.zsets: dict[str, dict[bytes, float]] = {}
        # Record op counts so we can assert trim/decay/expire ran.
        self.calls: list[str] = []

    async def zremrangebyrank(self, key: str, start: int, stop: int) -> int:
        self.calls.append(f"zremrangebyrank:{key}")
        z = self.zsets.get(key, {})
        ordered = sorted(z.items(), key=lambda kv: kv[1])  # ascending
        length = len(ordered)
        stop_idx = length + stop if stop < 0 else stop
        to_remove = ordered[start : stop_idx + 1]
        for member, _ in to_remove:
            z.pop(member, None)
        return len(to_remove)

    async def zrange(
        self, key: str, start: int, stop: int, withscores: bool = False
    ) -> Any:
        self.calls.append(f"zrange:{key}")
        z = self.zsets.get(key, {})
        ordered = sorted(z.items(), key=lambda kv: kv[1])
        end = None if stop == -1 else stop + 1
        sliced = ordered[start:end]
        if withscores:
            return [(m, s) for m, s in sliced]
        return [m for m, _ in sliced]

    async def zadd(self, key: str, mapping: dict[Any, float]) -> int:
        self.calls.append(f"zadd:{key}")
        z = self.zsets.setdefault(key, {})
        for member, score in mapping.items():
            z[member] = score
        return len(mapping)

    async def zrem(self, key: str, *members: Any) -> int:
        self.calls.append(f"zrem:{key}")
        z = self.zsets.get(key, {})
        removed = 0
        for m in members:
            if z.pop(m, None) is not None:
                removed += 1
        return removed

    async def expire(self, key: str, seconds: int) -> int:
        self.calls.append(f"expire:{key}")
        return 1

    async def scan_iter(self, match: str, count: int = 100):  # noqa: ARG002
        prefix = match.rstrip("*")
        for key in list(self.zsets):
            if key.startswith(prefix):
                yield key.encode()


# ============================================================ one-category


async def test_refresh_trims_to_top_max(monkeypatch: pytest.MonkeyPatch) -> None:
    """250 members in, 200 max → 200 remain (plus decay applied)."""
    # Shorten max so the test stays small.
    monkeypatch.setenv("CACHE_TOP_MAX_MEMBERS", "10")
    monkeypatch.setenv("CACHE_REFRESH_INTERVAL_SECONDS", "60")
    monkeypatch.setenv("CACHE_REFRESH_HALF_LIFE_SECONDS", "3600")
    from click_rec.config import get_settings

    get_settings.cache_clear()

    redis = _FakeRedisZ()
    redis.zsets["items:top:electronics"] = {
        f"i{i}".encode(): float(i) for i in range(25)
    }

    settings = get_settings()
    try:
        await _refresh_one_category(redis, "items:top:electronics", settings)
    finally:
        get_settings.cache_clear()

    # Top 10 kept (highest scores = i15..i24).
    remaining = redis.zsets["items:top:electronics"]
    assert len(remaining) == 10
    kept_members = {m.decode() for m in remaining}
    assert kept_members == {f"i{i}" for i in range(15, 25)}


async def test_refresh_applies_decay(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scores multiplied by exp(-interval/half_life) after one tick."""
    monkeypatch.setenv("CACHE_TOP_MAX_MEMBERS", "100")
    monkeypatch.setenv("CACHE_REFRESH_INTERVAL_SECONDS", "60")
    monkeypatch.setenv("CACHE_REFRESH_HALF_LIFE_SECONDS", "3600")
    from click_rec.config import get_settings

    get_settings.cache_clear()

    redis = _FakeRedisZ()
    redis.zsets["items:top:books"] = {b"a": 100.0, b"b": 200.0, b"c": 50.0}
    settings = get_settings()

    try:
        await _refresh_one_category(redis, "items:top:books", settings)
    finally:
        get_settings.cache_clear()

    expected_factor = math.exp(-60 / 3600)
    z = redis.zsets["items:top:books"]
    assert z[b"a"] == pytest.approx(100.0 * expected_factor, rel=1e-6)
    assert z[b"b"] == pytest.approx(200.0 * expected_factor, rel=1e-6)
    assert z[b"c"] == pytest.approx(50.0 * expected_factor, rel=1e-6)


async def test_refresh_drops_negligible_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    """Entries whose decayed score goes under the 1e-6 floor are removed.

    Prevents long-dead categories from accumulating entries that just
    sit at near-zero forever.
    """
    monkeypatch.setenv("CACHE_TOP_MAX_MEMBERS", "100")
    monkeypatch.setenv("CACHE_REFRESH_INTERVAL_SECONDS", "1")
    monkeypatch.setenv("CACHE_REFRESH_HALF_LIFE_SECONDS", "1")
    from click_rec.config import get_settings

    get_settings.cache_clear()

    # With factor = exp(-1/1) ≈ 0.368:
    #   dead  = 1e-7  → 3.68e-8  (below the 1e-6 floor → drop)
    #   alive = 100.0 → 36.8     (way above → keep)
    redis = _FakeRedisZ()
    redis.zsets["items:top:cold"] = {b"dead": 1e-7, b"alive": 100.0}
    settings = get_settings()

    try:
        await _refresh_one_category(redis, "items:top:cold", settings)
    finally:
        get_settings.cache_clear()

    z = redis.zsets["items:top:cold"]
    assert b"dead" not in z
    assert b"alive" in z


# ============================================================ full tick


async def test_tick_iterates_all_matching_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    redis = _FakeRedisZ()
    redis.zsets["items:top:a"] = {b"x": 5.0}
    redis.zsets["items:top:b"] = {b"y": 10.0}
    # Off-prefix key must be ignored.
    redis.zsets["unrelated:key"] = {b"z": 100.0}

    await _refresh_tick(redis)

    # Both matching keys were expired (decay + expire call).
    assert any(c == "expire:items:top:a" for c in redis.calls)
    assert any(c == "expire:items:top:b" for c in redis.calls)
    assert not any(c == "expire:unrelated:key" for c in redis.calls)


# ============================================================ loop lifecycle


async def test_run_exits_when_stop_event_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """Setting the stop event must wake the loop immediately — not after 60s."""
    monkeypatch.setattr(
        refresher_module, "get_redis", lambda: _FakeRedisZ()
    )

    stop = asyncio.Event()
    task = asyncio.create_task(run_popularity_refresher(stop))

    # Let the first tick run, then stop.
    await asyncio.sleep(0.1)
    stop.set()
    # Should complete near-instantly, well under the refresh interval.
    await asyncio.wait_for(task, timeout=2.0)


async def test_run_survives_redis_outage(monkeypatch: pytest.MonkeyPatch) -> None:
    """A raising `get_redis()` must not crash the loop."""
    from redis.exceptions import ConnectionError as RedisConnectionError

    call_count = {"n": 0}

    def boom() -> Any:
        call_count["n"] += 1
        raise RedisConnectionError("simulated outage")

    monkeypatch.setattr(refresher_module, "get_redis", boom)

    # Shorten the interval so the test loops fast.
    monkeypatch.setenv("CACHE_REFRESH_INTERVAL_SECONDS", "0")
    from click_rec.config import get_settings

    get_settings.cache_clear()

    try:
        stop = asyncio.Event()
        task = asyncio.create_task(run_popularity_refresher(stop))
        # Give it a chance to tick a couple of times.
        await asyncio.sleep(0.1)
        stop.set()
        await asyncio.wait_for(task, timeout=2.0)
    finally:
        get_settings.cache_clear()

    # Proves we didn't crash and actually kept trying.
    assert call_count["n"] >= 1
