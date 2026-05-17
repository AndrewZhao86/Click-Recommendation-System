"""Unit tests for Phase 5 cache-aside with stampede protection.

A fake Redis implements only the surface `cache_aside.py` touches: GET,
SET (with `nx` / `ex` / `px`), DELETE, and EVAL (for the ownership
check). A fake SQLAlchemy session records `SELECT` calls so we can
assert the stampede test only hits Postgres once.
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from typing import Any

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

from click_rec.cache import cache_aside as cache_aside_module
from click_rec.cache.cache_aside import get_item_cached
from click_rec.models.schemas import ItemDTO

# ------------------------------------------------------------------ fakes


class _FakeRedisKV:
    """Minimal Redis fake for the cache-aside flow.

    Supports only what `cache_aside.py` uses:
      - `get(key)`
      - `set(key, value, nx=..., ex=..., px=...)`
      - `delete(key)`
      - `eval(script, numkeys, *keys_and_args)` — release-lock Lua
    """

    def __init__(self) -> None:
        self.kv: dict[str, bytes] = {}
        self._expires_at: dict[str, float] = {}
        # Exposed for tests that want to assert op counts / ordering.
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        # Test knobs: when set, the matching op raises.
        self.fail_on: set[str] = set()

    def _expire(self, key: str) -> None:
        exp = self._expires_at.get(key)
        if exp is not None and time.monotonic() > exp:
            self.kv.pop(key, None)
            self._expires_at.pop(key, None)

    async def get(self, key: str) -> bytes | None:
        self.calls.append(("get", (key,)))
        if "get" in self.fail_on:
            raise RedisConnectionError("fake get failure")
        self._expire(key)
        return self.kv.get(key)

    async def set(
        self,
        key: str,
        value: bytes,
        nx: bool = False,
        ex: int | None = None,
        px: int | None = None,
    ) -> bool | None:
        self.calls.append(("set", (key, value, nx, ex, px)))
        if "set" in self.fail_on:
            raise RedisConnectionError("fake set failure")
        if nx and key in self.kv:
            return None
        self.kv[key] = value
        if ex is not None:
            self._expires_at[key] = time.monotonic() + ex
        if px is not None:
            self._expires_at[key] = time.monotonic() + (px / 1000.0)
        return True

    async def delete(self, *keys: str) -> int:
        self.calls.append(("delete", keys))
        removed = 0
        for k in keys:
            if self.kv.pop(k, None) is not None:
                removed += 1
            self._expires_at.pop(k, None)
        return removed

    async def eval(self, script: str, numkeys: int, *keys_and_args: Any) -> int:
        self.calls.append(("eval", (script, numkeys, *keys_and_args)))
        # Minimal interpretation of the release-lock Lua: GET key, if
        # equals argv[1] then DEL it.
        key = keys_and_args[0]
        expected = keys_and_args[numkeys]
        current = self.kv.get(key)
        if current == expected:
            self.kv.pop(key, None)
            self._expires_at.pop(key, None)
            return 1
        return 0


class _FakeSession:
    """Async-session fake for items. Counts select calls."""

    def __init__(self, rows: dict[str, ItemDTO]) -> None:
        self.rows = rows
        self.select_count = 0

    async def execute(self, stmt: Any) -> Any:
        self.select_count += 1
        # Pull the literal item_id out of the WHERE clause by walking
        # the compiled bind parameters — avoids depending on real SQL.
        try:
            compiled = stmt.compile(compile_kwargs={"literal_binds": True})
            rendered = str(compiled)
        except Exception:
            rendered = str(stmt)

        matched_id: str | None = None
        for row_id in self.rows:
            if f"'{row_id}'" in rendered or f'"{row_id}"' in rendered:
                matched_id = row_id
                break

        dto = self.rows.get(matched_id) if matched_id else None

        class _Result:
            def __init__(self, val: ItemDTO | None) -> None:
                self._val = val

            def scalar_one_or_none(self_inner) -> ItemDTO | None:
                return self_inner._val

        return _Result(dto)


def _make_item(item_id: str = "i1") -> ItemDTO:
    return ItemDTO(
        id=item_id,
        title=f"Title {item_id}",
        description="desc",
        category="electronics/headphones",
        brand="Acme",
        price=99.99,
        created_at=datetime(2026, 4, 22, 10, 0, 0, tzinfo=UTC),
        popularity_score=0.0,
    )


# Every test lives inside this monkeypatch — cache_aside.get_redis()
# is what the module under test calls, and we replace the singleton
# with the fake on a per-test basis.
@pytest.fixture
def redis_and_patch(monkeypatch: pytest.MonkeyPatch) -> _FakeRedisKV:
    fake = _FakeRedisKV()
    monkeypatch.setattr(cache_aside_module, "get_redis", lambda: fake)
    return fake


# ============================================================ hit path


async def test_hit_path_does_not_touch_postgres(redis_and_patch: _FakeRedisKV) -> None:
    """Pre-populated cache → one GET, no DB call."""
    import orjson

    item = _make_item("i_hit")
    redis_and_patch.kv["item:i_hit"] = orjson.dumps(item.model_dump(mode="json"))

    session = _FakeSession({"i_hit": item})
    result = await get_item_cached("i_hit", session)  # type: ignore[arg-type]

    assert result is not None
    assert result.id == "i_hit"
    assert session.select_count == 0


async def test_null_sentinel_is_a_hit_and_returns_none(redis_and_patch: _FakeRedisKV) -> None:
    """A cached `null` sentinel must short-circuit — no DB call."""
    redis_and_patch.kv["item:i_missing"] = b"null"
    session = _FakeSession({})
    result = await get_item_cached("i_missing", session)  # type: ignore[arg-type]
    assert result is None
    assert session.select_count == 0


# ============================================================ miss path


async def test_miss_populates_cache_and_reads_db_once(redis_and_patch: _FakeRedisKV) -> None:
    item = _make_item("i_miss")
    session = _FakeSession({"i_miss": item})

    result = await get_item_cached("i_miss", session)  # type: ignore[arg-type]
    assert result is not None and result.id == "i_miss"
    assert session.select_count == 1
    # Cache written with a JSON payload.
    assert "item:i_miss" in redis_and_patch.kv
    assert redis_and_patch.kv["item:i_miss"] != b"null"
    # Lock key released.
    assert "lock:item:i_miss" not in redis_and_patch.kv


async def test_miss_with_unknown_item_caches_null_sentinel(redis_and_patch: _FakeRedisKV) -> None:
    """Negative caching: an unknown item must write `null` to guard Postgres."""
    session = _FakeSession({})  # no rows

    result = await get_item_cached("i_nope", session)  # type: ignore[arg-type]
    assert result is None
    assert session.select_count == 1
    assert redis_and_patch.kv.get("item:i_nope") == b"null"


# ============================================================ stampede


async def test_stampede_100_concurrent_callers_select_once(
    monkeypatch: pytest.MonkeyPatch, redis_and_patch: _FakeRedisKV
) -> None:
    """100 concurrent gets on a cold key → exactly 1 Postgres SELECT.

    The 99 losers either poll until the winner fills the cache, or —
    because asyncio cooperative scheduling gives the winner a chance to
    finish before any loser polls — may return from their first post-
    wait GET. Either way: **exactly one DB read**.
    """
    item = _make_item("i_stampede")
    session = _FakeSession({"i_stampede": item})

    # Keep poll interval tiny so losers wake up quickly.
    from click_rec.config import get_settings

    monkeypatch.setenv("CACHE_LOCK_WAIT_POLL_MS", "1")
    monkeypatch.setenv("CACHE_LOCK_WAIT_MAX_MS", "2000")
    get_settings.cache_clear()

    try:
        results = await asyncio.gather(
            *(get_item_cached("i_stampede", session) for _ in range(100))  # type: ignore[arg-type]
        )
    finally:
        get_settings.cache_clear()

    assert all(r is not None and r.id == "i_stampede" for r in results)
    assert session.select_count == 1, f"expected exactly 1 DB read, got {session.select_count}"


# ============================================================ graceful degradation


async def test_redis_get_failure_falls_back_to_postgres(
    redis_and_patch: _FakeRedisKV,
) -> None:
    """`GET` raises → DB fallback, no 5xx surfaced, request completes."""
    redis_and_patch.fail_on.add("get")
    item = _make_item("i_degrade")
    session = _FakeSession({"i_degrade": item})

    result = await get_item_cached("i_degrade", session)  # type: ignore[arg-type]
    assert result is not None and result.id == "i_degrade"
    assert session.select_count == 1


async def test_redis_set_failure_still_returns_db_value(
    redis_and_patch: _FakeRedisKV,
) -> None:
    """Lock `SET` raises → DB fallback, caller still gets the value."""
    redis_and_patch.fail_on.add("set")
    item = _make_item("i_set_fail")
    session = _FakeSession({"i_set_fail": item})

    result = await get_item_cached("i_set_fail", session)  # type: ignore[arg-type]
    assert result is not None and result.id == "i_set_fail"


# ============================================================ lock semantics


async def test_lock_timeout_falls_through_to_db(
    monkeypatch: pytest.MonkeyPatch, redis_and_patch: _FakeRedisKV
) -> None:
    """Losers whose filler crashed (lock held but key never set) must not
    wait forever — they fall through to a direct DB read after timeout.
    """
    # Pre-hold the lock so every caller in this test loses it.
    redis_and_patch.kv["lock:item:i_timeout"] = b"someone_else"

    # Tiny timeout so the test is fast.
    monkeypatch.setenv("CACHE_LOCK_WAIT_MAX_MS", "40")
    monkeypatch.setenv("CACHE_LOCK_WAIT_POLL_MS", "5")
    from click_rec.config import get_settings

    get_settings.cache_clear()

    item = _make_item("i_timeout")
    session = _FakeSession({"i_timeout": item})
    try:
        result = await get_item_cached("i_timeout", session)  # type: ignore[arg-type]
    finally:
        get_settings.cache_clear()

    assert result is not None and result.id == "i_timeout"
    # One DB read (the fallback), and NO cache population (we don't own
    # the lock, so we can't risk racing the legitimate filler).
    assert session.select_count == 1
    assert "item:i_timeout" not in redis_and_patch.kv


async def test_lock_is_released_after_fill(redis_and_patch: _FakeRedisKV) -> None:
    item = _make_item("i_release")
    session = _FakeSession({"i_release": item})
    await get_item_cached("i_release", session)  # type: ignore[arg-type]
    assert "lock:item:i_release" not in redis_and_patch.kv
