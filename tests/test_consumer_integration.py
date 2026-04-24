"""Integration tests for the Phase 4a/4b consumer pipeline.

Spins up Kafka + Redis + Postgres via testcontainers, runs the consumer
worker against real brokers and a real DB, and asserts:

- Happy path: a published click ends up in `co_click`, `recent_clicks`,
  the popularity counter, AND the `user.profile.updates` topic.
- Idempotency: replaying the same event twice yields a single co_click
  bump (Phase 4 review C4(a) success path + producer/consumer dedupe).
- Poison pill: malformed JSON ends up on `user.clicks.dlq` and the
  worker keeps consuming.
- DLQ replay: `replay_dlq.run()` re-publishes parseable rows and
  quarantines unparseable ones (Phase 4 review D4).
- Co-click `(N+1)`-th click: with N prior clicks the current click is
  *included* in pairs (Phase 4 review C1).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

pytestmark = pytest.mark.integration

try:
    import orjson
    import testcontainers  # noqa: F401
    from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
    from testcontainers.kafka import KafkaContainer
    from testcontainers.postgres import PostgresContainer
    from testcontainers.redis import RedisContainer
except ImportError:
    pytest.skip("testcontainers, aiokafka, or orjson not installed", allow_module_level=True)

from sqlalchemy import select  # noqa: E402
from uuid6 import uuid7  # noqa: E402

from click_rec.kafka.topics import (  # noqa: E402
    USER_CLICKS,
    USER_CLICKS_DLQ,
    USER_PROFILE_UPDATES,
)

# ---------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def kafka_container() -> Any:
    with KafkaContainer() as kc:
        yield kc


@pytest.fixture(scope="module")
def redis_container() -> Any:
    with RedisContainer() as rc:
        yield rc


@pytest.fixture(scope="module")
def postgres_container() -> Any:
    with PostgresContainer("pgvector/pgvector:pg16") as pc:
        yield pc


@pytest.fixture(scope="module")
def configured_settings(
    kafka_container: Any,
    redis_container: Any,
    postgres_container: Any,
) -> Iterator[None]:
    mp = pytest.MonkeyPatch()
    bootstrap = kafka_container.get_bootstrap_server()
    redis_host = redis_container.get_container_host_ip()
    redis_port = redis_container.get_exposed_port(6379)

    pg_url = postgres_container.get_connection_url().replace(
        "postgresql+psycopg2://", "postgresql+asyncpg://"
    )
    mp.setenv("KAFKA_BOOTSTRAP", bootstrap)
    mp.setenv("REDIS_URL", f"redis://{redis_host}:{redis_port}/0")
    mp.setenv("DATABASE_URL", pg_url)

    from click_rec.config import get_settings

    get_settings.cache_clear()
    yield
    mp.undo()
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _clear_item_cache() -> Iterator[None]:
    from click_rec.kafka import enrichment as enr_mod

    enr_mod._item_cache.clear()
    enr_mod._item_negative_cache.clear()
    yield
    enr_mod._item_cache.clear()
    enr_mod._item_negative_cache.clear()


@pytest.fixture
async def db_schema(configured_settings: None) -> AsyncIterator[None]:
    """Create the minimal Phase 4 tables (item, user_account, co_click)."""
    from sqlalchemy import text

    from click_rec.db.base import dispose_engine, get_engine
    from click_rec.models import Base

    engine = get_engine()
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await dispose_engine()


@pytest.fixture
async def seeded_items(db_schema: None) -> list[str]:
    """Insert two items so resolve_item_meta + co_click can succeed."""
    from sqlalchemy import insert

    from click_rec.db.base import get_sessionmaker
    from click_rec.models.item import Item

    sm = get_sessionmaker()
    item_ids = ["i_int_1", "i_int_2"]
    async with sm() as session:
        await session.execute(
            insert(Item),
            [
                {
                    "id": "i_int_1",
                    "title": "Wireless headphones X",
                    "description": "noise cancelling",
                    "category": "electronics/headphones",
                    "brand": "Acme",
                    "price": 99.99,
                },
                {
                    "id": "i_int_2",
                    "title": "Bluetooth earbuds Y",
                    "description": "sport",
                    "category": "electronics/headphones",
                    "brand": "Beta",
                    "price": 49.99,
                },
            ],
        )
        await session.commit()
    return item_ids


@pytest.fixture
async def started_redis(configured_settings: None) -> AsyncIterator[None]:
    from click_rec.cache.redis_client import start_redis, stop_redis

    await start_redis()
    yield
    await stop_redis()


@pytest.fixture
async def consumer_pool(
    kafka_container: Any,
    seeded_items: list[str],
    started_redis: None,
) -> AsyncIterator[asyncio.Task[Any]]:
    from click_rec.kafka.consumer import run_consumer_pool

    task = asyncio.create_task(run_consumer_pool(workers=1))
    # Give the worker a moment to join the group + subscribe.
    await asyncio.sleep(2.0)
    yield task
    task.cancel()
    import contextlib as _ctx
    with _ctx.suppress(asyncio.CancelledError, Exception):
        await task


# ---------------------------------------------------------------- helpers


def _click_payload(
    user_id: str,
    item_id: str,
    event_id: UUID | None = None,
) -> bytes:
    body = {
        "event_id": str(event_id or uuid7()),
        "event_type": "click",
        "event_version": 1,
        "timestamp": datetime.now(tz=UTC).isoformat().replace("+00:00", "Z"),
        "user_id": user_id,
        "session_id": "s_int",
        "query": "wireless",
        "item_id": item_id,
        "rank_position": 1,
        "dwell_ms": 3000,
    }
    return orjson.dumps(body)


async def _publish(bootstrap: str, topic: str, key: str | None, value: bytes) -> None:
    p = AIOKafkaProducer(
        bootstrap_servers=bootstrap,
        acks="all",
        enable_idempotence=True,
    )
    await p.start()
    try:
        await p.send_and_wait(topic, value, key=key.encode() if key else None)
    finally:
        await p.stop()


async def _drain(
    bootstrap: str, topic: str, timeout_s: float = 8.0
) -> list[dict[str, Any]]:
    c = AIOKafkaConsumer(
        topic,
        bootstrap_servers=bootstrap,
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        group_id=None,
    )
    await c.start()
    out: list[dict[str, Any]] = []
    try:
        deadline = asyncio.get_event_loop().time() + timeout_s
        while asyncio.get_event_loop().time() < deadline:
            batch = await c.getmany(timeout_ms=500)
            for _tp, records in batch.items():
                for r in records:
                    try:
                        out.append(orjson.loads(r.value))
                    except orjson.JSONDecodeError:
                        out.append({"_raw": r.value.decode("utf-8", errors="replace")})
    finally:
        await c.stop()
    return out


async def _wait_for(
    fn: Any, *, timeout_s: float = 8.0, interval_s: float = 0.25
) -> bool:
    deadline = asyncio.get_event_loop().time() + timeout_s
    while asyncio.get_event_loop().time() < deadline:
        if await fn():
            return True
        await asyncio.sleep(interval_s)
    return False


# =============================================================== happy path


async def test_happy_path_writes_redis_postgres_and_profile_topic(
    kafka_container: Any, consumer_pool: asyncio.Task[Any]
) -> None:
    bootstrap = kafka_container.get_bootstrap_server()
    user_id = "u_happy"

    # Click two distinct items so we can assert co_click + recent_clicks.
    await _publish(bootstrap, USER_CLICKS.name, user_id, _click_payload(user_id, "i_int_1"))
    await _publish(bootstrap, USER_CLICKS.name, user_id, _click_payload(user_id, "i_int_2"))

    from click_rec.cache.redis_client import get_redis
    from click_rec.db.base import get_sessionmaker
    from click_rec.models.co_click import CoClick

    async def co_click_present() -> bool:
        sm = get_sessionmaker()
        async with sm() as s:
            row = (
                await s.execute(
                    select(CoClick).where(
                        CoClick.item_a == "i_int_1", CoClick.item_b == "i_int_2"
                    )
                )
            ).first()
            return row is not None

    assert await _wait_for(co_click_present, timeout_s=15.0)

    # Redis recent_clicks contains both items.
    r = get_redis()
    members = await r.zrevrange(f"user:{user_id}:recent_clicks", 0, -1)
    assert {m.decode() for m in members} == {"i_int_1", "i_int_2"}

    # Profile-update topic received two events.
    profile_msgs = await _drain(bootstrap, USER_PROFILE_UPDATES.name)
    user_msgs = [m for m in profile_msgs if m.get("user_id") == user_id]
    assert len(user_msgs) >= 2


# =============================================================== idempotency


async def test_replaying_same_event_does_not_double_increment(
    kafka_container: Any, consumer_pool: asyncio.Task[Any]
) -> None:
    """C4(a) success path: a marker is set, replays short-circuit.

    Even without the marker, ZADD with the same score is idempotent, but
    the popularity INCR is *not* — the marker is what protects it.
    """
    from click_rec.cache.redis_client import get_redis

    bootstrap = kafka_container.get_bootstrap_server()
    user_id = "u_dup"
    eid = uuid7()
    payload = _click_payload(user_id, "i_int_1", event_id=eid)

    r = get_redis()
    cat_key = "popularity:electronics/headphones"
    before_raw = await r.get(cat_key)
    before = int(before_raw) if before_raw else 0

    # Send the exact same bytes twice.
    await _publish(bootstrap, USER_CLICKS.name, user_id, payload)
    await asyncio.sleep(2.0)
    await _publish(bootstrap, USER_CLICKS.name, user_id, payload)
    await asyncio.sleep(3.0)

    after_raw = await r.get(cat_key)
    after = int(after_raw) if after_raw else 0
    # Exactly one increment should have happened thanks to the marker.
    assert after - before == 1, f"expected exactly +1, got {after - before}"


# =============================================================== poison pill


async def test_poison_pill_goes_to_dlq_and_consumer_keeps_running(
    kafka_container: Any, consumer_pool: asyncio.Task[Any]
) -> None:
    bootstrap = kafka_container.get_bootstrap_server()

    # Garbage that JSON can't parse.
    await _publish(bootstrap, USER_CLICKS.name, "u_poison", b"\xde\xad\xbe\xef not json")
    # Then a valid click — the worker must still process it.
    valid_payload = _click_payload("u_poison_recover", "i_int_1")
    await _publish(bootstrap, USER_CLICKS.name, "u_poison_recover", valid_payload)

    dlq = await _drain(bootstrap, USER_CLICKS_DLQ.name, timeout_s=15.0)
    assert any(m.get("reason") == "parse_or_validation_error" for m in dlq), dlq

    from click_rec.cache.redis_client import get_redis

    r = get_redis()
    async def recovered() -> bool:
        members = await r.zrange(
            "user:u_poison_recover:recent_clicks", 0, -1
        )
        return len(members) >= 1

    assert await _wait_for(recovered, timeout_s=10.0)


# =============================================================== DLQ replay


async def test_replay_dlq_skips_malformed_payloads_and_replays_valid_ones(
    kafka_container: Any, consumer_pool: asyncio.Task[Any], tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Phase 4 review D4: poison pills must not crash replay_dlq."""
    bootstrap = kafka_container.get_bootstrap_server()

    # Trigger a poison-pill DLQ entry (raw string original_payload).
    await _publish(bootstrap, USER_CLICKS.name, "u_dlq_p", b"this is not json at all")

    # And a synthetic DLQ row that DOES have a valid `original_payload`
    # (the consumer would only do this if a real downstream side-effect
    # failed; we simulate it by publishing directly to the DLQ).
    valid_event_id = str(uuid7())
    fake_dlq = {
        "original_topic": USER_CLICKS.name,
        "original_partition": 0,
        "original_offset": 9999,
        "original_key": "u_dlq_v",
        "original_payload": {
            "event_id": valid_event_id,
            "event_type": "click",
            "event_version": 1,
            "timestamp": datetime.now(tz=UTC).isoformat().replace("+00:00", "Z"),
            "user_id": "u_dlq_v",
            "session_id": "s",
            "query": "x",
            "item_id": "i_int_1",
            "rank_position": 0,
        },
        "reason": "fake",
        "error_class": "Synthetic",
        "error_message": "test",
        "stack_trace": "",
        "failed_at_ms": 0,
    }
    await _publish(bootstrap, USER_CLICKS_DLQ.name, None, orjson.dumps(fake_dlq))

    # Wait so the consumer-emitted DLQ row appears too.
    await asyncio.sleep(3.0)

    quarantine = tmp_path / "quarantine.jsonl"
    monkeypatch.setattr("scripts.replay_dlq.QUARANTINE_PATH", quarantine)

    from scripts.replay_dlq import replay_dlq

    replayed, skipped = await replay_dlq(bootstrap=bootstrap, poll_timeout_ms=500)
    assert replayed >= 1
    assert skipped >= 1
    assert quarantine.exists()
    # The replayed event eventually reaches the main topic; consumer enriches it.
    from click_rec.cache.redis_client import get_redis

    r = get_redis()

    async def replayed_landed() -> bool:
        members = await r.zrange("user:u_dlq_v:recent_clicks", 0, -1)
        return len(members) >= 1

    assert await _wait_for(replayed_landed, timeout_s=15.0)


# =============================================================== C1 verification


async def test_sixth_click_is_included_in_co_click_pairs(
    kafka_container: Any, consumer_pool: asyncio.Task[Any], seeded_items: list[str],
) -> None:
    """Phase 4 review C1: with `session_window` priors, the current click
    must NOT be sliced off — co_click must contain the new pair.
    """
    from sqlalchemy import insert

    from click_rec.db.base import get_sessionmaker
    from click_rec.models.co_click import CoClick
    from click_rec.models.item import Item

    # Insert 5 extra items so we have 7 in total to walk through.
    sm = get_sessionmaker()
    extra_ids = [f"i_extra_{i}" for i in range(5)]
    async with sm() as session:
        await session.execute(
            insert(Item),
            [
                {
                    "id": iid,
                    "title": iid,
                    "description": "",
                    "category": "electronics/headphones",
                    "brand": "Acme",
                    "price": 10.0 + idx,
                }
                for idx, iid in enumerate(extra_ids)
            ],
        )
        await session.commit()

    user_id = "u_window"
    bootstrap = kafka_container.get_bootstrap_server()
    # Send 5 prior clicks to fill the window, THEN a 6th click on a new item.
    sequence = extra_ids + ["i_int_2"]
    for iid in sequence:
        await _publish(bootstrap, USER_CLICKS.name, user_id, _click_payload(user_id, iid))
        await asyncio.sleep(0.1)

    async def pair_exists() -> bool:
        async with sm() as s:
            # Last extra item paired with the new click — must exist.
            a, b = sorted(["i_extra_4", "i_int_2"])
            row = (
                await s.execute(
                    select(CoClick).where(CoClick.item_a == a, CoClick.item_b == b)
                )
            ).first()
            return row is not None

    assert await _wait_for(pair_exists, timeout_s=15.0)
