"""Integration tests for Phase 3 — Kafka + Redis via testcontainers.

Guarded by `@pytest.mark.integration` and `pytest.importorskip("testcontainers")`
so developers without Docker can skip this file. Matches the Phase 1 principle:
"integration tests against real Kafka/Redis/Postgres via testcontainers, not
mocks."
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import pytest

pytestmark = pytest.mark.integration

testcontainers = pytest.importorskip("testcontainers")
kafka_mod = pytest.importorskip("testcontainers.kafka")
redis_mod = pytest.importorskip("testcontainers.redis")
asgi_lifespan = pytest.importorskip("asgi_lifespan")
aiokafka = pytest.importorskip("aiokafka")

from httpx import ASGITransport, AsyncClient  # noqa: E402

from click_rec.kafka.topics import USER_CLICKS  # noqa: E402


@pytest.fixture(scope="module")
def kafka_container() -> Any:
    with kafka_mod.KafkaContainer() as kc:
        yield kc


@pytest.fixture(scope="module")
def redis_container() -> Any:
    with redis_mod.RedisContainer() as rc:
        yield rc


@pytest.fixture(scope="module")
def monkeypatch_module() -> pytest.MonkeyPatch:
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


@pytest.fixture(scope="module")
def configured_settings(
    kafka_container: Any,
    redis_container: Any,
    monkeypatch_module: pytest.MonkeyPatch,
) -> None:
    """Point settings at the live containers.

    `get_settings` is `@lru_cache`d, and several modules (`producer`,
    `redis_client`, `app`) already imported the function at module load —
    so a `reload(config)` would not help those holders. Instead we set
    the env vars, then clear the cache so the next `get_settings()` call
    (which happens inside `start_producer` / `start_redis` at lifespan
    time) reads a fresh `Settings` pointed at the containers.
    """
    bootstrap = kafka_container.get_bootstrap_server()
    host = redis_container.get_container_host_ip()
    port = redis_container.get_exposed_port(6379)
    monkeypatch_module.setenv("KAFKA_BOOTSTRAP", bootstrap)
    monkeypatch_module.setenv("REDIS_URL", f"redis://{host}:{port}/0")

    from click_rec.config import get_settings

    get_settings.cache_clear()


@pytest.fixture
async def live_client(configured_settings: None) -> AsyncIterator[AsyncClient]:
    from click_rec.api.app import app

    async with asgi_lifespan.LifespanManager(app, startup_timeout=60):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            yield c


def _click_body(event_id: str | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {
        "timestamp": "2026-04-19T10:00:00Z",
        "user_id": "u_int",
        "session_id": "s_int",
        "query": "integration test query",
        "item_id": "i_int",
        "rank_position": 1,
    }
    if event_id is not None:
        body["event_id"] = event_id
    return body


async def _drain_clicks_topic(bootstrap: str, timeout_s: float = 10.0) -> list[dict[str, Any]]:
    consumer = aiokafka.AIOKafkaConsumer(
        USER_CLICKS.name,
        bootstrap_servers=bootstrap,
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        group_id=None,
    )
    await consumer.start()
    messages: list[dict[str, Any]] = []
    try:
        deadline = asyncio.get_event_loop().time() + timeout_s
        while asyncio.get_event_loop().time() < deadline:
            batch = await consumer.getmany(timeout_ms=500)
            for _tp, records in batch.items():
                for r in records:
                    messages.append(json.loads(r.value))
            if messages and not batch:
                break
    finally:
        await consumer.stop()
    return messages


async def test_click_publishes_to_kafka(live_client: AsyncClient, kafka_container: Any) -> None:
    resp = await live_client.post("/events/click", json=_click_body())
    assert resp.status_code == 202
    eid = resp.json()["event_id"]

    messages = await _drain_clicks_topic(kafka_container.get_bootstrap_server())
    assert any(m["event_id"] == eid for m in messages)


async def test_duplicate_event_id_is_deduped(
    live_client: AsyncClient, kafka_container: Any
) -> None:
    from uuid6 import uuid7

    eid = str(uuid7())
    body = _click_body(event_id=eid)
    for _ in range(5):
        resp = await live_client.post("/events/click", json=body)
        assert resp.status_code == 202

    messages = await _drain_clicks_topic(kafka_container.get_bootstrap_server())
    matches = [m for m in messages if m["event_id"] == eid]
    assert len(matches) == 1
