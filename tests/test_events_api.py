"""Unit tests for `/events/*` — producer + redis are patched.

`httpx.AsyncClient(transport=ASGITransport(app))` bypasses the lifespan, so
these tests do not require Kafka / Redis to be running.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from aiokafka.errors import KafkaTimeoutError
from httpx import ASGITransport, AsyncClient
from uuid6 import uuid7

from click_rec.api.app import app
from click_rec.api.routers import events as events_module
from click_rec.kafka.topics import USER_CLICKS, USER_IMPRESSIONS, USER_SEARCHES


class _FakeProducer:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    async def publish(self, topic: str, key: str, event: dict[str, Any]) -> None:
        self.calls.append((topic, key, event))


class _FakeDedupe:
    def __init__(self) -> None:
        self.seen: set[str] = set()

    async def claim(self, event_id: uuid.UUID) -> bool:
        key = str(event_id)
        if key in self.seen:
            return False
        self.seen.add(key)
        return True


@pytest.fixture
def fake_producer(monkeypatch: pytest.MonkeyPatch) -> _FakeProducer:
    fake = _FakeProducer()
    monkeypatch.setattr(events_module, "publish_event", fake.publish)
    return fake


@pytest.fixture
def fake_dedupe(monkeypatch: pytest.MonkeyPatch) -> _FakeDedupe:
    fake = _FakeDedupe()
    monkeypatch.setattr(events_module, "dedupe_event_id", fake.claim)
    return fake


@pytest.fixture
async def client() -> AsyncClient:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def _click_body(**overrides: Any) -> dict[str, Any]:
    body = {
        "timestamp": "2026-04-19T10:00:00Z",
        "user_id": "u1",
        "session_id": "s1",
        "query": "wireless headphones",
        "item_id": "i42",
        "rank_position": 3,
        "dwell_ms": 5000,
    }
    body.update(overrides)
    return body


def _impression_body(**overrides: Any) -> dict[str, Any]:
    body = {
        "event_type": "impression",
        "timestamp": "2026-04-19T10:00:00Z",
        "user_id": "u1",
        "session_id": "s1",
        "query": "wireless headphones",
        "result_ids": ["i1", "i2", "i3"],
        "page": 1,
    }
    body.update(overrides)
    return body


def _search_body(**overrides: Any) -> dict[str, Any]:
    body = {
        "event_type": "search",
        "timestamp": "2026-04-19T10:00:00Z",
        "user_id": "u2",
        "session_id": "s2",
        "query": "running shoes",
        "filters": {},
    }
    body.update(overrides)
    return body


async def test_click_happy_path(
    client: AsyncClient, fake_producer: _FakeProducer, fake_dedupe: _FakeDedupe
) -> None:
    eid = str(uuid7())
    resp = await client.post("/events/click", json=_click_body(event_id=eid))
    assert resp.status_code == 202
    body = resp.json()
    assert body == {"event_id": eid, "status": "accepted"}
    assert len(fake_producer.calls) == 1
    topic, key, payload = fake_producer.calls[0]
    assert topic == USER_CLICKS.name
    assert key == "u1"
    assert payload["event_id"] == eid
    assert "server_ts" in payload
    assert payload["event_type"] == "click"


async def test_click_generates_uuid7_when_missing(
    client: AsyncClient, fake_producer: _FakeProducer, fake_dedupe: _FakeDedupe
) -> None:
    resp = await client.post("/events/click", json=_click_body())
    assert resp.status_code == 202
    returned = uuid.UUID(resp.json()["event_id"])
    assert returned.version == 7
    assert fake_producer.calls[0][2]["event_id"] == str(returned)


async def test_duplicate_event_id_skips_publish(
    client: AsyncClient, fake_producer: _FakeProducer, fake_dedupe: _FakeDedupe
) -> None:
    eid = str(uuid7())
    body = _click_body(event_id=eid)
    first = await client.post("/events/click", json=body)
    second = await client.post("/events/click", json=body)
    assert first.status_code == 202
    assert first.json()["status"] == "accepted"
    assert second.status_code == 202
    assert second.json()["status"] == "duplicate"
    assert len(fake_producer.calls) == 1


async def test_extra_fields_rejected(
    client: AsyncClient, fake_producer: _FakeProducer, fake_dedupe: _FakeDedupe
) -> None:
    body = _click_body(mystery_field="nope")
    resp = await client.post("/events/click", json=body)
    assert resp.status_code == 422
    assert fake_producer.calls == []


async def test_impression_routes_to_impressions_topic(
    client: AsyncClient, fake_producer: _FakeProducer, fake_dedupe: _FakeDedupe
) -> None:
    resp = await client.post("/events/impression", json=_impression_body())
    assert resp.status_code == 202
    assert fake_producer.calls[0][0] == USER_IMPRESSIONS.name


async def test_search_routes_to_searches_topic(
    client: AsyncClient, fake_producer: _FakeProducer, fake_dedupe: _FakeDedupe
) -> None:
    resp = await client.post("/events/search", json=_search_body())
    assert resp.status_code == 202
    assert fake_producer.calls[0][0] == USER_SEARCHES.name


async def test_batch_fans_out_by_event_type(
    client: AsyncClient, fake_producer: _FakeProducer, fake_dedupe: _FakeDedupe
) -> None:
    batch = {
        "events": [
            _click_body(event_type="click"),
            _impression_body(),
            _search_body(),
        ]
    }
    resp = await client.post("/events/batch", json=batch)
    assert resp.status_code == 202
    results = resp.json()["results"]
    assert len(results) == 3
    topics = {call[0] for call in fake_producer.calls}
    assert topics == {USER_CLICKS.name, USER_IMPRESSIONS.name, USER_SEARCHES.name}


async def test_batch_preserves_per_user_ordering(
    client: AsyncClient, fake_producer: _FakeProducer, fake_dedupe: _FakeDedupe
) -> None:
    """Events with the same user_id must publish in input order.

    Per-user ordering is the partitioning contract (plan §6 / §9). The
    batch handler groups by `user_id` and processes each group
    sequentially so two `send_and_wait` calls on the same key cannot
    race at the broker.
    """
    events = [
        _click_body(event_type="click", user_id="u_order", item_id=f"i{i}", rank_position=i)
        for i in range(5)
    ]
    resp = await client.post("/events/batch", json={"events": events})
    assert resp.status_code == 202
    user_u_calls = [c for c in fake_producer.calls if c[1] == "u_order"]
    assert [c[2]["item_id"] for c in user_u_calls] == ["i0", "i1", "i2", "i3", "i4"]


async def test_batch_isolates_per_event_errors(
    client: AsyncClient,
    fake_dedupe: _FakeDedupe,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single failing event must not tank the whole batch."""
    calls: list[tuple[str, str, dict[str, Any]]] = []

    async def flaky_publish(topic: str, key: str, event: dict[str, Any]) -> None:
        calls.append((topic, key, event))
        if event["item_id"] == "boom":
            raise RuntimeError("broker hiccup")

    monkeypatch.setattr(events_module, "publish_event", flaky_publish)

    batch = {
        "events": [
            _click_body(event_type="click", user_id="u_a", item_id="ok1"),
            _click_body(event_type="click", user_id="u_b", item_id="boom"),
            _click_body(event_type="click", user_id="u_c", item_id="ok2"),
        ]
    }
    resp = await client.post("/events/batch", json=batch)
    assert resp.status_code == 202
    results = resp.json()["results"]
    statuses = [r["status"] for r in results]
    assert statuses.count("accepted") == 2
    assert statuses.count("error") == 1
    assert "RuntimeError" in results[1]["detail"]


async def test_batch_over_limit_is_rejected(
    client: AsyncClient, fake_producer: _FakeProducer, fake_dedupe: _FakeDedupe
) -> None:
    batch = {"events": [_click_body(event_type="click") for _ in range(501)]}
    resp = await client.post("/events/batch", json=batch)
    assert resp.status_code == 422


async def test_kafka_unavailable_returns_503_and_releases_claim(
    client: AsyncClient,
    fake_dedupe: _FakeDedupe,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When Kafka is down the dedupe claim must be rolled back.

    Without the rollback, a retried `event_id` would come back as
    `duplicate` while nothing ever reached the broker — exactly the
    failure the dedupe is supposed to prevent.
    """
    released: list[uuid.UUID] = []

    async def boom(topic: str, key: str, event: dict[str, Any]) -> None:
        raise KafkaTimeoutError("broker down")

    async def record_release(event_id: uuid.UUID) -> None:
        released.append(event_id)

    monkeypatch.setattr(events_module, "publish_event", boom)
    monkeypatch.setattr(events_module, "release_event_id", record_release)

    eid = str(uuid7())
    resp = await client.post("/events/click", json=_click_body(event_id=eid))
    assert resp.status_code == 503
    assert resp.headers["Retry-After"] == "1"
    assert [str(e) for e in released] == [eid]


async def test_oversized_single_body_returns_413(
    client: AsyncClient, fake_producer: _FakeProducer, fake_dedupe: _FakeDedupe
) -> None:
    padded = _click_body(query="x" * 2000)
    resp = await client.post("/events/click", json=padded)
    assert resp.status_code == 413
    assert fake_producer.calls == []
