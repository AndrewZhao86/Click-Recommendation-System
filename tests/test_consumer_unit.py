"""Unit tests for the Phase 4a/4b enrichment consumer.

DB session, Redis client, and Kafka producer are all replaced with fakes
so these tests run in <1s without Docker. The integration test suite in
`tests/test_consumer_integration.py` covers the real Kafka / Redis /
Postgres path via `testcontainers`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import pytest
from uuid6 import uuid7

from click_rec.kafka import consumer as consumer_module
from click_rec.kafka import enrichment as enrichment_module
from click_rec.kafka.consumer import ProcessOutcome, _process_one, _send_to_dlq
from click_rec.kafka.enrichment import (
    apply_enrichment,
    resolve_item_meta,
    upsert_co_clicks,
)
from click_rec.models.schemas import ClickEventDTO


# Phase 4 review M1: the process-local item cache leaks across tests
# without explicit clearing. Pytest doesn't tear down module-level state.
# Same applies to the negative cache.
@pytest.fixture(autouse=True)
def _clear_item_cache() -> AsyncIterator[None]:
    enrichment_module._item_cache.clear()
    enrichment_module._item_negative_cache.clear()
    yield
    enrichment_module._item_cache.clear()
    enrichment_module._item_negative_cache.clear()


# ------------------------------------------------------------------ fakes


class _FakeRedisPipeline:
    def __init__(self, parent: _FakeRedis) -> None:
        self._parent = parent
        self._ops: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def zadd(self, key: str, mapping: dict[str, float]) -> _FakeRedisPipeline:
        self._ops.append(("zadd", (key, mapping), {}))
        return self

    def zremrangebyrank(self, key: str, start: int, stop: int) -> _FakeRedisPipeline:
        self._ops.append(("zremrangebyrank", (key, start, stop), {}))
        return self

    def expire(self, key: str, seconds: int) -> _FakeRedisPipeline:
        self._ops.append(("expire", (key, seconds), {}))
        return self

    def incr(self, key: str) -> _FakeRedisPipeline:
        self._ops.append(("incr", (key,), {}))
        return self

    def zincrby(self, key: str, amount: float, member: str) -> _FakeRedisPipeline:
        self._ops.append(("zincrby", (key, amount, member), {}))
        return self

    async def execute(self) -> list[Any]:
        results: list[Any] = []
        for op, args, _ in self._ops:
            if op == "zadd":
                key, mapping = args
                z = self._parent.sorted_sets.setdefault(key, {})
                for member, score in mapping.items():
                    z[member] = score
                results.append(len(mapping))
            elif op == "zremrangebyrank":
                key, start, stop = args
                z = self._parent.sorted_sets.get(key, {})
                items = sorted(z.items(), key=lambda kv: kv[1])  # ascending
                # Translate negative stop like real redis: -1 means last.
                length = len(items)
                stop_idx = length + stop if stop < 0 else stop
                to_remove = items[start : stop_idx + 1]
                for member, _ in to_remove:
                    z.pop(member, None)
                results.append(len(to_remove))
            elif op == "expire":
                results.append(1)
            elif op == "incr":
                key = args[0]
                self._parent.counters[key] = self._parent.counters.get(key, 0) + 1
                results.append(self._parent.counters[key])
            elif op == "zincrby":
                key, amount, member = args
                z = self._parent.sorted_sets.setdefault(key, {})
                z[member] = z.get(member, 0.0) + amount
                results.append(z[member])
        self._ops.clear()
        return results


class _FakeRedis:
    def __init__(self) -> None:
        self.sorted_sets: dict[str, dict[str, float]] = {}
        self.counters: dict[str, int] = {}
        self.kv: dict[str, bytes] = {}

    def pipeline(self, transaction: bool = True) -> _FakeRedisPipeline:
        return _FakeRedisPipeline(self)

    async def zrevrange(self, key: str, start: int, stop: int) -> list[bytes]:
        z = self.sorted_sets.get(key, {})
        ordered = sorted(z.items(), key=lambda kv: -kv[1])
        sliced = ordered[start : stop + 1] if stop >= 0 else ordered[start:]
        return [m.encode() for m, _ in sliced]

    async def exists(self, key: str) -> int:
        return 1 if key in self.kv else 0

    async def set(self, key: str, value: bytes, ex: int | None = None,
                  nx: bool = False) -> bool | None:
        if nx and key in self.kv:
            return None
        self.kv[key] = value
        return True

    async def delete(self, key: str) -> int:
        return 1 if self.kv.pop(key, None) is not None else 0


class _FakeProducer:
    def __init__(self, fail: bool = False) -> None:
        self.calls: list[tuple[str, Any, str | None]] = []
        self.fail = fail

    async def send_and_wait(self, topic: str, value: Any, key: str | None = None) -> None:
        if self.fail:
            from aiokafka.errors import KafkaError

            raise KafkaError("fake broker outage")
        self.calls.append((topic, value, key))


class _FakeSession:
    """Minimal async session: records executed statements, no real DB."""

    def __init__(self, item_rows: dict[str, tuple[str, float]] | None = None) -> None:
        self.item_rows = item_rows or {}
        self.executed: list[Any] = []
        self.committed = 0

    async def execute(self, stmt: Any) -> Any:
        self.executed.append(stmt)
        # Detect the resolve_item_meta SELECT: returns (category, price).
        compiled = str(stmt)
        if "FROM item" in compiled and "WHERE" in compiled:
            class _Result:
                def __init__(self, rows: dict[str, tuple[str, float]]) -> None:
                    self._rows = rows

                def first(self_inner) -> Any:
                    # The test patches resolve_item_meta directly when it
                    # needs deterministic data; this fallback returns
                    # None so we fail loudly if a test forgets.
                    return None
            return _Result(self.item_rows)
        return None

    async def commit(self) -> None:
        self.committed += 1

    async def __aenter__(self) -> _FakeSession:
        return self

    async def __aexit__(self, *_: Any) -> None:
        return None


class _FakeRecord:
    def __init__(
        self,
        value: bytes,
        offset: int = 0,
        partition: int = 0,
        topic: str = "user.clicks",
        key: bytes | None = b"u1",
    ) -> None:
        self.value = value
        self.offset = offset
        self.partition = partition
        self.topic = topic
        self.key = key


# ------------------------------------------------------------------ helpers


def _click_event(
    event_id: UUID | None = None,
    user_id: str = "u1",
    item_id: str = "i1",
) -> ClickEventDTO:
    return ClickEventDTO(
        event_id=event_id or uuid7(),
        timestamp=datetime(2026, 4, 22, 10, 0, 0, tzinfo=UTC),
        user_id=user_id,
        session_id="s1",
        query="wireless headphones",
        item_id=item_id,
        rank_position=2,
        dwell_ms=4500,
    )


def _click_payload(**overrides: Any) -> bytes:
    import orjson

    body: dict[str, Any] = {
        "event_id": str(uuid7()),
        "event_type": "click",
        "event_version": 1,
        "timestamp": "2026-04-22T10:00:00Z",
        "user_id": "u1",
        "session_id": "s1",
        "query": "wireless headphones",
        "item_id": "i1",
        "rank_position": 2,
        "dwell_ms": 4500,
    }
    body.update(overrides)
    return orjson.dumps(body)


# ============================================================ enrichment


async def test_resolve_item_meta_caches_after_first_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Second call for the same item must NOT re-execute the SELECT."""
    calls: list[str] = []

    class _Row:
        def __init__(self, category: str, price: float) -> None:
            self.category = category
            self.price = price

    class _RecordingSession:
        async def execute(self_inner, stmt: Any) -> Any:
            calls.append(str(stmt))

            class _Result:
                def first(_self) -> Any:
                    return _Row("electronics/headphones", 49.99)

            return _Result()

    sess = _RecordingSession()
    cat, price = await resolve_item_meta(sess, "i_cache_test")  # type: ignore[arg-type]
    assert (cat, price) == ("electronics/headphones", 49.99)
    cat2, price2 = await resolve_item_meta(sess, "i_cache_test")  # type: ignore[arg-type]
    assert (cat2, price2) == ("electronics/headphones", 49.99)
    assert len(calls) == 1


async def test_resolve_item_meta_caches_negative_lookups() -> None:
    """A second lookup for an unknown item must NOT re-query the DB."""
    calls: list[str] = []

    class _RecordingSession:
        async def execute(self_inner, stmt: Any) -> Any:
            calls.append(str(stmt))

            class _Result:
                def first(_self) -> Any:
                    return None  # Item doesn't exist.

            return _Result()

    sess = _RecordingSession()
    assert await resolve_item_meta(sess, "i_missing") is None  # type: ignore[arg-type]
    assert await resolve_item_meta(sess, "i_missing") is None  # type: ignore[arg-type]
    assert len(calls) == 1


async def test_upsert_co_clicks_dedupes_input_and_skips_self_pairs() -> None:
    sess = _FakeSession()
    await upsert_co_clicks(sess, ["i1", "i1", "i2"])  # type: ignore[arg-type]
    # Should produce a single bulk INSERT for one pair (i1, i2), not three.
    assert len(sess.executed) == 1


async def test_upsert_co_clicks_noop_under_two_items() -> None:
    sess = _FakeSession()
    await upsert_co_clicks(sess, ["i1"])  # type: ignore[arg-type]
    assert sess.executed == []


async def test_upsert_co_clicks_on_conflict_is_an_accumulator() -> None:
    """The ON CONFLICT clause is `count = count + 1` by design.

    Replays re-increment by 1. The consumer dedupe marker in
    `_process_one` catches the happy-path replay (same message
    redelivered before ack); crash-mid-pipeline replays leak a bounded
    amount of drift, which Phase 5 decay absorbs. If someone ever
    switches this to `GREATEST(count, 1)` or a no-op on conflict they
    must also update the enrichment module docstring — hence this
    contract test.
    """
    from sqlalchemy.dialects import postgresql

    sess = _FakeSession()
    await upsert_co_clicks(sess, ["i1", "i2"])  # type: ignore[arg-type]
    assert len(sess.executed) == 1
    rendered = str(
        sess.executed[0].compile(
            dialect=postgresql.dialect(),
            compile_kwargs={"literal_binds": False},
        )
    )
    assert "ON CONFLICT" in rendered.upper()
    # The SET clause must reference the existing column plus the excluded
    # value — anything else would change the replay semantics.
    assert "co_click.count +" in rendered


async def test_apply_enrichment_includes_current_click_in_session_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Phase 4 review C1: with N prior items already present, the current
    click must NOT get sliced out of the co-click pairs.
    """
    redis_fake = _FakeRedis()
    # Prime 5 prior clicks (= session_window).
    for i, ts in enumerate([1, 2, 3, 4, 5]):
        redis_fake.sorted_sets.setdefault("user:u1:recent_clicks", {})[f"prior{i}"] = ts

    captured_pairs: list[list[str]] = []

    async def fake_upsert(session: Any, item_ids: list[str]) -> None:
        captured_pairs.append(item_ids)

    monkeypatch.setattr(enrichment_module, "upsert_co_clicks", fake_upsert)

    async def fake_resolve(session: Any, item_id: str) -> tuple[str, float]:
        return ("electronics/headphones", 99.0)

    monkeypatch.setattr(enrichment_module, "resolve_item_meta", fake_resolve)

    producer = _FakeProducer()
    sess = _FakeSession()
    event = _click_event(item_id="i_current")
    await apply_enrichment(
        event=event,
        session=sess,  # type: ignore[arg-type]
        redis_client=redis_fake,  # type: ignore[arg-type]
        producer=producer,  # type: ignore[arg-type]
    )

    assert len(captured_pairs) == 1
    items = captured_pairs[0]
    # Current click MUST be in the window.
    assert "i_current" in items
    # Window honours the cap.
    from click_rec.config import get_settings

    assert len(items) == get_settings().session_window
    # Current click is first (newest).
    assert items[0] == "i_current"


async def test_apply_enrichment_publishes_profile_update(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_resolve(session: Any, item_id: str) -> tuple[str, float]:
        return ("books/scifi", 12.50)

    monkeypatch.setattr(enrichment_module, "resolve_item_meta", fake_resolve)
    monkeypatch.setattr(enrichment_module, "upsert_co_clicks",
                        lambda session, items: _noop_async())

    producer = _FakeProducer()
    redis_fake = _FakeRedis()
    sess = _FakeSession()
    event = _click_event()
    await apply_enrichment(
        event=event,
        session=sess,  # type: ignore[arg-type]
        redis_client=redis_fake,  # type: ignore[arg-type]
        producer=producer,  # type: ignore[arg-type]
    )

    from click_rec.kafka.topics import USER_PROFILE_UPDATES

    assert len(producer.calls) == 1
    topic, payload, key = producer.calls[0]
    assert topic == USER_PROFILE_UPDATES.name
    assert key == event.user_id
    assert payload["category"] == "books/scifi"
    assert payload["item_id"] == event.item_id


async def _noop_async() -> None:
    return None


async def test_apply_enrichment_skips_unknown_item(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_resolve(session: Any, item_id: str) -> tuple[str, float] | None:
        return None

    monkeypatch.setattr(enrichment_module, "resolve_item_meta", fake_resolve)
    producer = _FakeProducer()
    sess = _FakeSession()
    redis_fake = _FakeRedis()
    await apply_enrichment(
        event=_click_event(),
        session=sess,  # type: ignore[arg-type]
        redis_client=redis_fake,  # type: ignore[arg-type]
        producer=producer,  # type: ignore[arg-type]
    )
    assert producer.calls == []


# ============================================================ _process_one


async def test_process_one_marks_processed_after_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Phase 4 review C4(a): the consumer marker is set AFTER side-effects."""
    marked: list[UUID] = []
    checked: list[UUID] = []

    async def fake_check(event_id: UUID) -> bool:
        checked.append(event_id)
        return False

    async def fake_mark(event_id: UUID) -> None:
        marked.append(event_id)

    enrichment_called: list[ClickEventDTO] = []

    async def fake_enrichment(*, event: ClickEventDTO, session: Any,
                              redis_client: Any, producer: Any) -> None:
        # Assert the marker has NOT been set yet at this point.
        assert event.event_id not in marked
        enrichment_called.append(event)

    monkeypatch.setattr(consumer_module, "is_consumer_event_processed", fake_check)
    monkeypatch.setattr(consumer_module, "mark_consumer_event_processed", fake_mark)
    monkeypatch.setattr(consumer_module, "apply_enrichment", fake_enrichment)
    monkeypatch.setattr(consumer_module, "_get_redis_for_consumer", lambda: _FakeRedis())
    monkeypatch.setattr(consumer_module, "get_sessionmaker", lambda: _FakeSession)

    producer = _FakeProducer()
    record = _FakeRecord(_click_payload())
    outcome = await _process_one(record, producer, worker_id=0)  # type: ignore[arg-type]

    assert outcome == ProcessOutcome.OK
    assert len(enrichment_called) == 1
    assert marked == [enrichment_called[0].event_id]


async def test_process_one_skips_when_already_processed(monkeypatch: pytest.MonkeyPatch) -> None:
    async def already_processed(event_id: UUID) -> bool:
        return True

    enrichment_calls: list[Any] = []

    async def fake_enrichment(**_: Any) -> None:
        enrichment_calls.append(1)

    monkeypatch.setattr(consumer_module, "is_consumer_event_processed", already_processed)
    monkeypatch.setattr(consumer_module, "apply_enrichment", fake_enrichment)
    monkeypatch.setattr(consumer_module, "_get_redis_for_consumer", lambda: _FakeRedis())

    producer = _FakeProducer()
    record = _FakeRecord(_click_payload())
    outcome = await _process_one(record, producer, worker_id=0)  # type: ignore[arg-type]
    assert outcome == ProcessOutcome.OK
    assert enrichment_calls == []


async def test_process_one_poison_pill_to_dlq(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(consumer_module, "is_consumer_event_processed",
                        _async_returning(False))
    producer = _FakeProducer()
    record = _FakeRecord(b"this is not json")
    outcome = await _process_one(record, producer, worker_id=0)  # type: ignore[arg-type]
    assert outcome == ProcessOutcome.DLQ_OK
    assert len(producer.calls) == 1
    topic, payload, _ = producer.calls[0]
    from click_rec.kafka.topics import USER_CLICKS_DLQ

    assert topic == USER_CLICKS_DLQ.name
    assert payload["reason"] == "parse_or_validation_error"


async def test_process_one_retries_then_dlq(monkeypatch: pytest.MonkeyPatch) -> None:
    """After `consumer_max_retries` failed attempts the message goes to DLQ.

    Phase 4 review C4(a): the success-path marker MUST NOT be set when
    we end up in the DLQ branch.
    """
    monkeypatch.setattr(consumer_module, "is_consumer_event_processed",
                        _async_returning(False))
    marker_calls: list[UUID] = []

    async def fake_mark(event_id: UUID) -> None:
        marker_calls.append(event_id)

    async def always_fails(**_: Any) -> None:
        raise RuntimeError("simulated db error")

    monkeypatch.setattr(consumer_module, "mark_consumer_event_processed", fake_mark)
    monkeypatch.setattr(consumer_module, "apply_enrichment", always_fails)
    monkeypatch.setattr(consumer_module, "_get_redis_for_consumer", lambda: _FakeRedis())
    monkeypatch.setattr(consumer_module, "get_sessionmaker", lambda: _FakeSession)
    # No-op the back-off so the test stays under a second.
    import asyncio as _asyncio

    monkeypatch.setattr(_asyncio, "sleep", _async_returning(None))

    producer = _FakeProducer()
    record = _FakeRecord(_click_payload())
    outcome = await _process_one(record, producer, worker_id=0)  # type: ignore[arg-type]

    assert outcome == ProcessOutcome.DLQ_OK
    assert marker_calls == []  # Must NOT mark on failure path.
    # DLQ message present and contains a real stack trace (review C3).
    assert len(producer.calls) == 1
    dlq_payload = producer.calls[0][1]
    assert dlq_payload["error_class"] == "RuntimeError"
    assert "simulated db error" in dlq_payload["stack_trace"]
    assert "Traceback" in dlq_payload["stack_trace"]


async def test_process_one_dlq_failure_returns_dlq_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Phase 4 review C2: when DLQ publish itself fails, return DLQ_FAILED so
    the worker loop holds the offset back instead of silently advancing.
    """
    monkeypatch.setattr(consumer_module, "is_consumer_event_processed",
                        _async_returning(False))

    producer = _FakeProducer(fail=True)
    record = _FakeRecord(b"this is not json")
    outcome = await _process_one(record, producer, worker_id=0)  # type: ignore[arg-type]
    assert outcome == ProcessOutcome.DLQ_FAILED


# ============================================================ _send_to_dlq


async def test_send_to_dlq_formats_traceback_from_passed_exception() -> None:
    """Phase 4 review C3: format from the exception object, not the active context."""
    producer = _FakeProducer()

    def _raises() -> None:
        raise ValueError("fixture error")

    captured: BaseException | None = None
    try:
        _raises()
    except ValueError as exc:
        captured = exc

    record = _FakeRecord(b'{"a": 1}')
    sent = await _send_to_dlq(producer, record, captured, "test_reason")  # type: ignore[arg-type]
    assert sent is True
    payload = producer.calls[0][1]
    assert payload["error_class"] == "ValueError"
    assert payload["error_message"] == "fixture error"
    assert "Traceback" in payload["stack_trace"]
    assert "fixture error" in payload["stack_trace"]


async def test_send_to_dlq_handles_malformed_payload_bytes() -> None:
    """Raw bytes that aren't JSON go through as a `replace`-decoded string."""
    producer = _FakeProducer()
    record = _FakeRecord(b"\xff\xfe not json")
    sent = await _send_to_dlq(producer, record, RuntimeError("x"), "test")  # type: ignore[arg-type]
    assert sent is True
    assert isinstance(producer.calls[0][1]["original_payload"], str)


# ============================================================ helpers


def _async_returning(value: Any) -> Any:
    async def _f(*_a: Any, **_k: Any) -> Any:
        return value
    return _f
