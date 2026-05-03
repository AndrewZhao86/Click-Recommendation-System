"""Phase 8 — query embedding cache-aside in `ranker.embedder`.

The cache is what turns the cold-encode tail (~2 s under load) into the
warm-path latency that fits the 150 ms p95 budget. These tests use a
fake Redis client so the cache logic is exercised without needing a
running container; the SentenceTransformer load is also stubbed so the
suite stays under a second.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from click_rec.ranker import embedder
from click_rec.telemetry import metrics


class _FakeRedis:
    """Minimal async stand-in for `redis.asyncio.Redis`.

    Only `get` and `set` are exercised by `encode_query`. `set` ignores
    `ex=` because TTL behaviour is verified by the integration test, not
    here.
    """

    def __init__(self) -> None:
        self.store: dict[str, bytes] = {}
        self.get_calls = 0
        self.set_calls = 0

    async def get(self, key: str) -> bytes | None:
        self.get_calls += 1
        return self.store.get(key)

    async def set(self, key: str, value: bytes, ex: int | None = None) -> bool:
        self.set_calls += 1
        self.store[key] = value
        return True


@pytest.fixture
def fake_redis(monkeypatch: pytest.MonkeyPatch) -> _FakeRedis:
    fake = _FakeRedis()
    monkeypatch.setattr(embedder, "get_redis", lambda: fake)
    return fake


@pytest.fixture
def stub_model(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Replace the SentenceTransformer load + encode with a counter."""
    counters = {"encode": 0}

    expected_vec = [0.1, 0.2, 0.3, 0.4]

    async def fake_load() -> Any:
        return object()

    async def fake_encode(_model: Any, query: str) -> list[float]:
        counters["encode"] += 1
        return expected_vec

    monkeypatch.setattr(embedder, "_load_model", fake_load)
    monkeypatch.setattr(embedder, "_encode_blocking_offloaded", fake_encode)
    return counters


def _hit_value(key_type: str) -> float:
    return metrics.cache_hit_total.labels(
        key_type=key_type, status="value"
    )._value.get()  # type: ignore[attr-defined]


def _miss_value(key_type: str) -> float:
    return metrics.cache_miss_total.labels(key_type=key_type)._value.get()  # type: ignore[attr-defined]


async def test_first_call_misses_then_caches(
    fake_redis: _FakeRedis, stub_model: dict[str, int]
) -> None:
    miss_before = _miss_value("query_embedding")
    vec = await embedder.encode_query("wireless headphones")
    assert stub_model["encode"] == 1
    assert fake_redis.set_calls == 1
    # Cache value is float32 bytes; round-trip should match within fp32
    # precision.
    assert len(fake_redis.store) == 1
    cached_bytes = next(iter(fake_redis.store.values()))
    np.testing.assert_allclose(
        np.frombuffer(cached_bytes, dtype=np.float32),
        np.asarray(vec, dtype=np.float32),
    )
    assert _miss_value("query_embedding") == miss_before + 1


async def test_repeat_call_serves_from_cache(
    fake_redis: _FakeRedis, stub_model: dict[str, int]
) -> None:
    hit_before = _hit_value("query_embedding")

    first = await embedder.encode_query("noise cancelling earbuds")
    assert stub_model["encode"] == 1

    second = await embedder.encode_query("noise cancelling earbuds")
    # No second encode — second call served from cache.
    assert stub_model["encode"] == 1
    # Hit counter advanced.
    assert _hit_value("query_embedding") == hit_before + 1
    # Decoded list equals the original up to fp32 round-trip.
    np.testing.assert_allclose(second, first, rtol=1e-6, atol=1e-6)


async def test_redis_unavailable_falls_back_to_direct_encode(
    monkeypatch: pytest.MonkeyPatch, stub_model: dict[str, int]
) -> None:
    """No Redis singleton means the encoder must still produce a vector."""

    def _no_redis() -> Any:
        raise RuntimeError("redis not started")

    monkeypatch.setattr(embedder, "get_redis", _no_redis)

    vec = await embedder.encode_query("anything")
    assert vec == [0.1, 0.2, 0.3, 0.4]
    assert stub_model["encode"] == 1


async def test_redis_get_error_falls_back_without_writing(
    monkeypatch: pytest.MonkeyPatch, stub_model: dict[str, int]
) -> None:
    """A Redis GET that raises must not block the request and must skip the SET."""
    from redis.exceptions import ConnectionError as RedisConnectionError

    class BrokenRedis:
        def __init__(self) -> None:
            self.set_calls = 0

        async def get(self, key: str) -> bytes | None:
            raise RedisConnectionError("boom")

        async def set(self, *_: Any, **__: Any) -> bool:
            self.set_calls += 1
            return True

    broken = BrokenRedis()
    monkeypatch.setattr(embedder, "get_redis", lambda: broken)

    vec = await embedder.encode_query("query")
    assert vec == [0.1, 0.2, 0.3, 0.4]
    assert stub_model["encode"] == 1
    # We deliberately skip SET when GET failed — re-issuing on a broken
    # connection just amplifies the outage.
    assert broken.set_calls == 0


async def test_distinct_queries_get_distinct_keys(
    fake_redis: _FakeRedis, stub_model: dict[str, int]
) -> None:
    await embedder.encode_query("query one")
    await embedder.encode_query("query two")
    assert stub_model["encode"] == 2
    assert len(fake_redis.store) == 2
