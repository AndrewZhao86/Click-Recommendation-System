"""Unit tests for `understand_query` cache + fallback semantics."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from click_rec.llm import client as client_mod
from click_rec.llm import schemas
from click_rec.llm.config import LLMConfig
from click_rec.llm.query_understanding import understand_query

# ---------------------------------------------------------------- fakes


class _FakeRedis:
    def __init__(self, *, get_returns: bytes | None = None) -> None:
        self.store: dict[str, bytes] = {}
        self.get_returns = get_returns
        self.set_calls: list[tuple[str, bytes, int | None]] = []
        self.fail = False

    async def get(self, key: str) -> bytes | None:
        if self.fail:
            from redis.exceptions import ConnectionError as RedisConnectionError

            raise RedisConnectionError("fake")
        if self.get_returns is not None:
            return self.get_returns
        return self.store.get(key)

    async def set(self, key: str, value: bytes | str, ex: int | None = None) -> bool:
        if isinstance(value, str):
            value = value.encode("utf-8")
        self.store[key] = value
        self.set_calls.append((key, value, ex))
        return True


class _FakeUsage:
    prompt_token_count = 1
    candidates_token_count = 1


class _FakeResponse:
    def __init__(self, *, text: str | None = None, parsed: Any = None) -> None:
        self.text = text
        self.parsed = parsed
        self.usage_metadata = _FakeUsage()
        self.candidates: list[Any] = []


class _FakeModels:
    def __init__(self, response: Any, *, sleep_s: float = 0.0) -> None:
        self.response = response
        self.sleep_s = sleep_s
        self.calls = 0

    async def generate_content(self, **_kwargs: Any) -> Any:
        self.calls += 1
        if self.sleep_s > 0:
            await asyncio.sleep(self.sleep_s)
        return self.response


class _FakeSdk:
    def __init__(self, models: _FakeModels) -> None:
        class _Aio:
            pass

        aio = _Aio()
        aio.models = models  # type: ignore[attr-defined]
        self.aio = aio


@pytest.fixture(autouse=True)
def _reset() -> None:
    client_mod.reset_for_tests()
    yield
    client_mod.reset_for_tests()


def _inject_client(response: Any, *, sleep_s: float = 0.0) -> _FakeModels:
    models = _FakeModels(response, sleep_s=sleep_s)
    sdk = _FakeSdk(models)
    client_mod.set_client_for_tests(client_mod.GeminiClient(sdk=sdk))
    return models


# ---------------------------------------------------------------- tests


async def test_cache_hit_skips_llm_call() -> None:
    cached = b'{"category": "shoes", "attrs": ["running"], "price_bias": null}'
    redis_fake = _FakeRedis(get_returns=cached)
    models = _inject_client(_FakeResponse(text='{"category": "ignored"}'))

    result = await understand_query("running shoes", redis_client=redis_fake, cfg=LLMConfig())
    assert result is not None
    assert result.category == "shoes"
    assert models.calls == 0  # never went to the LLM


async def test_cache_miss_calls_llm_then_setex() -> None:
    redis_fake = _FakeRedis()
    models = _inject_client(
        _FakeResponse(text='{"category": "headphones", "attrs": ["wireless"], "price_bias": "low"}')
    )

    cfg = LLMConfig()
    result = await understand_query("cheap wireless headphones", redis_client=redis_fake, cfg=cfg)
    assert result is not None
    assert result.category == "headphones"
    assert result.price_bias == schemas.PriceBias.low
    assert models.calls == 1
    # SETEX written with the configured TTL.
    assert len(redis_fake.set_calls) == 1
    _, _, ttl = redis_fake.set_calls[0]
    assert ttl == cfg.query_understanding_cache_ttl


async def test_timeout_returns_none_no_cache_pollute() -> None:
    redis_fake = _FakeRedis()
    _inject_client(_FakeResponse(text='{"category": "x"}'), sleep_s=0.5)

    result = await understand_query(
        "anything",
        redis_client=redis_fake,
        cfg=LLMConfig(query_understanding_timeout=0.05),
    )
    assert result is None
    # Timeouts must not write to the cache.
    assert redis_fake.set_calls == []


async def test_redis_unavailable_falls_through_to_llm() -> None:
    redis_fake = _FakeRedis()
    redis_fake.fail = True
    models = _inject_client(
        _FakeResponse(text='{"category": "laptops", "attrs": [], "price_bias": null}')
    )

    result = await understand_query("laptops", redis_client=redis_fake, cfg=LLMConfig())
    assert result is not None
    assert result.category == "laptops"
    assert models.calls == 1
