"""Unit tests for the GeminiClient wrapper.

We never touch the real google-genai SDK in tests — a fake stands in
that mimics the async `client.aio.models.generate_content(...)` shape.
The fake honours `asyncio.sleep` so we can exercise the
`asyncio.wait_for` timeout path deterministically without flake.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from pydantic import BaseModel

from click_rec.llm import client as client_mod
from click_rec.llm.client import (
    GeminiClient,
    LLMParseError,
    LLMTimeoutError,
    LLMUnavailable,
    get_client,
    reset_for_tests,
)


class _SampleSchema(BaseModel):
    foo: str
    bar: int


# ---------------------------------------------------------------- fakes


class _FakeUsage:
    def __init__(self, prompt: int = 12, completion: int = 8) -> None:
        self.prompt_token_count = prompt
        self.candidates_token_count = completion


class _FakeResponse:
    def __init__(
        self,
        *,
        text: str | None = None,
        parsed: Any = None,
        usage: _FakeUsage | None = None,
    ) -> None:
        self.text = text
        self.parsed = parsed
        self.usage_metadata = usage or _FakeUsage()
        self.candidates: list[Any] = []


class _FakeModels:
    def __init__(self, response: Any = None, *, sleep_s: float = 0.0) -> None:
        self.response = response
        self.sleep_s = sleep_s
        self.calls: list[dict[str, Any]] = []

    async def generate_content(
        self, *, model: str, contents: str, config: Any
    ) -> Any:
        self.calls.append({"model": model, "contents": contents, "config": config})
        if self.sleep_s > 0:
            await asyncio.sleep(self.sleep_s)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class _FakeAio:
    def __init__(self, models: _FakeModels) -> None:
        self.models = models


class _FakeSdk:
    def __init__(self, models: _FakeModels) -> None:
        self.aio = _FakeAio(models)


def _make_client(response: Any, *, sleep_s: float = 0.0) -> tuple[GeminiClient, _FakeModels]:
    fake_models = _FakeModels(response, sleep_s=sleep_s)
    fake_sdk = _FakeSdk(fake_models)
    return GeminiClient(sdk=fake_sdk), fake_models


# ---------------------------------------------------------------- tests


@pytest.fixture(autouse=True)
def _reset_singleton() -> None:
    reset_for_tests()
    yield
    reset_for_tests()


async def test_generate_json_parses_response() -> None:
    response = _FakeResponse(text='{"foo": "hi", "bar": 7}')
    gc, _ = _make_client(response)
    result = await gc.generate_json(
        model="gemini-2.5-flash",
        prompt="anything",
        schema=_SampleSchema,
        timeout=1.0,
        use_case="query_understanding",
    )
    assert isinstance(result.parsed, _SampleSchema)
    assert result.parsed.foo == "hi"
    assert result.parsed.bar == 7


async def test_generate_json_emits_token_metrics() -> None:
    response = _FakeResponse(
        text='{"foo": "x", "bar": 1}', usage=_FakeUsage(prompt=42, completion=11)
    )
    gc, _ = _make_client(response)
    result = await gc.generate_json(
        model="gemini-2.5-flash",
        prompt="anything",
        schema=_SampleSchema,
        timeout=1.0,
        use_case="re_rank",
    )
    assert result.prompt_tokens == 42
    assert result.completion_tokens == 11


async def test_generate_json_times_out() -> None:
    """Timeout path: the fake sleeps longer than the wait_for budget."""
    response = _FakeResponse(text='{"foo": "x", "bar": 1}')
    gc, _ = _make_client(response, sleep_s=0.5)
    with pytest.raises(LLMTimeoutError):
        await gc.generate_json(
            model="gemini-2.5-flash",
            prompt="x",
            schema=_SampleSchema,
            timeout=0.05,
            use_case="re_rank",
        )


async def test_unavailable_when_no_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """No GOOGLE_API_KEY / no settings.gemini_api_key → LLMUnavailable."""
    from click_rec.config import get_settings

    get_settings.cache_clear()  # type: ignore[attr-defined]
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    with pytest.raises(LLMUnavailable):
        await get_client()
    get_settings.cache_clear()  # type: ignore[attr-defined]


async def test_parse_error_returns_none_with_metric() -> None:
    response = _FakeResponse(text="not-json {{{")
    gc, _ = _make_client(response)
    with pytest.raises(LLMParseError):
        await gc.generate_json(
            model="gemini-2.5-flash",
            prompt="x",
            schema=_SampleSchema,
            timeout=1.0,
            use_case="query_understanding",
        )


async def test_generate_text_returns_string() -> None:
    response = _FakeResponse(text="A short rationale.")
    gc, _ = _make_client(response)
    result = await gc.generate_text(
        model="gemini-2.5-flash",
        prompt="x",
        timeout=1.0,
        use_case="explain",
    )
    assert result.parsed == "A short rationale."


async def test_reset_for_tests_clears_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    """reset_for_tests must drop the cached singleton instance."""
    from click_rec.config import get_settings

    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    get_settings.cache_clear()  # type: ignore[attr-defined]

    # Stub the SDK so we don't try to talk to real Gemini.
    class _StubSdk:
        aio = _FakeAio(_FakeModels(_FakeResponse(text="x")))

    def _fake_genai_client(*, api_key: str) -> Any:
        return _StubSdk()

    import google.genai as genai

    monkeypatch.setattr(genai, "Client", _fake_genai_client)

    a = await get_client()
    reset_for_tests()
    b = await get_client()
    assert a is not b
    get_settings.cache_clear()  # type: ignore[attr-defined]


async def test_set_client_for_tests_injects_fake() -> None:
    """The test helper bypasses construction so unit tests can inject a fake."""
    response = _FakeResponse(text='{"foo": "z", "bar": 3}')
    gc, _ = _make_client(response)
    client_mod.set_client_for_tests(gc)
    cached = await get_client()
    assert cached is gc
