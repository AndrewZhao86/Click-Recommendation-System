"""Singleton wrapper around `google.genai.Client` with timeout discipline.

Mirrors the lazy-singleton + `asyncio.Lock` pattern in
`click_rec.ranker.embedder` — first call constructs the client, all
subsequent calls reuse it. The Gemini API is HTTP-only, so unlike
`embedder.warm()` we do *not* pre-warm in the FastAPI lifespan; a
startup HTTP roundtrip is more expensive than the cold-call tax it
would amortise.

Timeout discipline: the `google-genai` SDK has a documented bug where
its internal `httpx` timeout doesn't propagate (python-genai #911), so
every call is wrapped in `asyncio.wait_for(...)` — the only reliable
cancellation. On `TimeoutError` we increment `llm_timeout_total`,
emit a structured log, and surface a `LLMTimeoutError` to the caller;
the caller is responsible for fail-open (re-rank → hybrid order,
explain → 503).

Token usage is read from `response.usage_metadata` and emitted to
`llm_token_usage_total{use_case,kind}` so dashboards can answer
"are we burning the free-tier quota?" without a separate ledger.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ValidationError

from click_rec.config import get_settings
from click_rec.telemetry.metrics import (
    llm_request_latency_seconds,
    llm_request_total,
    llm_timeout_total,
    llm_token_usage_total,
)

if TYPE_CHECKING:  # pragma: no cover
    from google.genai import Client as GenAIClient

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class LLMResult:
    """Outcome of a single LLM call.

    `parsed` is the parsed Pydantic model (for structured calls), the
    raw text (for `generate_text`), or `None` on parse error.
    """

    parsed: Any
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float


class LLMError(Exception):
    """Base for LLM-layer errors."""


class LLMTimeoutError(LLMError):
    """`asyncio.wait_for` cancelled the underlying SDK call."""


class LLMUnavailable(LLMError):
    """No `GOOGLE_API_KEY` / `GEMINI_API_KEY` configured."""


class LLMParseError(LLMError):
    """The SDK returned but the payload didn't fit the schema."""


# ----------------------------------------------------------------- singleton


_client: GeminiClient | None = None
_client_lock: asyncio.Lock | None = None


def _get_lock() -> asyncio.Lock:
    """Lazy-create the lock so module import stays off-loop."""
    global _client_lock
    if _client_lock is None:
        _client_lock = asyncio.Lock()
    return _client_lock


async def get_client() -> GeminiClient:
    """Return the process-local `GeminiClient`, constructing on first call."""
    global _client
    if _client is not None:
        return _client
    lock = _get_lock()
    async with lock:
        if _client is not None:
            return _client
        settings = get_settings()
        # Settings.__init__ already mirrors GEMINI_API_KEY -> GOOGLE_API_KEY
        # via the lru_cache wrapper, so either env name works here.
        api_key = settings.gemini_api_key or _env_google_key()
        if not api_key:
            raise LLMUnavailable(
                "GOOGLE_API_KEY / GEMINI_API_KEY unset — LLM features are disabled"
            )
        # Lazy import: the google-genai package pulls in protobuf and
        # several MB of transitive deps. Loading at module import would
        # tax `pytest --collect-only` and the `--help` path.
        from google import genai

        sdk = genai.Client(api_key=api_key)
        _client = GeminiClient(sdk=sdk)
        return _client


def _env_google_key() -> str:
    import os

    return os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or ""


def reset_for_tests() -> None:
    """Clear singleton + lock state. Mirrors `embedder.reset_for_tests`.

    Required so `pytest -p no:randomly` and reordered runs don't leak
    a real client / a stubbed client across tests.
    """
    global _client, _client_lock
    _client = None
    _client_lock = None


def set_client_for_tests(client: GeminiClient) -> None:
    """Inject a fake client. Test helper, not part of the public surface."""
    global _client
    _client = client


# ----------------------------------------------------------------- client


class GeminiClient:
    """Thin async wrapper enforcing timeout + metrics on every call."""

    def __init__(self, *, sdk: GenAIClient) -> None:
        self._sdk = sdk

    async def generate_json(
        self,
        *,
        model: str,
        prompt: str,
        schema: type[BaseModel],
        timeout: float,
        use_case: str,
    ) -> LLMResult:
        """Structured-output call. Returns `LLMResult.parsed = schema(...)`.

        On timeout: raises `LLMTimeoutError`.
        On parse failure: raises `LLMParseError`.
        On any other SDK error: raises `LLMError` (`outcome="error"`).
        """
        # Lazy import for the same reason as `get_client`.
        from google.genai import types as genai_types

        cfg = genai_types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
        )
        return await self._call(
            model=model,
            prompt=prompt,
            config=cfg,
            timeout=timeout,
            use_case=use_case,
            schema=schema,
        )

    async def generate_text(
        self,
        *,
        model: str,
        prompt: str,
        timeout: float,
        use_case: str,
        max_output_tokens: int | None = None,
    ) -> LLMResult:
        """Plain-text generation. Returns `LLMResult.parsed = str`."""
        from google.genai import types as genai_types

        cfg = genai_types.GenerateContentConfig(
            response_mime_type="text/plain",
            max_output_tokens=max_output_tokens,
        )
        return await self._call(
            model=model,
            prompt=prompt,
            config=cfg,
            timeout=timeout,
            use_case=use_case,
            schema=None,
        )

    async def _call(
        self,
        *,
        model: str,
        prompt: str,
        config: Any,
        timeout: float,
        use_case: str,
        schema: type[BaseModel] | None,
    ) -> LLMResult:
        start = time.monotonic()
        try:
            response = await asyncio.wait_for(
                self._sdk.aio.models.generate_content(model=model, contents=prompt, config=config),
                timeout=timeout,
            )
        except TimeoutError as exc:
            elapsed_ms = (time.monotonic() - start) * 1000.0
            llm_timeout_total.labels(use_case=use_case).inc()
            llm_request_total.labels(use_case=use_case, outcome="timeout").inc()
            llm_request_latency_seconds.labels(use_case=use_case).observe(elapsed_ms / 1000.0)
            logger.warning(
                "llm_timeout",
                extra={"use_case": use_case, "model": model, "timeout_s": timeout},
            )
            raise LLMTimeoutError(f"{use_case} timed out after {timeout}s") from exc
        except Exception as exc:
            elapsed_ms = (time.monotonic() - start) * 1000.0
            llm_request_total.labels(use_case=use_case, outcome="error").inc()
            llm_request_latency_seconds.labels(use_case=use_case).observe(elapsed_ms / 1000.0)
            logger.exception(
                "llm_error",
                extra={"use_case": use_case, "model": model, "error": str(exc)},
            )
            raise LLMError(str(exc)) from exc

        elapsed_ms = (time.monotonic() - start) * 1000.0
        prompt_tok, completion_tok = _extract_token_counts(response)
        if prompt_tok:
            llm_token_usage_total.labels(use_case=use_case, kind="prompt").inc(prompt_tok)
        if completion_tok:
            llm_token_usage_total.labels(use_case=use_case, kind="completion").inc(completion_tok)
        llm_request_latency_seconds.labels(use_case=use_case).observe(elapsed_ms / 1000.0)

        # Parse path.
        if schema is not None:
            parsed = _parse_structured(response, schema, use_case)
            if parsed is None:
                # _parse_structured already incremented parse_error.
                raise LLMParseError(f"{use_case} response did not match schema")
            llm_request_total.labels(use_case=use_case, outcome="success").inc()
            return LLMResult(
                parsed=parsed,
                prompt_tokens=prompt_tok,
                completion_tokens=completion_tok,
                latency_ms=elapsed_ms,
            )

        text = _extract_text(response)
        if not text:
            llm_request_total.labels(use_case=use_case, outcome="parse_error").inc()
            raise LLMParseError(f"{use_case} returned empty text")
        llm_request_total.labels(use_case=use_case, outcome="success").inc()
        return LLMResult(
            parsed=text,
            prompt_tokens=prompt_tok,
            completion_tokens=completion_tok,
            latency_ms=elapsed_ms,
        )


# ----------------------------------------------------------------- helpers


def _extract_text(response: Any) -> str:
    """Return the response text, defensively handling SDK shape drift."""
    text = getattr(response, "text", None)
    if isinstance(text, str) and text:
        return text.strip()
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return ""
    content = getattr(candidates[0], "content", None)
    parts = getattr(content, "parts", None) or []
    chunks: list[str] = []
    for p in parts:
        t = getattr(p, "text", None)
        if isinstance(t, str):
            chunks.append(t)
    return "".join(chunks).strip()


def _parse_structured(response: Any, schema: type[BaseModel], use_case: str) -> BaseModel | None:
    """Decode JSON from the response and validate against `schema`.

    The SDK exposes a `.parsed` attribute when `response_schema=` is set,
    but it's a Pydantic model on some versions and a dict on others —
    coerce defensively. Increments `llm_request_total{outcome="parse_error"}`
    on failure so callers can distinguish it from a transport error.
    """
    parsed_attr = getattr(response, "parsed", None)
    if isinstance(parsed_attr, schema):
        return parsed_attr
    if isinstance(parsed_attr, dict):
        try:
            return schema.model_validate(parsed_attr)
        except ValidationError:
            pass
    text = _extract_text(response)
    if not text:
        llm_request_total.labels(use_case=use_case, outcome="parse_error").inc()
        return None
    try:
        payload = json.loads(text)
        return schema.model_validate(payload)
    except (json.JSONDecodeError, ValidationError) as exc:
        llm_request_total.labels(use_case=use_case, outcome="parse_error").inc()
        logger.warning(
            "llm_parse_error",
            extra={"use_case": use_case, "error": str(exc), "text_preview": text[:200]},
        )
        return None


def _extract_token_counts(response: Any) -> tuple[int, int]:
    """Best-effort `(prompt, completion)` extraction from `usage_metadata`."""
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return 0, 0
    prompt_tok = int(getattr(usage, "prompt_token_count", 0) or 0)
    completion_tok = int(getattr(usage, "candidates_token_count", 0) or 0)
    return prompt_tok, completion_tok
