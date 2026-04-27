"""Cached structured query intent extraction.

Phase 7 ships the function (and its cache + token bookkeeping); Phase 8a
threads the resulting `QueryIntent` into candidate filters / re-rank
prompts. Failure-mode contract: timeout / parse / unavailable all return
`None`, never raise — the caller falls back to the raw query.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import TYPE_CHECKING

from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import RedisError
from redis.exceptions import TimeoutError as RedisTimeoutError

from click_rec.llm.client import (
    LLMError,
    LLMParseError,
    LLMTimeoutError,
    LLMUnavailable,
    get_client,
)
from click_rec.llm.config import LLMConfig, runtime_model
from click_rec.llm.prompts import QUERY_UNDERSTANDING_PROMPT
from click_rec.llm.schemas import QueryIntent
from click_rec.telemetry.metrics import (
    llm_cache_hit_total,
    llm_fallback_total,
)

if TYPE_CHECKING:  # pragma: no cover
    import redis.asyncio as redis

logger = logging.getLogger(__name__)

_REDIS_ERRORS = (RedisConnectionError, RedisTimeoutError, RedisError)
_USE_CASE = "query_understanding"


def _cache_key(query: str) -> str:
    digest = hashlib.sha256(query.strip().lower().encode("utf-8")).hexdigest()
    return f"llm:qu:{digest}"


async def understand_query(
    query: str,
    *,
    redis_client: redis.Redis | None,
    cfg: LLMConfig,
) -> QueryIntent | None:
    """Extract a structured QueryIntent for `query`. Fail-open returns None."""
    key = _cache_key(query)

    if redis_client is not None:
        try:
            raw = await redis_client.get(key)
        except _REDIS_ERRORS as exc:
            logger.warning("llm_qu_cache_get_failed", extra={"error": str(exc)})
            raw = None
        if raw is not None:
            try:
                payload = json.loads(raw)
                llm_cache_hit_total.labels(use_case=_USE_CASE).inc()
                return QueryIntent.model_validate(payload)
            except Exception:
                # Stale / corrupt cache entry — fall through to a fresh call.
                logger.warning("llm_qu_cache_parse_failed", extra={"key": key})

    try:
        client = await get_client()
    except LLMUnavailable:
        llm_fallback_total.labels(use_case=_USE_CASE, reason="unavailable").inc()
        return None

    prompt = QUERY_UNDERSTANDING_PROMPT.format(query=query)
    try:
        result = await client.generate_json(
            model=runtime_model(),
            prompt=prompt,
            schema=QueryIntent,
            timeout=cfg.query_understanding_timeout,
            use_case=_USE_CASE,
        )
    except LLMTimeoutError:
        llm_fallback_total.labels(use_case=_USE_CASE, reason="timeout").inc()
        return None
    except LLMParseError:
        llm_fallback_total.labels(use_case=_USE_CASE, reason="parse_error").inc()
        return None
    except LLMError:
        llm_fallback_total.labels(use_case=_USE_CASE, reason="error").inc()
        return None

    intent: QueryIntent = result.parsed

    if redis_client is not None:
        try:
            await redis_client.set(
                key,
                json.dumps(intent.model_dump(mode="json")),
                ex=cfg.query_understanding_cache_ttl,
            )
        except _REDIS_ERRORS as exc:
            logger.warning("llm_qu_cache_set_failed", extra={"error": str(exc)})

    return intent
