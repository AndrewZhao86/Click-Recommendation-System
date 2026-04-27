"""`GET /explain` backing — short NL rationale for (user, item).

Cached in Redis for 1h by `(user_id, item_id)` to absorb demo refreshes
without burning the free-tier quota. Reuses
`click_rec.ranker.features.load_user_context` (recent click history) and
`click_rec.cache.cache_aside.get_item_cached` (item metadata) so the
endpoint pays at most one Postgres roundtrip — and zero on a warm cache.

Failure semantics: any LLM-side failure (timeout, parse, unavailable)
returns `None`. The HTTP layer surfaces that as 503 with `Retry-After`.
"""

from __future__ import annotations

import logging
from collections import Counter as CCounter
from typing import TYPE_CHECKING

from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import RedisError
from redis.exceptions import TimeoutError as RedisTimeoutError
from sqlalchemy import text as sql_text
from sqlalchemy.ext.asyncio import AsyncSession

from click_rec.cache.cache_aside import get_item_cached
from click_rec.llm.client import (
    LLMError,
    LLMTimeoutError,
    LLMUnavailable,
    get_client,
)
from click_rec.llm.config import LLMConfig, runtime_model
from click_rec.llm.prompts import EXPLAIN_PROMPT
from click_rec.ranker.config import load_ranker_config
from click_rec.ranker.features import load_user_context
from click_rec.telemetry.metrics import (
    llm_cache_hit_total,
    llm_fallback_total,
)

if TYPE_CHECKING:  # pragma: no cover
    import redis.asyncio as redis

logger = logging.getLogger(__name__)

_REDIS_ERRORS = (RedisConnectionError, RedisTimeoutError, RedisError)
_USE_CASE = "explain"


def _cache_key(user_id: str, item_id: str) -> str:
    return f"llm:explain:{user_id}:{item_id}"


_USER_RECENT_SQL = sql_text(
    """
    SELECT category, brand, price
      FROM item
     WHERE id = ANY(:ids)
    """
)


async def _build_user_history_block(
    *,
    user_id: str,
    session: AsyncSession,
    redis_client: redis.Redis | None,
) -> str:
    """Histogram of recent categories + brands + average price.

    Mirrors the user-profile-summary that the re-ranker consumes; we
    inline it here instead of importing to keep the explain prompt
    independent of any future re-ranker prompt drift.
    """
    cfg = load_ranker_config()
    ctx = await load_user_context(
        user_id=user_id,
        session=session,
        redis_client=redis_client,
        cfg=cfg,
    )
    if not ctx.recent_item_ids:
        return "(no recent activity)"

    res = await session.execute(_USER_RECENT_SQL, {"ids": ctx.recent_item_ids})
    rows = res.mappings().all()
    if not rows:
        return "(no recent activity)"

    cats = CCounter(r["category"] for r in rows if r.get("category"))
    brands = CCounter(r["brand"] for r in rows if r.get("brand"))

    parts: list[str] = []
    if cats:
        top_cats = ", ".join(f"{c} (x{n})" for c, n in cats.most_common(3))
        parts.append(f"top categories: {top_cats}")
    if brands:
        top_brands = ", ".join(f"{b} (x{n})" for b, n in brands.most_common(3))
        parts.append(f"top brands: {top_brands}")
    if ctx.avg_price is not None:
        parts.append(f"avg recent price ~${ctx.avg_price:.0f}")
    return "; ".join(parts) if parts else "(no recent activity)"


async def explain_recommendation(
    *,
    user_id: str,
    item_id: str,
    session: AsyncSession,
    redis_client: redis.Redis | None,
    cfg: LLMConfig,
) -> str | None:
    """Return a 1-2 sentence rationale, or `None` on any failure path."""
    key = _cache_key(user_id, item_id)

    if redis_client is not None:
        try:
            raw = await redis_client.get(key)
        except _REDIS_ERRORS as exc:
            logger.warning("llm_explain_cache_get_failed", extra={"error": str(exc)})
            raw = None
        if raw is not None:
            llm_cache_hit_total.labels(use_case=_USE_CASE).inc()
            if isinstance(raw, bytes):
                return raw.decode("utf-8")
            return str(raw)

    item = await get_item_cached(item_id, session)
    if item is None:
        # Treat unknown item as fallback — caller surfaces 503 (rather
        # than 404) because explain is a soft feature; the recommended
        # item id may simply be stale.
        llm_fallback_total.labels(use_case=_USE_CASE, reason="error").inc()
        return None

    history_block = await _build_user_history_block(
        user_id=user_id, session=session, redis_client=redis_client
    )
    item_block = (
        f"id={item.id} | title={item.title} | category={item.category} | "
        f"brand={item.brand} | price=${item.price:.2f}"
    )

    try:
        client = await get_client()
    except LLMUnavailable:
        llm_fallback_total.labels(use_case=_USE_CASE, reason="unavailable").inc()
        return None

    prompt = EXPLAIN_PROMPT.format(user_history=history_block, item_block=item_block)
    try:
        result = await client.generate_text(
            model=runtime_model(),
            prompt=prompt,
            timeout=cfg.explain_timeout,
            use_case=_USE_CASE,
            max_output_tokens=200,
        )
    except LLMTimeoutError:
        llm_fallback_total.labels(use_case=_USE_CASE, reason="timeout").inc()
        return None
    except LLMError:
        llm_fallback_total.labels(use_case=_USE_CASE, reason="error").inc()
        return None

    rationale: str = result.parsed

    if redis_client is not None:
        try:
            await redis_client.set(key, rationale, ex=cfg.explain_cache_ttl)
        except _REDIS_ERRORS as exc:
            logger.warning("llm_explain_cache_set_failed", extra={"error": str(exc)})

    return rationale
