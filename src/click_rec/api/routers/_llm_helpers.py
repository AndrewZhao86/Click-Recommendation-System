"""Shared helpers for `/search` and `/recommendations` LLM wiring.

Two thin functions both routes need:

- `maybe_redis()` — return the Redis singleton or None when the
  lifespan hasn't started it (tests, health-check probes).
- `build_summary()` — assemble the prompt-ready profile string from a
  pre-loaded `UserContext`, optionally appending an intent hint.

The route is responsible for loading the `UserContext` (once) and
passing it into both `rank()` and `build_summary()`. Doing the load at
the route level avoids a second Redis ZREVRANGE + Postgres SELECT per
`use_llm=true` request — `rank()` would otherwise reload the same
context internally.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from click_rec.llm import build_user_profile_summary

if TYPE_CHECKING:  # pragma: no cover
    import redis.asyncio as redis

    from click_rec.llm.schemas import QueryIntent
    from click_rec.ranker import UserContext


def maybe_redis() -> redis.Redis | None:
    """Return the Redis singleton, or None when not started."""
    from click_rec.cache.redis_client import get_redis

    try:
        return get_redis()
    except RuntimeError:
        return None


def _format_intent(intent: QueryIntent | None) -> str | None:
    """Render `QueryIntent` into the compact `category=…; attrs=…; price_bias=…` form.

    Returns None for "no useful signal" so the caller can omit the
    intent appendix entirely rather than emitting a stub.
    """
    if intent is None:
        return None
    parts: list[str] = []
    if intent.category:
        parts.append(f"category={intent.category}")
    if intent.attrs:
        parts.append(f"attrs={','.join(intent.attrs[:5])}")
    if intent.price_bias is not None:
        parts.append(f"price_bias={intent.price_bias.value}")
    return "; ".join(parts) if parts else None


def build_summary(
    *,
    user_ctx: UserContext,
    intent: QueryIntent | None,
) -> str:
    """Build the prompt-ready user-profile summary, with optional intent hint.

    Synchronous — no IO. Reuses the `UserContext` the route has already
    loaded (and handed to `rank()`), so a `use_llm=true` request makes
    one set of Redis + Postgres round-trips for context, not two.
    """
    base = build_user_profile_summary(
        recent_categories=user_ctx.recent_categories,
        recent_brands=user_ctx.recent_brands,
        avg_price=user_ctx.avg_price,
    )
    intent_str = _format_intent(intent)
    if intent_str:
        return f"{base}; intent: {intent_str}"
    return base
