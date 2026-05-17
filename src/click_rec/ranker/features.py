"""Feature extraction — five signals + user-context loader.

All features here return one float per candidate, in the same order as
the input `cands` list. Cold-start is handled at the feature level so
the scorer never has to special-case missing inputs:

- `personal_score` returns 0.0 for users with no recent clicks.
- `co_click_score` returns 0.0 with no recent clicks (and skips the DB call).
- `price_fit_score` returns 0.0 (not 1.0) when avg price is unknown —
  cold-start users shouldn't get a free boost from a default-1.0 column.

Redis errors fall open: log `cache_unavailable`, increment
`cache_unavailable_total{operation=...}`, and substitute a sensible
default (Postgres `popularity_score` for the popularity feature; cold
context for `load_user_context`).
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
import redis.asyncio as redis
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import RedisError
from redis.exceptions import TimeoutError as RedisTimeoutError
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from click_rec.ranker.candidates import _coerce_embedding
from click_rec.ranker.config import RankerConfig
from click_rec.ranker.schemas import RankingCandidate
from click_rec.telemetry.metrics import cache_unavailable_total

logger = logging.getLogger(__name__)

_REDIS_ERRORS = (RedisConnectionError, RedisTimeoutError, RedisError)


# ============================================================ user context


@dataclass(slots=True)
class UserContext:
    """Per-request user state. Cold-start = empty list / None / None / [] / [].

    `recent_categories` and `recent_brands` (Phase 8a) are derived from
    the same row set that produces `profile_vec` / `avg_price` — adding
    them costs two extra columns in the same SELECT, no extra round trip.
    Both are top-3-by-frequency over the recent-click rows so the
    re-rank prompt summary stays compact and high-signal.
    """

    recent_item_ids: list[str]
    profile_vec: list[float] | None
    avg_price: float | None
    recent_categories: list[str] = field(default_factory=list)
    recent_brands: list[str] = field(default_factory=list)


_USER_LOAD_SQL = text(
    """
    SELECT id, price, embedding, category, brand
      FROM item
     WHERE id = ANY(:ids)
    """
)


_TOP_RECENT_K = 3


async def load_user_context(
    *,
    user_id: str | None,
    session: AsyncSession,
    redis_client: redis.Redis | None,
    cfg: RankerConfig,
) -> UserContext:
    """Read recent_clicks → bulk SELECT embeddings/prices → average.

    Returns a cold-start `UserContext` when:
    - `user_id is None` (anonymous request).
    - `redis_client is None` (caller couldn't get the singleton).
    - `ZREVRANGE` returns no members.
    - Redis raises (fail-open with metric increment).
    """
    cold = UserContext(recent_item_ids=[], profile_vec=None, avg_price=None)
    if user_id is None or redis_client is None:
        return cold

    n = cfg.personal_recent_clicks_n
    try:
        raw = await redis_client.zrevrange(f"user:{user_id}:recent_clicks", 0, n - 1)
    except _REDIS_ERRORS as exc:
        logger.warning(
            "cache_unavailable: user recent_clicks read failed",
            extra={"user_id": user_id, "error": str(exc)},
        )
        cache_unavailable_total.labels(operation="user_context").inc()
        return cold

    recent_ids = [b.decode() if isinstance(b, bytes) else b for b in raw]
    if not recent_ids:
        return cold

    res = await session.execute(_USER_LOAD_SQL, {"ids": recent_ids})
    rows = res.mappings().all()
    if not rows:
        # Stale recent_clicks (items since deleted). Treat as cold.
        return UserContext(
            recent_item_ids=recent_ids,
            profile_vec=None,
            avg_price=None,
            recent_categories=[],
            recent_brands=[],
        )

    prices = [float(r["price"]) for r in rows if r["price"] is not None]
    avg_price = float(np.mean(prices)) if prices else None

    vecs: list[list[float]] = []
    for r in rows:
        emb = _coerce_embedding(r["embedding"])
        if emb is not None:
            vecs.append(emb)
    if vecs:
        arr = np.asarray(vecs, dtype=np.float64)
        profile_vec = arr.mean(axis=0).tolist()
    else:
        profile_vec = None

    # Top-3 by frequency over the same in-flight rows (no extra round
    # trip). `Counter.most_common` is insertion-stable so ties are broken
    # by first occurrence — deterministic for tests.
    cat_counter = Counter(r["category"] for r in rows if r["category"])
    brand_counter = Counter(r["brand"] for r in rows if r["brand"])
    recent_categories = [c for c, _ in cat_counter.most_common(_TOP_RECENT_K)]
    recent_brands = [b for b, _ in brand_counter.most_common(_TOP_RECENT_K)]

    return UserContext(
        recent_item_ids=recent_ids,
        profile_vec=profile_vec,
        avg_price=avg_price,
        recent_categories=recent_categories,
        recent_brands=recent_brands,
    )


# ============================================================ features


async def popularity_scores(
    cands: list[RankingCandidate], redis_client: redis.Redis | None
) -> list[float]:
    """`ZSCORE items:top:{category} {item_id}` per candidate, pipelined.

    Falls back to the Postgres `popularity_score` column on Redis error
    (or when no client is available — e.g. tests without started Redis).
    Membership-miss in the ZSET (returns None) is treated as 0.0 — the
    item simply isn't a top hit in its category right now.
    """
    if not cands:
        return []
    if redis_client is None:
        return [c.popularity_score for c in cands]

    try:
        pipe = redis_client.pipeline(transaction=False)
        for c in cands:
            pipe.zscore(f"items:top:{c.category}", c.item_id)
        results = await pipe.execute()
    except _REDIS_ERRORS as exc:
        logger.warning(
            "cache_unavailable: popularity ZSCORE failed",
            extra={"error": str(exc)},
        )
        cache_unavailable_total.labels(operation="popularity_zscore").inc()
        return [c.popularity_score for c in cands]

    out: list[float] = []
    for cand, score in zip(cands, results, strict=True):
        if score is None:
            out.append(0.0)
        else:
            try:
                out.append(float(score))
            except (TypeError, ValueError):
                out.append(cand.popularity_score)
    return out


def recency_scores(cands: list[RankingCandidate], cfg: RankerConfig) -> list[float]:
    """Half-life decay on item age in days. Pure Python."""
    now = datetime.now(tz=UTC)
    half_life = cfg.recency_half_life_days
    out: list[float] = []
    for c in cands:
        created = c.created_at
        if created.tzinfo is None:
            created = created.replace(tzinfo=UTC)
        age_days = max(0.0, (now - created).total_seconds() / 86_400.0)
        out.append(math.exp(-age_days / half_life))
    return out


def personal_scores(cands: list[RankingCandidate], profile_vec: list[float] | None) -> list[float]:
    """Cosine similarity between user profile vector and each candidate.

    `profile_vec is None` (cold-start) → all 0.0. Per-candidate missing
    embedding → 0.0 individually (rest of the pool still scored).
    Computed in Python with numpy because the seeder didn't L2-normalise
    the catalog (matches plan §6 scope decision 7) — IVFFLAT uses cosine
    via `<=>`, but a precomputed dot-product over ~200 candidates is
    cheaper than a second SQL round trip.

    Negative cosines are clamped to 0.0 — an item whose embedding points
    *away* from the user's profile shouldn't be silently re-mapped to a
    midrange contribution by per-query min-max. Treating "anti-aligned"
    the same as "no signal" matches user intuition: irrelevant items
    don't earn a recommendation boost.
    """
    if profile_vec is None or not cands:
        return [0.0] * len(cands)

    # Fast path: cosine was computed in pgvector at fetch time and is
    # already on the candidate. Avoids transferring per-row embeddings
    # and the numpy loop below. We require all candidates to have the
    # value — partial fall-back to numpy here would need the embedding
    # column the new SQL no longer fetches.
    if all(c.personal_raw is not None for c in cands):
        return [max(0.0, float(c.personal_raw or 0.0)) for c in cands]

    pv = np.asarray(profile_vec, dtype=np.float64)
    pv_norm = float(np.linalg.norm(pv))
    if pv_norm == 0.0:
        return [0.0] * len(cands)

    out: list[float] = []
    for c in cands:
        if c.embedding is None:
            out.append(0.0)
            continue
        cv = np.asarray(c.embedding, dtype=np.float64)
        cv_norm = float(np.linalg.norm(cv))
        if cv_norm == 0.0:
            out.append(0.0)
            continue
        cos = float(np.dot(pv, cv) / (pv_norm * cv_norm))
        out.append(max(0.0, cos))
    return out


_CO_CLICK_SQL = text(
    """
    SELECT item_a, item_b, count
      FROM co_click
     WHERE (item_a = ANY(:recent) AND item_b = ANY(:cands))
        OR (item_a = ANY(:cands) AND item_b = ANY(:recent))
    """
)


async def co_click_scores(
    cands: list[RankingCandidate],
    recent_item_ids: list[str],
    session: AsyncSession,
) -> list[float]:
    """Sum of co-click counts between each candidate and recent items.

    Cold-start (no recent items) → all 0.0, no DB call.

    The SQL respects the canonicalisation enforced by
    `enrichment.upsert_co_clicks` (`item_a = min(a,b), item_b = max(a,b)`)
    by querying both directions explicitly.
    """
    if not cands or not recent_item_ids:
        return [0.0] * len(cands)

    cand_ids = [c.item_id for c in cands]
    res = await session.execute(_CO_CLICK_SQL, {"recent": recent_item_ids, "cands": cand_ids})
    rows = res.mappings().all()

    recent_set = set(recent_item_ids)
    cand_set = set(cand_ids)
    counts: dict[str, float] = {cid: 0.0 for cid in cand_ids}
    for row in rows:
        a, b, count = row["item_a"], row["item_b"], int(row["count"])
        # Add to whichever side is the candidate. Both items being
        # candidate + recent (a self-link) is impossible because the
        # canonicalisation enforces a < b — so we never double-count.
        if a in cand_set and b in recent_set:
            counts[a] = counts.get(a, 0.0) + float(count)
        if b in cand_set and a in recent_set:
            counts[b] = counts.get(b, 0.0) + float(count)
    return [counts.get(cid, 0.0) for cid in cand_ids]


def price_fit_scores(
    cands: list[RankingCandidate],
    user_avg_price: float | None,
    cfg: RankerConfig,
) -> list[float]:
    """`exp(-|price - avg| / scale)` per candidate.

    Cold-start → 0.0 (not 1.0): a default-1.0 here would artificially
    boost cold-start users above warm ones for an arbitrary baseline.
    """
    if user_avg_price is None or not cands:
        return [0.0] * len(cands)
    scale = cfg.price_fit_scale or 1.0
    return [math.exp(-abs(c.price - user_avg_price) / scale) for c in cands]


# ============================================================ orchestration helper


async def gather_features(
    *,
    cands: list[RankingCandidate],
    user_ctx: UserContext,
    redis_client: redis.Redis | None,
    session: AsyncSession,
    cfg: RankerConfig,
) -> dict[str, list[Any]]:
    """Compute all 7 feature columns for a candidate set.

    `bm25` / `vector` come straight off the candidate (channel raw scores
    survive into the scorer untouched and become first-class normalised
    features there). The other five features are computed here.
    """
    import asyncio

    pop_task = popularity_scores(cands, redis_client)
    cc_task = co_click_scores(cands, user_ctx.recent_item_ids, session)
    pop, cc = await asyncio.gather(pop_task, cc_task)

    recency = recency_scores(cands, cfg)
    personal = personal_scores(cands, user_ctx.profile_vec)
    price_fit = price_fit_scores(cands, user_ctx.avg_price, cfg)

    return {
        "bm25": [c.bm25_raw for c in cands],
        "vector": [c.vector_raw for c in cands],
        "popularity": pop,
        "recency": recency,
        "personal": personal,
        "co_click": cc,
        "price_fit": price_fit,
    }
