"""Candidate generation: BM25 + pgvector → dedupe → bulk row fetch.

Two SQL queries run concurrently via `asyncio.gather`, then the result
sets are merged by `item_id` keeping per-channel raw scores as
`Optional[float]`. The merge uses `None` (not 0) for "channel didn't
surface this candidate" because `0.0` is itself a valid BM25 score and
the scorer needs to distinguish the two when min-max-normalising.

`websearch_to_tsquery` is the parser at the public edge — `plainto_tsquery`
blows up on user-typed quotes / operators. The replay path keeps using
`plainto_tsquery` because corpus-generated queries are clean (see
`scripts/replay_clicks.py:158`).

The vector query uses pgvector's `<=>` operator (cosine distance, range
`[0, 2]`) because the IVFFLAT index was built with `vector_cosine_ops`.
We expose `1 - distance` as a similarity in `[-1, 1]` (or `[0, 1]` for
the typical case where vectors are in the same hemisphere).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import RedisError
from redis.exceptions import TimeoutError as RedisTimeoutError
from sqlalchemy import bindparam, text
from sqlalchemy.ext.asyncio import AsyncSession

from click_rec.ranker.config import RankerConfig
from click_rec.ranker.schemas import RankingCandidate
from click_rec.telemetry.metrics import cache_unavailable_total

if TYPE_CHECKING:  # pragma: no cover
    import redis.asyncio as redis

logger = logging.getLogger(__name__)

_REDIS_ERRORS = (RedisConnectionError, RedisTimeoutError, RedisError)
_GLOBAL_TOP_KEY = "items:top:_global"


def _coerce_embedding(emb: object) -> list[float] | None:
    """Normalise whatever pgvector sends back into a plain list[float].

    Raw text() queries bypass the ORM's Vector type, so asyncpg returns
    the pgvector wire representation as a plain string "[v1,v2,...]"
    rather than a numpy array. Calling list() on that string gives
    individual characters — hence the "inhomogeneous shape" crash in
    np.asarray when different embeddings produce different-length char
    lists. This function handles all three cases the DB layer can return.
    """
    if emb is None:
        return None
    if isinstance(emb, list):
        return emb
    if isinstance(emb, str):
        try:
            return [float(x) for x in emb.strip("[]").split(",") if x.strip()]
        except ValueError:
            return None
    try:
        return list(emb)  # numpy ndarray via pgvector asyncpg codec
    except TypeError:
        return None


@dataclass(slots=True)
class RawCandidate:
    item_id: str
    bm25_raw: float | None
    vector_raw: float | None


_BM25_SQL = text(
    """
    SELECT id, ts_rank_cd(tsv, q) AS bm25_score
      FROM item, websearch_to_tsquery('english', :q) AS q
     WHERE tsv @@ q
     ORDER BY bm25_score DESC
     LIMIT :k
    """
)

# `:qvec` is bound as a string ("[0.1, 0.2, ...]") and CAST to vector
# inside the statement because asyncpg + SQLAlchemy cannot natively
# bind a Python list to the pgvector type without the pgvector
# SQLAlchemy adapter being on the path for this specific cast.
_VECTOR_SQL = text(
    """
    SELECT id, 1 - (embedding <=> CAST(:qvec AS vector)) AS vector_score
      FROM item
     WHERE embedding IS NOT NULL
     ORDER BY embedding <=> CAST(:qvec AS vector)
     LIMIT :k
    """
)


def _format_vector(vec: list[float]) -> str:
    """pgvector accepts the textual `[0.1,0.2,...]` form on INSERT/CAST."""
    return "[" + ",".join(f"{v:.6f}" for v in vec) + "]"


async def _bm25(session: AsyncSession, query: str, k: int) -> list[tuple[str, float]]:
    res = await session.execute(_BM25_SQL, {"q": query, "k": k})
    return [(row.id, float(row.bm25_score)) for row in res]


async def _vector(
    session: AsyncSession, qvec: list[float], k: int
) -> list[tuple[str, float]]:
    res = await session.execute(
        _VECTOR_SQL, {"qvec": _format_vector(qvec), "k": k}
    )
    return [(row.id, float(row.vector_score)) for row in res]


async def _global_top_candidates(
    redis_client: redis.Redis | None, k: int
) -> list[RawCandidate]:
    """Hard cold-start candidate pool: top-K from `items:top:_global`.

    Fail-open: any Redis error returns an empty list — the route layer
    surfaces the empty result rather than 5xx-ing. The consumer
    enrichment maintains this key as a single global sorted set bumped
    on every click and capped by the popularity refresher.
    """
    if redis_client is None:
        return []
    try:
        members = await redis_client.zrevrange(_GLOBAL_TOP_KEY, 0, k - 1)
    except _REDIS_ERRORS as exc:
        logger.warning(
            "cache_unavailable: items:top:_global read failed",
            extra={"error": str(exc)},
        )
        cache_unavailable_total.labels(operation="global_top").inc()
        return []
    out: list[RawCandidate] = []
    for raw in members:
        item_id = raw.decode() if isinstance(raw, bytes) else raw
        out.append(RawCandidate(item_id=item_id, bm25_raw=None, vector_raw=None))
    return out


async def _category_top_candidates(
    redis_client: redis.Redis | None, categories: list[str], k_per_cat: int
) -> list[RawCandidate]:
    """Cold-ish-start: top-K from `items:top:{category}` for the user's
    top recent categories. Returns at most `len(categories) * k_per_cat`
    candidates, deduplicated by item_id.
    """
    if redis_client is None or not categories:
        return []
    seen: dict[str, RawCandidate] = {}
    for cat in categories:
        try:
            members = await redis_client.zrevrange(f"items:top:{cat}", 0, k_per_cat - 1)
        except _REDIS_ERRORS as exc:
            logger.warning(
                "cache_unavailable: items:top:{cat} read failed",
                extra={"category": cat, "error": str(exc)},
            )
            cache_unavailable_total.labels(operation="category_top").inc()
            continue
        for raw in members:
            item_id = raw.decode() if isinstance(raw, bytes) else raw
            if item_id not in seen:
                seen[item_id] = RawCandidate(
                    item_id=item_id, bm25_raw=None, vector_raw=None
                )
    return list(seen.values())


async def generate_candidates(
    *,
    query: str | None,
    qvec: list[float] | None,
    session: AsyncSession,
    cfg: RankerConfig,
    profile_vec: list[float] | None = None,
    recent_categories: list[str] | None = None,
    redis_client: redis.Redis | None = None,
) -> list[RawCandidate]:
    """Run BM25 + vector concurrently, dedupe, truncate to `candidate_cap`.

    Modes:

    - **Query mode** (`query is not None`): BM25 + vector ANN against
      `qvec`. The Phase 6 path.
    - **No-query / recommendations mode** (`query is None`): Phase 8a.
      Skips BM25 entirely; uses `profile_vec` as the ANN query vector
      when available, falling back to per-category Redis top-sets, then
      to the global popularity sorted set on a hard cold start.

    Both channels execute concurrently. asyncpg connections are not
    multiplexed, so true parallelism requires a second `AsyncSession` —
    we open one from the module-level sessionmaker for the vector query.
    This collapses the two ~150 ms queries from sequential (~300 ms) to
    the slower of the two (~150 ms).
    """
    if query is not None:
        # Query path needs both channels; if `qvec` is missing we degrade
        # to BM25-only rather than 5xx.
        if qvec is None:
            bm25_rows = await _bm25(session, query, cfg.bm25_k)
            vector_rows: list[tuple[str, float]] = []
        else:
            from click_rec.db.base import get_sessionmaker

            async with get_sessionmaker()() as session2:
                bm25_rows, vector_rows = await asyncio.gather(
                    _bm25(session, query, cfg.bm25_k),
                    _vector(session2, qvec, cfg.vector_k),
                )

        merged: dict[str, RawCandidate] = {}
        for item_id, score in bm25_rows:
            merged[item_id] = RawCandidate(
                item_id=item_id, bm25_raw=score, vector_raw=None
            )
        for item_id, score in vector_rows:
            if item_id in merged:
                merged[item_id].vector_raw = score
            else:
                merged[item_id] = RawCandidate(
                    item_id=item_id, bm25_raw=None, vector_raw=score
                )

        cands = list(merged.values())
        if len(cands) > cfg.candidate_cap:
            cands = cands[: cfg.candidate_cap]
        return cands

    # No-query path. Tier 1: profile-vector ANN. Tier 2: Redis category
    # top-sets. Tier 3: global popularity sorted set.
    if profile_vec is not None:
        vector_rows = await _vector(session, profile_vec, cfg.vector_k)
        cands = [
            RawCandidate(item_id=iid, bm25_raw=None, vector_raw=score)
            for iid, score in vector_rows
        ]
        if cands:
            if len(cands) > cfg.candidate_cap:
                cands = cands[: cfg.candidate_cap]
            return cands

    if recent_categories:
        # Spread the candidate budget evenly across the top categories so
        # one dominant category doesn't crowd out the rest.
        per_cat = max(1, cfg.candidate_cap // max(1, len(recent_categories)))
        cands = await _category_top_candidates(
            redis_client, recent_categories, per_cat
        )
        if cands:
            if len(cands) > cfg.candidate_cap:
                cands = cands[: cfg.candidate_cap]
            return cands

    return await _global_top_candidates(redis_client, cfg.candidate_cap)


# Two SELECT variants. Both deliberately *omit* the `embedding` column —
# transferring it for ~98 candidates × 384 floats is ~150 KB per request,
# and the only consumer was `personal_scores` (cosine vs. user profile
# vector). When `profile_vec` is known at fetch time we push that cosine
# into pgvector via the `<=>` operator and return a single float per row
# (`personal_raw`) instead of the whole vector. When it isn't, the
# personal feature is 0 anyway (cold start), so dropping the column is
# free.
_FETCH_ROWS_SQL = text(
    """
    SELECT id, title, description, category, brand, price, created_at,
           popularity_score
      FROM item
     WHERE id = ANY(:ids)
    """
).bindparams(bindparam("ids", expanding=False))

_FETCH_ROWS_WITH_PERSONAL_SQL = text(
    """
    SELECT id, title, description, category, brand, price, created_at,
           popularity_score,
           1 - (embedding <=> CAST(:pvec AS vector)) AS personal_raw
      FROM item
     WHERE id = ANY(:ids)
    """
).bindparams(bindparam("ids", expanding=False))


async def fetch_candidate_rows(
    session: AsyncSession,
    raw: list[RawCandidate],
    profile_vec: list[float] | None = None,
) -> list[RankingCandidate]:
    """Bulk SELECT all candidate rows in a single round trip.

    Result order matches `raw` order so feature vectors can be assembled
    by index without a separate join-by-id step.

    When `profile_vec` is provided, the per-candidate cosine similarity
    against it is computed in SQL and returned as `personal_raw`,
    avoiding the need to ship the full embedding column over the wire.
    """
    if not raw:
        return []
    ids = [c.item_id for c in raw]
    if profile_vec is not None:
        params = {"ids": ids, "pvec": _format_vector(profile_vec)}
        res = await session.execute(_FETCH_ROWS_WITH_PERSONAL_SQL, params)
    else:
        res = await session.execute(_FETCH_ROWS_SQL, {"ids": ids})
    rows = res.mappings().all()

    by_id: dict[str, RankingCandidate] = {}
    for row in rows:
        personal_raw = (
            float(row["personal_raw"])
            if profile_vec is not None and row.get("personal_raw") is not None
            else None
        )
        by_id[row["id"]] = RankingCandidate(
            item_id=row["id"],
            title=row["title"],
            description=row["description"],
            category=row["category"],
            brand=row["brand"],
            price=float(row["price"]),
            created_at=_ensure_datetime(row["created_at"]),
            popularity_score=float(row["popularity_score"] or 0.0),
            embedding=None,
            bm25_raw=None,
            vector_raw=None,
            personal_raw=personal_raw,
        )

    out: list[RankingCandidate] = []
    for c in raw:
        cand = by_id.get(c.item_id)
        if cand is None:
            # Indexed in BM25/vector but no longer in the table — race
            # with a delete, or test-only schema drift. Drop silently.
            continue
        cand.bm25_raw = c.bm25_raw
        cand.vector_raw = c.vector_raw
        out.append(cand)
    return out


def _ensure_datetime(val: object) -> datetime:
    """Coerce to a datetime — pgsql returns datetime already, but tests
    sometimes inject strings via raw INSERTs."""
    if isinstance(val, datetime):
        return val
    if isinstance(val, str):
        return datetime.fromisoformat(val)
    raise TypeError(f"unexpected created_at type: {type(val).__name__}")
