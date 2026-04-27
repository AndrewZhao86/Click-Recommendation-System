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

import logging
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import bindparam, text
from sqlalchemy.ext.asyncio import AsyncSession

from click_rec.ranker.config import RankerConfig
from click_rec.ranker.schemas import RankingCandidate

logger = logging.getLogger(__name__)


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


async def generate_candidates(
    *,
    query: str,
    qvec: list[float],
    session: AsyncSession,
    cfg: RankerConfig,
) -> list[RawCandidate]:
    """Run BM25 + vector concurrently, dedupe, truncate to `candidate_cap`.

    Both channels run on the same `AsyncSession`, so they're sequential
    on the wire — gather is still useful here for clean failure
    isolation (one channel raising doesn't tank the other) but is
    strictly serial because asyncpg connections aren't multiplexed.
    Future Phase 8c work could split into two sessions to parallelise.
    """
    import asyncio

    bm25_rows, vector_rows = await asyncio.gather(
        _bm25(session, query, cfg.bm25_k),
        _vector(session, qvec, cfg.vector_k),
    )

    merged: dict[str, RawCandidate] = {}
    for item_id, score in bm25_rows:
        merged[item_id] = RawCandidate(item_id=item_id, bm25_raw=score, vector_raw=None)
    for item_id, score in vector_rows:
        if item_id in merged:
            merged[item_id].vector_raw = score
        else:
            merged[item_id] = RawCandidate(
                item_id=item_id, bm25_raw=None, vector_raw=score
            )

    # Truncation order: keep the union as-is, then cap by total. A
    # smarter strategy (round-robin across channels) is Phase 8b
    # territory; for Phase 6 the cap rarely binds at 100/100 → ≤200.
    cands = list(merged.values())
    if len(cands) > cfg.candidate_cap:
        cands = cands[: cfg.candidate_cap]
    return cands


_FETCH_ROWS_SQL = text(
    """
    SELECT id, title, description, category, brand, price, created_at,
           popularity_score, embedding
      FROM item
     WHERE id = ANY(:ids)
    """
).bindparams(bindparam("ids", expanding=False))


async def fetch_candidate_rows(
    session: AsyncSession, raw: list[RawCandidate]
) -> list[RankingCandidate]:
    """Bulk SELECT all candidate rows in a single round trip.

    Result order matches `raw` order so feature vectors can be assembled
    by index without a separate join-by-id step.
    """
    if not raw:
        return []
    ids = [c.item_id for c in raw]
    res = await session.execute(_FETCH_ROWS_SQL, {"ids": ids})
    rows = res.mappings().all()

    by_id: dict[str, RankingCandidate] = {}
    for row in rows:
        embedding = row["embedding"]
        # pgvector returns numpy.ndarray for the Vector type via the
        # SQLAlchemy adapter. Convert to plain list for downstream code
        # that uses numpy explicitly (avoid type-soup).
        if embedding is not None and not isinstance(embedding, list):
            try:
                embedding = list(embedding)
            except TypeError:
                # Fall back to None; feature path treats it as "no embedding".
                embedding = None
        by_id[row["id"]] = RankingCandidate(
            item_id=row["id"],
            title=row["title"],
            description=row["description"],
            category=row["category"],
            brand=row["brand"],
            price=float(row["price"]),
            created_at=_ensure_datetime(row["created_at"]),
            popularity_score=float(row["popularity_score"] or 0.0),
            embedding=embedding,
            bm25_raw=None,
            vector_raw=None,
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
