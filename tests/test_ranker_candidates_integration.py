"""Integration tests for BM25 + pgvector candidate generation.

Reuses the testcontainers Postgres+Redis fixture pattern from
`test_cache_aside_integration.py`. Pre-computed embedding fixtures —
no MiniLM load — keep the test below 5s on a cold container.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest

pytestmark = pytest.mark.integration

try:
    import testcontainers  # noqa: F401
    from testcontainers.postgres import PostgresContainer
except ImportError:
    pytest.skip("testcontainers not installed", allow_module_level=True)


@pytest.fixture(scope="module")
def postgres_container() -> Any:
    with PostgresContainer("pgvector/pgvector:pg16") as pc:
        yield pc


@pytest.fixture
def configured_settings(postgres_container: Any) -> Iterator[None]:
    mp = pytest.MonkeyPatch()
    pg_url = postgres_container.get_connection_url().replace(
        "postgresql+psycopg2://", "postgresql+asyncpg://"
    )
    mp.setenv("DATABASE_URL", pg_url)

    from click_rec.config import get_settings

    get_settings.cache_clear()
    yield
    mp.undo()
    get_settings.cache_clear()


@pytest.fixture
async def db_schema(configured_settings: None) -> AsyncIterator[None]:
    from sqlalchemy import text

    from click_rec.db.base import dispose_engine, get_engine
    from click_rec.models import Base

    engine = get_engine()
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await dispose_engine()


def _vec(seed: int, dim: int = 384) -> list[float]:
    """Deterministic synthetic embedding."""
    import numpy as np

    rng = np.random.default_rng(seed)
    v = rng.normal(size=dim).astype("float64")
    # Don't L2-normalise — the production catalog isn't normalised either.
    return v.tolist()


@pytest.fixture
async def seeded_items(db_schema: None) -> list[dict[str, Any]]:
    from datetime import UTC, datetime

    from sqlalchemy import insert

    from click_rec.db.base import get_sessionmaker
    from click_rec.models.item import Item

    rows = [
        {
            "id": "i_wireless_headphones",
            "title": "Sony WH-1000 Wireless Headphones",
            "description": "Noise cancelling wireless headphones for travel.",
            "category": "electronics/headphones",
            "brand": "Sony",
            "price": 299.99,
            "popularity_score": 5.0,
            "embedding": _vec(seed=1),
            "created_at": datetime(2026, 4, 1, tzinfo=UTC),
        },
        {
            "id": "i_running_shoes",
            "title": "Nike Pegasus Running Shoes",
            "description": "Lightweight running shoes for daily training.",
            "category": "apparel/mens-running-shoes",
            "brand": "Nike",
            "price": 129.99,
            "popularity_score": 3.0,
            "embedding": _vec(seed=2),
            "created_at": datetime(2026, 4, 1, tzinfo=UTC),
        },
        {
            "id": "i_blender",
            "title": "Vitamix Pro Blender",
            "description": "Heavy-duty kitchen blender for smoothies.",
            "category": "home/blenders",
            "brand": "Vitamix",
            "price": 449.99,
            "popularity_score": 1.0,
            "embedding": _vec(seed=3),
            "created_at": datetime(2026, 4, 1, tzinfo=UTC),
        },
    ]
    sm = get_sessionmaker()
    async with sm() as session:
        await session.execute(insert(Item), rows)
        await session.commit()
    return rows


# ============================================================ tests


async def test_bm25_returns_hits_for_exact_noun(
    seeded_items: list[dict[str, Any]],
) -> None:
    from click_rec.db.base import get_sessionmaker
    from click_rec.ranker.candidates import _bm25

    sm = get_sessionmaker()
    async with sm() as session:
        rows = await _bm25(session, "wireless headphones", k=10)

    ids = [r[0] for r in rows]
    assert "i_wireless_headphones" in ids
    # BM25 rank for the exact-match item should be > 0.
    score_for_target = next(s for i, s in rows if i == "i_wireless_headphones")
    assert score_for_target > 0


async def test_vector_returns_hits_for_query_vector(
    seeded_items: list[dict[str, Any]],
) -> None:
    """A query vector close to one item's embedding picks that item first."""
    from click_rec.db.base import get_sessionmaker
    from click_rec.ranker.candidates import _vector

    sm = get_sessionmaker()
    target_vec = _vec(seed=1)  # same seed → identical to i_wireless_headphones
    async with sm() as session:
        rows = await _vector(session, target_vec, k=3)

    assert rows[0][0] == "i_wireless_headphones"


async def test_typo_query_falls_through_to_vector_only(
    seeded_items: list[dict[str, Any]],
) -> None:
    """A query with no tsv match still returns vector candidates."""
    from click_rec.db.base import get_sessionmaker
    from click_rec.ranker.candidates import _bm25, generate_candidates
    from click_rec.ranker.config import RankerConfig

    sm = get_sessionmaker()
    async with sm() as session:
        bm25_rows = await _bm25(session, "asdfqwerty", k=10)
    assert bm25_rows == []

    cfg = RankerConfig(bm25_k=10, vector_k=10, candidate_cap=20)
    qvec = _vec(seed=1)
    async with sm() as session:
        cands = await generate_candidates(
            query="asdfqwerty", qvec=qvec, session=session, cfg=cfg
        )
    # Vector channel still surfaces results even when BM25 finds nothing.
    assert len(cands) > 0
    assert all(c.bm25_raw is None for c in cands)
    assert all(c.vector_raw is not None for c in cands)


async def test_dedupe_across_channels(
    seeded_items: list[dict[str, Any]],
) -> None:
    """An item that hits both channels surfaces once with both raw scores."""
    from click_rec.db.base import get_sessionmaker
    from click_rec.ranker.candidates import generate_candidates
    from click_rec.ranker.config import RankerConfig

    sm = get_sessionmaker()
    cfg = RankerConfig(bm25_k=10, vector_k=10, candidate_cap=20)
    qvec = _vec(seed=1)
    async with sm() as session:
        cands = await generate_candidates(
            query="wireless headphones", qvec=qvec, session=session, cfg=cfg
        )

    by_id = {c.item_id: c for c in cands}
    target = by_id.get("i_wireless_headphones")
    assert target is not None
    assert target.bm25_raw is not None
    assert target.vector_raw is not None
