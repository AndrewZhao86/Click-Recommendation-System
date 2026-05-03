"""Full-pipeline integration tests with real Postgres + Redis.

The MiniLM model load is mocked away — we inject a deterministic query
vector via `embedder.encode_query` so the test stays under 5s on a cold
container. The candidate-gen, feature-extraction, scoring, and Redis
fail-open paths all run for real.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest

pytestmark = pytest.mark.integration

try:
    import testcontainers  # noqa: F401
    from testcontainers.postgres import PostgresContainer
    from testcontainers.redis import RedisContainer
except ImportError:
    pytest.skip("testcontainers not installed", allow_module_level=True)


@pytest.fixture(scope="module")
def redis_container() -> Any:
    with RedisContainer() as rc:
        yield rc


@pytest.fixture(scope="module")
def postgres_container() -> Any:
    with PostgresContainer("pgvector/pgvector:pg16") as pc:
        yield pc


@pytest.fixture
def configured_settings(
    redis_container: Any, postgres_container: Any
) -> Iterator[None]:
    mp = pytest.MonkeyPatch()
    redis_host = redis_container.get_container_host_ip()
    redis_port = redis_container.get_exposed_port(6379)
    pg_url = postgres_container.get_connection_url().replace(
        "postgresql+psycopg2://", "postgresql+asyncpg://"
    )
    mp.setenv("REDIS_URL", f"redis://{redis_host}:{redis_port}/0")
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


@pytest.fixture
async def started_redis(configured_settings: None) -> AsyncIterator[None]:
    from click_rec.cache.redis_client import start_redis, stop_redis

    await start_redis()
    yield
    with contextlib.suppress(Exception):
        await stop_redis()


def _vec(seed: int, dim: int = 384) -> list[float]:
    import numpy as np

    rng = np.random.default_rng(seed)
    return rng.normal(size=dim).astype("float64").tolist()


@pytest.fixture
async def seeded_world(db_schema: None, started_redis: None) -> dict[str, Any]:
    from datetime import UTC, datetime

    from sqlalchemy import insert

    from click_rec.db.base import get_sessionmaker
    from click_rec.models.item import Item
    from click_rec.models.user import UserAccount

    rows = [
        {
            "id": f"i_hp_{i:02d}",
            "title": f"Wireless Headphones model HP{i}",
            "description": "Bluetooth wireless over-ear headphones.",
            "category": "electronics/headphones",
            "brand": "Sony" if i % 2 == 0 else "Bose",
            "price": 100.0 + i * 10,
            "popularity_score": float(10 - i),
            "embedding": _vec(seed=100 + i),
            "created_at": datetime(2026, 4, 1, tzinfo=UTC),
        }
        for i in range(5)
    ]
    rows.append(
        {
            "id": "i_unrelated",
            "title": "Vitamix Pro Blender",
            "description": "Kitchen blender for smoothies.",
            "category": "home/blenders",
            "brand": "Vitamix",
            "price": 450.0,
            "popularity_score": 1.0,
            "embedding": _vec(seed=999),
            "created_at": datetime(2026, 4, 1, tzinfo=UTC),
        }
    )
    user_rows = [
        {
            "id": "u_test",
            "segment": "category_explorer",
            "country": "US",
        }
    ]
    sm = get_sessionmaker()
    async with sm() as session:
        await session.execute(insert(Item), rows)
        await session.execute(insert(UserAccount), user_rows)
        await session.commit()
    return {"items": rows, "users": user_rows}


@pytest.fixture
def stub_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid loading MiniLM in tests — return a deterministic vector."""
    from click_rec.ranker import embedder, pipeline

    async def fake_encode(query: str) -> list[float]:
        # Match the seed of the first item so vector channel finds it.
        return _vec(seed=100)

    monkeypatch.setattr(pipeline, "encode_query", fake_encode)
    monkeypatch.setattr(embedder, "encode_query", fake_encode)


# ============================================================ tests


async def test_rank_returns_breakdown_with_seven_keys(
    seeded_world: dict[str, Any], stub_embedder: None
) -> None:
    from click_rec.cache.redis_client import get_redis
    from click_rec.db.base import get_sessionmaker
    from click_rec.ranker import rank
    from click_rec.ranker.scorer import FEATURE_NAMES

    sm = get_sessionmaker()
    async with sm() as session:
        results = await rank(
            query="wireless headphones",
            user_id=None,
            session=session,
            redis_client=get_redis(),
            limit=5,
        )
    assert len(results) > 0
    for r in results:
        assert set(r.score_breakdown.keys()) == set(FEATURE_NAMES)
        # Score is the sum of weighted feature contributions.
        assert abs(r.score - sum(r.score_breakdown.values())) < 1e-9


async def test_personalisation_changes_ranking(
    seeded_world: dict[str, Any], stub_embedder: None
) -> None:
    """Two users with different recent_clicks → different personal_score columns.

    We don't assert exact ordering changes (BM25 / vector dominate when
    every candidate is in the same category) — we assert the personal
    column is non-zero for the warm user and zero for cold.
    """
    from click_rec.cache.redis_client import get_redis
    from click_rec.db.base import get_sessionmaker
    from click_rec.ranker import rank

    client = get_redis()
    # Seed a recent_clicks ZSET for the warm user — pick i_hp_00 (matches
    # vec seed=100 → close to query vec).
    await client.zadd("user:u_test:recent_clicks", {"i_hp_00": 1.0})

    sm = get_sessionmaker()
    async with sm() as session:
        cold = await rank(
            query="wireless headphones",
            user_id="u_no_history",
            session=session,
            redis_client=client,
            limit=5,
        )
    async with sm() as session:
        warm = await rank(
            query="wireless headphones",
            user_id="u_test",
            session=session,
            redis_client=client,
            limit=5,
        )

    cold_personal = [r.score_breakdown["personal"] for r in cold]
    warm_personal = [r.score_breakdown["personal"] for r in warm]

    # Cold user has no recent clicks → all zeros.
    assert all(p == 0.0 for p in cold_personal)
    # Warm user has one recent click → at least one non-zero personal contrib.
    assert any(p > 0.0 for p in warm_personal)


async def test_rank_with_no_query_uses_profile_vec(
    seeded_world: dict[str, Any], stub_embedder: None
) -> None:
    """`/recommendations` path: no query → profile_vec ANN candidate gen."""
    from click_rec.cache.redis_client import get_redis
    from click_rec.db.base import get_sessionmaker
    from click_rec.ranker import rank

    client = get_redis()
    # Seed a recent click → load_user_context derives a profile_vec.
    await client.zadd("user:u_test:recent_clicks", {"i_hp_00": 1.0})

    sm = get_sessionmaker()
    async with sm() as session:
        results = await rank(
            query=None,
            user_id="u_test",
            session=session,
            redis_client=client,
            limit=5,
        )
    # With a derived profile vector, ANN returns *some* candidates.
    assert len(results) > 0
    # No query → BM25 channel was skipped, so every breakdown's bm25
    # contribution is 0.0 (`None` raw → min-max normalised to 0).
    for r in results:
        assert r.score_breakdown["bm25"] == 0.0


async def test_rank_no_query_no_profile_falls_back_to_global_top(
    seeded_world: dict[str, Any], stub_embedder: None
) -> None:
    """Hard cold start: no recent_clicks, no profile_vec → items:top:_global."""
    from click_rec.cache.redis_client import get_redis
    from click_rec.db.base import get_sessionmaker
    from click_rec.ranker import rank

    client = get_redis()
    # Seed the global top set with a few items.
    await client.zadd(
        "items:top:_global",
        {"i_hp_00": 5.0, "i_hp_01": 3.0, "i_unrelated": 1.0},
    )

    sm = get_sessionmaker()
    async with sm() as session:
        results = await rank(
            query=None,
            user_id="u_no_history",
            session=session,
            redis_client=client,
            limit=5,
        )
    assert len(results) > 0
    returned_ids = {r.item.id for r in results}
    # At least one of the seeded global-top items came back.
    assert returned_ids & {"i_hp_00", "i_hp_01", "i_unrelated"}


async def test_redis_outage_degrades_gracefully(
    seeded_world: dict[str, Any], stub_embedder: None
) -> None:
    """A Redis hiccup must not 5xx — the pipeline serves with zeros."""
    from click_rec.db.base import get_sessionmaker
    from click_rec.ranker import rank
    from click_rec.telemetry.metrics import cache_unavailable_total

    def _total() -> float:
        total = 0.0
        for metric in cache_unavailable_total.collect():
            for sample in metric.samples:
                if sample.name == "cache_unavailable_total":
                    total += sample.value
        return total

    before = _total()

    from redis.exceptions import ConnectionError as RedisConnectionError

    class _BrokenRedis:
        async def zrevrange(self, *_a: Any, **_kw: Any) -> Any:
            raise RedisConnectionError("dead")

        def pipeline(self, transaction: bool = False) -> Any:  # noqa: ARG002
            class _P:
                def zscore(self_inner, *_a: Any, **_kw: Any) -> Any:
                    return self_inner

                async def execute(self_inner) -> Any:
                    raise RedisConnectionError("dead")
            return _P()

    sm = get_sessionmaker()
    async with sm() as session:
        results = await rank(
            query="wireless headphones",
            user_id="u_test",
            session=session,
            redis_client=_BrokenRedis(),
            limit=5,
        )
    assert len(results) > 0
    after = _total()
    assert after > before, "cache_unavailable_total did not increment under outage"
