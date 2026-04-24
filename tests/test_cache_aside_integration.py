"""Integration tests for the Phase 5 cache-aside path.

Real Redis + real Postgres via testcontainers. Asserts the two hard
guarantees that unit fakes can't prove:

1. **Stampede single-query**: 100 concurrent cold-key reads trigger
   exactly one `SELECT` (plan §8 Phase 5 verify block).
2. **Graceful degradation**: stopping the Redis container mid-test
   still returns the DB row and increments `cache_unavailable_total`.
"""

from __future__ import annotations

import asyncio
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


# ---------------------------------------------------------------- fixtures


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
    import contextlib

    from click_rec.cache.redis_client import start_redis, stop_redis

    await start_redis()
    yield
    with contextlib.suppress(Exception):
        await stop_redis()


@pytest.fixture
async def seeded_item(db_schema: None) -> str:
    from sqlalchemy import insert

    from click_rec.db.base import get_sessionmaker
    from click_rec.models.item import Item

    sm = get_sessionmaker()
    async with sm() as session:
        await session.execute(
            insert(Item),
            [
                {
                    "id": "i_cache_int",
                    "title": "Integration test item",
                    "description": "",
                    "category": "electronics",
                    "brand": "Acme",
                    "price": 19.99,
                }
            ],
        )
        await session.commit()
    return "i_cache_int"


# ============================================================ stampede


async def test_stampede_single_query(
    started_redis: None, seeded_item: str,
) -> None:
    """100 concurrent cold-key reads → exactly 1 Postgres SELECT.

    Counts SQL selects via a SQLAlchemy `before_cursor_execute` listener
    that filters for the `item` table — avoiding false positives from
    dialect-probing queries SQLAlchemy issues on engine init.
    """
    from sqlalchemy import event

    from click_rec.cache.cache_aside import get_item_cached
    from click_rec.db.base import get_engine, get_sessionmaker

    engine = get_engine()
    select_count = {"n": 0}

    def _before(conn, cursor, statement, *_args, **_kwargs):  # noqa: ANN001
        if "FROM item" in statement and "WHERE" in statement:
            select_count["n"] += 1

    event.listen(engine.sync_engine, "before_cursor_execute", _before)

    try:
        sm = get_sessionmaker()

        async def one_call() -> Any:
            async with sm() as session:
                return await get_item_cached(seeded_item, session)

        results = await asyncio.gather(*(one_call() for _ in range(100)))
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", _before)

    assert all(r is not None and r.id == seeded_item for r in results)
    assert select_count["n"] == 1, (
        f"expected exactly 1 SELECT, got {select_count['n']}"
    )


# ============================================================ graceful degradation


class TestRedisOutage:
    """Redis-outage test, isolated from the module-scoped Redis container.

    The test body calls `redis_container.stop()` to simulate an outage.
    A module-scoped container would be permanently killed by that stop
    and break every later test in the file; the class-level override
    below narrows `redis_container` to function scope so the damage is
    contained to this one test.

    Pytest's fixture-override semantics propagate this automatically:
    `configured_settings`, `db_schema`, `started_redis`, and
    `seeded_item` all transitively depend on `redis_container` and are
    re-resolved against the class-level override for any test inside
    this class. No duplicate fixture chain is needed.
    """

    @pytest.fixture
    def redis_container(self) -> Iterator[Any]:  # type: ignore[override]
        """Function-scoped Redis override. Suppresses the double-stop.

        The test calls `.stop()` explicitly; testcontainers' own
        teardown then tries to stop+remove an already-gone container,
        which raises `docker.errors.NotFound`. That's expected here —
        the outage is the whole point — so we swallow it.
        """
        import contextlib

        rc = RedisContainer()
        rc.start()
        try:
            yield rc
        finally:
            with contextlib.suppress(Exception):
                rc.stop()

    async def test_redis_outage_still_serves_and_increments_metric(
        self,
        redis_container: Any,
        started_redis: None,
        seeded_item: str,
    ) -> None:
        """Stop Redis mid-test → cache_aside falls through to Postgres,
        returns the item, and `cache_unavailable_total` is incremented.
        """
        from click_rec.cache.cache_aside import get_item_cached
        from click_rec.db.base import get_sessionmaker
        from click_rec.telemetry.metrics import cache_unavailable_total

        def _total() -> float:
            total = 0.0
            # prometheus_client's labels enumerate via `_metrics` internals.
            for metric in cache_unavailable_total.collect():
                for sample in metric.samples:
                    if sample.name == "cache_unavailable_total":
                        total += sample.value
            return total

        before = _total()

        # Stop the container so every subsequent Redis call fails. The
        # `started_redis` fixture's cleanup will later call `stop_redis()`,
        # whose `aclose()` will fail against the dead socket — that's
        # expected and handled by the `contextlib.suppress(Exception)` in
        # the fixture plus the `finally: _redis = None` in `stop_redis`.
        redis_container.stop()

        sm = get_sessionmaker()
        async with sm() as session:
            dto = await get_item_cached(seeded_item, session)

        assert dto is not None and dto.id == seeded_item
        after = _total()
        assert after > before, "cache_unavailable_total did not increment"
