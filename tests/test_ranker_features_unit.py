"""Unit tests for the Phase 6 feature extraction layer.

Uses an in-process Redis fake that supports the surface the features
module touches: `zrevrange`, `zscore`, and a minimal `pipeline()`. SQL
features (`co_click_scores`) use a hand-rolled async session fake that
records the bound params so we can assert the SQL was called with the
canonical-respecting ANY / ANY pair.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

from click_rec.ranker.config import RankerConfig
from click_rec.ranker.features import (
    co_click_scores,
    personal_scores,
    popularity_scores,
    price_fit_scores,
    recency_scores,
)
from click_rec.ranker.schemas import RankingCandidate

# ---------------------------------------------------------------- fakes


class _FakePipeline:
    def __init__(self, parent: _FakeRedis) -> None:
        self._parent = parent
        self._ops: list[tuple[str, tuple[Any, ...]]] = []

    def zscore(self, key: str, member: str) -> _FakePipeline:
        self._ops.append(("zscore", (key, member)))
        return self

    async def execute(self) -> list[Any]:
        if "execute" in self._parent.fail_on:
            raise RedisConnectionError("fake pipeline failure")
        out: list[Any] = []
        for op, args in self._ops:
            if op == "zscore":
                key, member = args
                out.append(self._parent.zsets.get(key, {}).get(member))
        return out


class _FakeRedis:
    def __init__(self) -> None:
        self.zsets: dict[str, dict[str, float]] = {}
        self.fail_on: set[str] = set()

    def pipeline(self, transaction: bool = False) -> _FakePipeline:  # noqa: ARG002
        return _FakePipeline(self)

    async def zrevrange(self, key: str, start: int, stop: int) -> list[bytes]:
        if "zrevrange" in self.fail_on:
            raise RedisConnectionError("fake zrevrange failure")
        members = self.zsets.get(key, {})
        ordered = sorted(members.items(), key=lambda kv: kv[1], reverse=True)
        # Redis-style inclusive stop semantics.
        sliced = ordered[start : (stop + 1) if stop != -1 else None]
        return [m.encode() for m, _ in sliced]


def _cand(
    item_id: str,
    *,
    category: str = "electronics/headphones",
    brand: str = "Acme",
    price: float = 100.0,
    created_at: datetime | None = None,
    embedding: list[float] | None = None,
    popularity_score: float = 0.0,
) -> RankingCandidate:
    return RankingCandidate(
        item_id=item_id,
        title=f"Item {item_id}",
        description="",
        category=category,
        brand=brand,
        price=price,
        created_at=created_at or datetime(2026, 4, 1, tzinfo=UTC),
        popularity_score=popularity_score,
        embedding=embedding,
        bm25_raw=None,
        vector_raw=None,
    )


# ---------------------------------------------------------------- popularity


async def test_popularity_reads_zscore_per_candidate() -> None:
    fake = _FakeRedis()
    fake.zsets["items:top:electronics/headphones"] = {"i1": 12.0, "i2": 3.5}
    cands = [_cand("i1"), _cand("i2"), _cand("i3")]
    out = await popularity_scores(cands, fake)
    assert out == [12.0, 3.5, 0.0]


async def test_popularity_falls_back_to_db_on_redis_error() -> None:
    fake = _FakeRedis()
    fake.fail_on.add("execute")
    cands = [_cand("i1", popularity_score=42.0), _cand("i2", popularity_score=7.0)]
    out = await popularity_scores(cands, fake)
    assert out == [42.0, 7.0]


async def test_popularity_with_no_redis_uses_db_column() -> None:
    cands = [_cand("i1", popularity_score=2.5)]
    out = await popularity_scores(cands, redis_client=None)
    assert out == [2.5]


# ---------------------------------------------------------------- recency


def test_recency_decays_monotonically() -> None:
    import math

    cfg = RankerConfig(recency_half_life_days=10.0)
    now = datetime.now(tz=UTC)
    cands = [
        _cand("new", created_at=now),
        _cand("mid", created_at=now - timedelta(days=10)),
        _cand("old", created_at=now - timedelta(days=60)),
    ]
    out = recency_scores(cands, cfg)
    assert out[0] > out[1] > out[2]
    # Formula is exp(-t / scale): t=scale → 1/e (the param is a time
    # constant, not a strict 0.5-half-life despite the name).
    assert abs(out[1] - math.exp(-1.0)) < 0.01


# ---------------------------------------------------------------- personal


def test_personal_returns_zeros_for_cold_user() -> None:
    cands = [_cand("i1", embedding=[1.0, 0.0, 0.0])]
    out = personal_scores(cands, profile_vec=None)
    assert out == [0.0]


def test_personal_handles_missing_embedding() -> None:
    cands = [
        _cand("with_emb", embedding=[1.0, 0.0, 0.0]),
        _cand("no_emb", embedding=None),
    ]
    out = personal_scores(cands, profile_vec=[1.0, 0.0, 0.0])
    assert out[0] == pytest.approx(1.0)
    assert out[1] == 0.0


def test_personal_cosine_directions() -> None:
    # Aligned: 1; orthogonal: 0; opposite clamped to 0 (negative cosines
    # would otherwise reward "less anti-aligned" items via min-max).
    cands = [
        _cand("aligned", embedding=[1.0, 0.0]),
        _cand("orthogonal", embedding=[0.0, 1.0]),
        _cand("opposite", embedding=[-1.0, 0.0]),
    ]
    out = personal_scores(cands, profile_vec=[1.0, 0.0])
    assert out[0] == pytest.approx(1.0)
    assert out[1] == pytest.approx(0.0)
    assert out[2] == pytest.approx(0.0)


def test_personal_clamps_negative_cosine_to_zero() -> None:
    """Anti-aligned items get 0, not a negative number."""
    cands = [_cand("anti", embedding=[-0.7, -0.7])]
    out = personal_scores(cands, profile_vec=[1.0, 1.0])
    assert out == [0.0]


# ---------------------------------------------------------------- price_fit


def test_price_fit_peaks_at_user_avg() -> None:
    cfg = RankerConfig(price_fit_scale=50.0)
    cands = [
        _cand("at", price=100.0),
        _cand("near", price=120.0),
        _cand("far", price=400.0),
    ]
    out = price_fit_scores(cands, user_avg_price=100.0, cfg=cfg)
    assert out[0] == pytest.approx(1.0)
    assert out[0] > out[1] > out[2]


def test_price_fit_zero_for_cold_user() -> None:
    cfg = RankerConfig()
    cands = [_cand("i1", price=100.0)]
    assert price_fit_scores(cands, user_avg_price=None, cfg=cfg) == [0.0]


# ---------------------------------------------------------------- co_click


class _FakeSession:
    """Minimal async-session fake: records `execute()` calls + canned rows."""

    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        self.executed: list[dict[str, Any]] = []
        self._rows = rows or []

    async def execute(self, _stmt: Any, params: dict[str, Any] | None = None) -> Any:
        self.executed.append(params or {})

        rows = self._rows

        class _Mappings:
            def all(self_inner) -> list[dict[str, Any]]:
                return rows

        class _Result:
            def mappings(self_inner) -> _Mappings:
                return _Mappings()

        return _Result()


async def test_co_click_empty_for_cold_start_user_skips_db() -> None:
    session = _FakeSession()
    cands = [_cand("i1"), _cand("i2")]
    out = await co_click_scores(cands, recent_item_ids=[], session=session)
    assert out == [0.0, 0.0]
    assert session.executed == []


async def test_co_click_respects_canonicalisation() -> None:
    """Pairs are stored as (min, max). The query must match either direction."""
    session = _FakeSession(
        rows=[
            {"item_a": "i1", "item_b": "z9", "count": 5},  # candidate=i1, recent=z9
            {"item_a": "a0", "item_b": "i2", "count": 3},  # candidate=i2, recent=a0
        ]
    )
    cands = [_cand("i1"), _cand("i2"), _cand("i3")]
    out = await co_click_scores(
        cands, recent_item_ids=["a0", "z9"], session=session
    )
    assert out == [5.0, 3.0, 0.0]
    # The SQL was called with both lists, in either direction.
    params = session.executed[0]
    assert set(params["recent"]) == {"a0", "z9"}
    assert set(params["cands"]) == {"i1", "i2", "i3"}
