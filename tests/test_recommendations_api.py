"""HTTP-layer tests for `GET /recommendations`.

Mirrors `test_search_api.py`: `ASGITransport` skips lifespan, `rank()`
and the LLM helpers are patched, and we assert FastAPI's
request-validation contract plus the Phase 8a `use_llm` wiring on the
no-query path.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from click_rec.api.app import app
from click_rec.api.routers import recommendations as rec_module
from click_rec.models.schemas import ItemDTO
from click_rec.ranker.features import UserContext
from click_rec.ranker.schemas import RankedItemDTO


def _empty_ctx() -> UserContext:
    return UserContext(
        recent_item_ids=[],
        profile_vec=None,
        avg_price=None,
        recent_categories=["headphones"],
        recent_brands=["Sony"],
    )


@pytest.fixture
async def client() -> AsyncClient:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def _ranked(item_id: str, score: float = 0.5) -> RankedItemDTO:
    return RankedItemDTO(
        item=ItemDTO(
            id=item_id,
            title=f"Item {item_id}",
            description="",
            category="electronics/headphones",
            brand="Acme",
            price=99.99,
            created_at=datetime(2026, 4, 1, tzinfo=UTC),
            popularity_score=0.0,
        ),
        score=score,
        score_breakdown={
            "bm25": 0.0,
            "vector": 0.4,
            "popularity": 0.2,
            "recency": 0.0,
            "personal": 0.3,
            "co_click": 0.0,
            "price_fit": 0.0,
        },
    )


@pytest.fixture
def stub_rank(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    async def fake_rank(*, query: str | None, **kwargs: Any) -> list[RankedItemDTO]:
        captured["query"] = query
        captured.update(kwargs)
        return [_ranked("i1", 0.8), _ranked("i2", 0.5)]

    monkeypatch.setattr(rec_module, "rank", fake_rank)
    return captured


@pytest.fixture(autouse=True)
def stub_redis(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rec_module, "maybe_redis", lambda: None)


@pytest.fixture(autouse=True)
def stub_user_context(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_load(**_kwargs: Any) -> UserContext:
        return _empty_ctx()

    monkeypatch.setattr(rec_module, "load_user_context", fake_load)


@pytest.fixture
def stub_summary(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def fake_build(**kwargs: Any) -> str:
        captured.update(kwargs)
        return "recent categories: headphones; recent brands: Sony"

    monkeypatch.setattr(rec_module, "build_summary", fake_build)
    return captured


# ============================================================ basic contract


async def test_recommendations_validates_user_id_required(client: AsyncClient) -> None:
    """Missing user_id → 422 (it's a required Query)."""
    resp = await client.get("/recommendations")
    assert resp.status_code == 422


async def test_recommendations_returns_top_10_for_warm_user(
    client: AsyncClient, stub_rank: dict[str, Any]
) -> None:
    resp = await client.get(
        "/recommendations", params={"user_id": "u_warm", "limit": 10}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, list)
    assert len(body) == 2  # stub returns 2; route slices to limit.
    # `query=None` must be passed all the way through to `rank()`.
    assert stub_rank["query"] is None
    assert stub_rank["user_id"] == "u_warm"


async def test_recommendations_default_limit_is_ten(
    client: AsyncClient, stub_rank: dict[str, Any]
) -> None:
    resp = await client.get("/recommendations", params={"user_id": "u1"})
    assert resp.status_code == 200
    # Asks `rank()` for ≥20 to give the LLM headroom.
    assert stub_rank["limit"] >= 20


async def test_recommendations_503_when_ranker_raises(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def boom(**_kwargs: Any) -> list[RankedItemDTO]:
        raise RuntimeError("pgvector died")

    monkeypatch.setattr(rec_module, "rank", boom)
    resp = await client.get("/recommendations", params={"user_id": "u1"})
    assert resp.status_code == 503


async def test_recommendations_cold_start_returns_empty(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When `rank()` returns [] (no profile, no recent_clicks, no global
    popularity), the route returns 200 with an empty list — not 503."""

    async def empty_rank(**_kwargs: Any) -> list[RankedItemDTO]:
        return []

    monkeypatch.setattr(rec_module, "rank", empty_rank)
    resp = await client.get("/recommendations", params={"user_id": "u_new"})
    assert resp.status_code == 200
    assert resp.json() == []


# ============================================================ use_llm wiring


async def test_recommendations_with_use_llm_true_attaches_rationales(
    client: AsyncClient,
    stub_rank: dict[str, Any],
    stub_summary: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`rerank_top_k` is called with `query=""` and rationales surface."""
    captured: dict[str, Any] = {}

    async def fake_rr(**kw: Any) -> list[RankedItemDTO]:
        captured.update(kw)
        out = [
            c.model_copy(update={"llm_rationale": f"rec:{c.item.id}"})
            for c in kw["candidates"]
        ]
        return out

    monkeypatch.setattr(rec_module, "rerank_top_k", fake_rr)
    resp = await client.get(
        "/recommendations", params={"user_id": "u1", "use_llm": "true"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body[0]["llm_rationale"] == "rec:i1"
    # The /recommendations contract: empty query + summary intent=None.
    assert captured["query"] == ""
    assert stub_summary["intent"] is None


async def test_recommendations_llm_failure_falls_back_to_hybrid(
    client: AsyncClient,
    stub_rank: dict[str, Any],
    stub_summary: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`rerank_top_k` returning None falls back to the hybrid order."""

    async def fake_rr(**_kw: Any) -> None:
        return None

    monkeypatch.setattr(rec_module, "rerank_top_k", fake_rr)
    resp = await client.get(
        "/recommendations", params={"user_id": "u1", "use_llm": "true"}
    )
    assert resp.status_code == 200
    body = resp.json()
    # Original order preserved.
    assert body[0]["item"]["id"] == "i1"
    assert body[0].get("llm_rationale") is None


async def test_recommendations_use_llm_false_skips_rerank(
    client: AsyncClient,
    stub_rank: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rr_calls: list[Any] = []

    async def fake_rr(**kw: Any) -> None:
        rr_calls.append(kw)
        return None

    monkeypatch.setattr(rec_module, "rerank_top_k", fake_rr)
    resp = await client.get(
        "/recommendations", params={"user_id": "u1", "use_llm": "false"}
    )
    assert resp.status_code == 200
    assert rr_calls == []
