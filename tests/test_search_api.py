"""HTTP-layer tests for `GET /search`.

`ASGITransport` bypasses lifespan, so we never start Redis / Kafka /
sentence-transformers in unit tests — `rank()` and the LLM helpers are
patched out and we assert FastAPI's request-validation contract plus
the Phase 8a `use_llm` wiring.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from click_rec.api.app import app
from click_rec.api.routers import search as search_module
from click_rec.llm.schemas import PriceBias, QueryIntent
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
            "bm25": 0.3,
            "vector": 0.2,
            "popularity": 0.0,
            "recency": 0.0,
            "personal": 0.0,
            "co_click": 0.0,
            "price_fit": 0.0,
        },
    )


@pytest.fixture
def stub_rank(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace `ranker.rank` with a recording stub."""
    captured: dict[str, Any] = {}

    async def fake_rank(*, query: str | None, **kwargs: Any) -> list[RankedItemDTO]:
        captured["query"] = query
        captured.update(kwargs)
        return [_ranked("i1", 0.7), _ranked("i2", 0.4)]

    monkeypatch.setattr(search_module, "rank", fake_rank)
    return captured


@pytest.fixture(autouse=True)
def stub_redis(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force `maybe_redis()` to return None so the route's no-Redis path runs."""
    monkeypatch.setattr(search_module, "maybe_redis", lambda: None)


@pytest.fixture(autouse=True)
def stub_user_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip the route's pre-load of `UserContext` — return a stable cold ctx."""

    async def fake_load(**_kwargs: Any) -> UserContext:
        return _empty_ctx()

    monkeypatch.setattr(search_module, "load_user_context", fake_load)


@pytest.fixture
def stub_summary(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace `build_summary` with a recording stub returning a fixed string."""
    captured: dict[str, Any] = {}

    def fake_build(**kwargs: Any) -> str:
        captured.update(kwargs)
        intent = kwargs.get("intent")
        if intent is not None:
            return "recent categories: headphones; intent: category=headphones"
        return "recent categories: headphones"

    monkeypatch.setattr(search_module, "build_summary", fake_build)
    return captured


async def test_search_returns_200_with_breakdowns(
    client: AsyncClient, stub_rank: dict[str, Any]
) -> None:
    resp = await client.get("/search", params={"q": "wireless headphones"})
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, list)
    assert len(body) == 2
    assert body[0]["item"]["id"] == "i1"
    assert body[0]["score"] == 0.7
    breakdown = body[0]["score_breakdown"]
    assert set(breakdown.keys()) == {
        "bm25",
        "vector",
        "popularity",
        "recency",
        "personal",
        "co_click",
        "price_fit",
    }
    assert stub_rank["query"] == "wireless headphones"


async def test_search_default_limit_is_ten(
    client: AsyncClient, stub_rank: dict[str, Any]
) -> None:
    """Phase 8a default is 10 (down from Phase 6's interim 20)."""
    resp = await client.get("/search", params={"q": "x"})
    assert resp.status_code == 200
    # `rank` is asked for at least 20 so the LLM has reorder headroom,
    # but the response is sliced to `limit` (default 10).
    assert stub_rank["limit"] >= 20


async def test_search_empty_query_returns_422(client: AsyncClient) -> None:
    resp = await client.get("/search", params={"q": ""})
    assert resp.status_code == 422


async def test_search_overlong_query_returns_422(client: AsyncClient) -> None:
    resp = await client.get("/search", params={"q": "x" * 201})
    assert resp.status_code == 422


async def test_search_limit_bounds(client: AsyncClient, stub_rank: dict[str, Any]) -> None:
    resp = await client.get("/search", params={"q": "laptop", "limit": 0})
    assert resp.status_code == 422
    resp = await client.get("/search", params={"q": "laptop", "limit": 101})
    assert resp.status_code == 422


async def test_search_passes_user_id_through(
    client: AsyncClient, stub_rank: dict[str, Any]
) -> None:
    resp = await client.get(
        "/search", params={"q": "shoes", "user_id": "u_42"}
    )
    assert resp.status_code == 200
    assert stub_rank["user_id"] == "u_42"


async def test_search_503_when_ranker_raises(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An embedder model-load failure surfaces as 503."""

    async def boom(**_kwargs: Any) -> list[RankedItemDTO]:
        raise RuntimeError("torch broke")

    monkeypatch.setattr(search_module, "rank", boom)
    resp = await client.get("/search", params={"q": "anything"})
    assert resp.status_code == 503


# ============================================================ Phase 8a: use_llm wiring


async def test_search_use_llm_false_skips_llm_call(
    client: AsyncClient,
    stub_rank: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`use_llm=false` must not invoke `understand_query` or `rerank_top_k`."""
    qu_calls: list[Any] = []
    rr_calls: list[Any] = []

    async def fake_qu(*a: Any, **kw: Any) -> None:
        qu_calls.append((a, kw))
        return None

    async def fake_rr(**kw: Any) -> None:
        rr_calls.append(kw)
        return None

    monkeypatch.setattr(search_module, "understand_query", fake_qu)
    monkeypatch.setattr(search_module, "rerank_top_k", fake_rr)

    resp = await client.get("/search", params={"q": "x", "use_llm": "false"})
    assert resp.status_code == 200
    assert qu_calls == []
    assert rr_calls == []


async def test_search_use_llm_true_attaches_rationales(
    client: AsyncClient,
    stub_rank: dict[str, Any],
    stub_summary: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`use_llm=true` reorders + attaches rationales returned by the LLM."""

    async def fake_qu(*a: Any, **kw: Any) -> QueryIntent:
        return QueryIntent(category="headphones", attrs=["wireless"], price_bias=PriceBias.low)

    async def fake_rr(**kw: Any) -> list[RankedItemDTO]:
        # Reverse + add rationales.
        out = []
        for c in reversed(kw["candidates"]):
            out.append(c.model_copy(update={"llm_rationale": f"because {c.item.id}"}))
        return out

    monkeypatch.setattr(search_module, "understand_query", fake_qu)
    monkeypatch.setattr(search_module, "rerank_top_k", fake_rr)

    resp = await client.get(
        "/search", params={"q": "wireless headphones", "use_llm": "true"}
    )
    assert resp.status_code == 200
    body = resp.json()
    # Reversed order: i2 first, then i1.
    assert body[0]["item"]["id"] == "i2"
    assert body[0]["llm_rationale"] == "because i2"
    assert body[1]["llm_rationale"] == "because i1"


async def test_search_intent_appended_to_summary(
    client: AsyncClient,
    stub_rank: dict[str, Any],
    stub_summary: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The intent string must be threaded into the summary passed to the LLM."""
    captured_summary: list[str | None] = []

    async def fake_qu(*a: Any, **kw: Any) -> QueryIntent:
        return QueryIntent(category="headphones", attrs=["wireless"], price_bias=None)

    async def fake_rr(**kw: Any) -> list[RankedItemDTO]:
        captured_summary.append(kw["user_profile_summary"])
        return list(kw["candidates"])

    monkeypatch.setattr(search_module, "understand_query", fake_qu)
    monkeypatch.setattr(search_module, "rerank_top_k", fake_rr)

    resp = await client.get(
        "/search",
        params={"q": "wireless headphones", "use_llm": "true", "user_id": "u1"},
    )
    assert resp.status_code == 200
    assert captured_summary[0] is not None
    assert "intent" in captured_summary[0]
    # The QueryIntent populated by `fake_qu` was passed into `build_summary`.
    assert stub_summary["intent"] is not None
    assert stub_summary["intent"].category == "headphones"


async def test_search_intent_failure_does_not_block_rerank(
    client: AsyncClient,
    stub_rank: dict[str, Any],
    stub_summary: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`understand_query` returning None still runs the re-rank with no intent."""
    rr_called = {"called": False}

    async def fake_qu(*a: Any, **kw: Any) -> None:
        return None  # fail-open

    async def fake_rr(**kw: Any) -> list[RankedItemDTO]:
        rr_called["called"] = True
        return list(kw["candidates"])

    monkeypatch.setattr(search_module, "understand_query", fake_qu)
    monkeypatch.setattr(search_module, "rerank_top_k", fake_rr)

    resp = await client.get(
        "/search", params={"q": "x", "use_llm": "true"}
    )
    assert resp.status_code == 200
    assert rr_called["called"] is True


async def test_search_use_llm_falls_back_when_rerank_returns_none(
    client: AsyncClient,
    stub_rank: dict[str, Any],
    stub_summary: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`rerank_top_k` returning None falls back to the hybrid order."""

    async def fake_qu(*a: Any, **kw: Any) -> None:
        return None

    async def fake_rr(**kw: Any) -> None:
        return None

    monkeypatch.setattr(search_module, "understand_query", fake_qu)
    monkeypatch.setattr(search_module, "rerank_top_k", fake_rr)

    resp = await client.get("/search", params={"q": "x", "use_llm": "true"})
    assert resp.status_code == 200
    body = resp.json()
    # Hybrid order preserved: i1 first.
    assert body[0]["item"]["id"] == "i1"
    assert body[0].get("llm_rationale") is None
