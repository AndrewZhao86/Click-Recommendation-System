"""HTTP-layer tests for `GET /search`.

`ASGITransport` bypasses lifespan, so we never start Redis / Kafka /
sentence-transformers in unit tests — `rank()` is patched out and we
assert FastAPI's request-validation contract directly.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from click_rec.api.app import app
from click_rec.api.routers import search as search_module
from click_rec.models.schemas import ItemDTO
from click_rec.ranker.schemas import RankedItemDTO


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

    async def fake_rank(*, query: str, **kwargs: Any) -> list[RankedItemDTO]:
        captured["query"] = query
        captured.update(kwargs)
        return [_ranked("i1", 0.7), _ranked("i2", 0.4)]

    monkeypatch.setattr(search_module, "rank", fake_rank)
    return captured


@pytest.fixture(autouse=True)
def stub_redis(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force `get_redis()` to RuntimeError so the route's `client=None` path runs."""
    def _raise() -> None:
        raise RuntimeError("redis not started")

    monkeypatch.setattr(search_module, "get_redis", _raise)


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
    # All seven feature keys present — debuggability bullet (plan §8 step 4).
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
