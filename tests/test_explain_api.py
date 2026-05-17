"""HTTP-layer tests for `GET /explain`.

`ASGITransport` bypasses lifespan, so we patch out the underlying
`explain_recommendation` function and assert the route's contract:
validation, 200 with a rationale, 503 when the LLM is unavailable.
"""

from __future__ import annotations

from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from click_rec.api.app import app
from click_rec.api.routers import explain as explain_module


@pytest.fixture
async def client() -> AsyncClient:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture(autouse=True)
def _stub_redis(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force `get_redis()` to RuntimeError so the route's `client=None` path runs."""

    def _raise() -> Any:
        raise RuntimeError("redis not started")

    monkeypatch.setattr(explain_module, "get_redis", _raise)


def _patch_explain(monkeypatch: pytest.MonkeyPatch, return_value: str | None) -> dict[str, int]:
    counters = {"calls": 0}

    async def fake_explain(**_kwargs: Any) -> str | None:
        counters["calls"] += 1
        return return_value

    monkeypatch.setattr(explain_module, "explain_recommendation", fake_explain)
    return counters


async def test_explain_returns_200_with_rationale(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_explain(monkeypatch, "Because you've been browsing Acme headphones.")
    resp = await client.get("/explain", params={"user_id": "u_42", "item_id": "i_1045"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["rationale"].startswith("Because you've been")


async def test_explain_503_when_llm_unavailable(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_explain(monkeypatch, None)
    resp = await client.get("/explain", params={"user_id": "u_42", "item_id": "i_1045"})
    assert resp.status_code == 503


async def test_explain_validates_query_lengths(client: AsyncClient) -> None:
    # Missing required user_id → 422
    resp = await client.get("/explain", params={"item_id": "x"})
    assert resp.status_code == 422
    # Empty user_id → 422 (min_length=1)
    resp = await client.get("/explain", params={"user_id": "", "item_id": "x"})
    assert resp.status_code == 422
    # Overlong → 422 (max_length=64)
    resp = await client.get("/explain", params={"user_id": "u" * 65, "item_id": "x"})
    assert resp.status_code == 422


async def test_explain_503_when_explainer_raises(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unexpected exception inside the explainer surfaces as 503."""

    async def boom(**_kwargs: Any) -> str:
        raise RuntimeError("kaboom")

    monkeypatch.setattr(explain_module, "explain_recommendation", boom)
    resp = await client.get("/explain", params={"user_id": "u_42", "item_id": "i_1045"})
    assert resp.status_code == 503
