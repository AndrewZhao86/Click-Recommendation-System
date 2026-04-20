"""Phase 1 smoke — `/health` responds without requiring the lifespan.

`httpx.AsyncClient(transport=ASGITransport(app))` bypasses lifespan startup,
so this test stays a pure unit test even after Phase 3 wired a Kafka /
Redis-dependent lifespan onto the app.
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from click_rec.api.app import app


@pytest.mark.asyncio
async def test_health() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["env"] == "local"
