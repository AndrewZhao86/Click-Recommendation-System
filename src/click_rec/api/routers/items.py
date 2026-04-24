"""`GET /items/{item_id}` — minimal item lookup for Phase 5 verification.

Scope is deliberately narrow: this exposes `ItemDTO` as-is for the
Locust hit-ratio test and the stampede integration test. Ranking,
personalisation, and `score_breakdown` belong to Phase 6 / 8a — *do not*
extend this handler with them; wire a new route instead.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from click_rec.cache.cache_aside import get_item_cached
from click_rec.db.base import get_sessionmaker
from click_rec.models.schemas import ItemDTO

router = APIRouter(prefix="/items", tags=["items"])


async def _session() -> AsyncIterator[AsyncSession]:  # pragma: no cover - thin FastAPI dep
    async with get_sessionmaker()() as session:
        yield session


SessionDep = Annotated[AsyncSession, Depends(_session)]


@router.get("/{item_id}", response_model=ItemDTO)
async def get_item(item_id: str, session: SessionDep) -> ItemDTO:
    dto = await get_item_cached(item_id, session)
    if dto is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="item not found")
    return dto
