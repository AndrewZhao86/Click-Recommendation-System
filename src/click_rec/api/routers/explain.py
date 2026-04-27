"""`GET /explain` — natural-language rationale for a (user, item) pair.

Scope is deliberately narrow: a thin wrapper over
`explain_recommendation()` for the Phase 7 demo bullet (plan §8 step 7
verify item 2). Caching, fallback, and prompt construction live in
`click_rec.llm.explain` — *do not* extend this handler with ranking or
profile logic; wire a new route in Phase 8a instead, mirroring the
note on `items.py`.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from click_rec.cache.redis_client import get_redis
from click_rec.db.base import get_sessionmaker
from click_rec.llm import explain_recommendation, load_llm_config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/explain", tags=["explain"])


class ExplainResponse(BaseModel):
    rationale: str


async def _session() -> AsyncIterator[AsyncSession]:  # pragma: no cover - thin FastAPI dep
    async with get_sessionmaker()() as session:
        yield session


SessionDep = Annotated[AsyncSession, Depends(_session)]


@router.get("", response_model=ExplainResponse)
async def explain(
    session: SessionDep,
    user_id: Annotated[str, Query(min_length=1, max_length=64)],
    item_id: Annotated[str, Query(min_length=1, max_length=64)],
) -> ExplainResponse:
    try:
        client = get_redis()
    except RuntimeError:
        client = None

    try:
        text = await explain_recommendation(
            user_id=user_id,
            item_id=item_id,
            session=session,
            redis_client=client,
            cfg=load_llm_config(),
        )
    except Exception as exc:
        logger.exception(
            "explain raised — returning 503",
            extra={"user_id": user_id, "item_id": item_id},
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="explain temporarily unavailable",
        ) from exc

    if text is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="explain temporarily unavailable",
        )
    return ExplainResponse(rationale=text)
