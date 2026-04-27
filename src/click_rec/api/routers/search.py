"""`GET /search` — minimal Phase 6 verification route.

Scope is deliberately narrow: thin wrapper over `ranker.rank()` for
end-to-end verification (`make eval-offline` and the curl smoke tests
in [phase6plan.md](../../../planning/phase6plan.md)). Phase 8a will
replace this route with the full `/recommendations` contract — `use_llm`
flag, latency-budget instrumentation, deeper response model. *Do not*
extend this handler; wire a new route in Phase 8a instead, mirroring the
note on `items.py`.

Notes on validation:
- `min_length=1` makes FastAPI return 422 on `?q=`. Without this,
  `websearch_to_tsquery('')` matches nothing and we'd silently return
  200 + `[]` for an obvious caller bug.
- `max_length=200` blunts a trivial DOS via gigantic tsquery payloads at
  the public edge.
- `use_llm` is **deliberately omitted** in Phase 6 — exposing a flag
  that always 501s would be API noise. Phase 8a introduces it cleanly.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from click_rec.cache.redis_client import get_redis
from click_rec.db.base import get_sessionmaker
from click_rec.ranker import rank
from click_rec.ranker.schemas import RankedItemDTO

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/search", tags=["search"])


async def _session() -> AsyncIterator[AsyncSession]:  # pragma: no cover - thin FastAPI dep
    async with get_sessionmaker()() as session:
        yield session


SessionDep = Annotated[AsyncSession, Depends(_session)]


@router.get("", response_model=list[RankedItemDTO])
async def search(
    session: SessionDep,
    q: Annotated[str, Query(min_length=1, max_length=200)],
    user_id: Annotated[str | None, Query(max_length=128)] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> list[RankedItemDTO]:
    try:
        client = get_redis()
    except RuntimeError:
        client = None

    try:
        return await rank(
            query=q,
            user_id=user_id,
            session=session,
            redis_client=client,
            limit=limit,
        )
    except Exception as exc:
        # `rank()` already fails open on dependency hiccups (Redis,
        # popularity column, etc.), so anything that escapes here is
        # either an embedder model-load failure or a real bug. Log the
        # traceback before converting to 503 — without this, a future
        # KeyError / scorer ValueError would surface as an opaque
        # "ranker unavailable" with no signal in the logs.
        logger.exception(
            "ranker raised — returning 503",
            extra={"query": q, "user_id": user_id},
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ranker unavailable",
        ) from exc
