"""`GET /recommendations` — Phase 8a no-query personalised feed.

Same shape as `/search` minus the query string and intent extraction:

- Required `user_id`.
- Pipeline runs in no-query mode: `rank()` skips BM25, uses the user's
  `profile_vec` for ANN candidate gen, falls back to per-category
  Redis top-sets, then to `items:top:_global` on a hard cold start.
- `use_llm=true` runs `rerank_top_k(query="", ...)`. The existing
  RERANK_PROMPT tolerates the empty query — see phase8plan.md "Risks"
  for the follow-up note about a sibling `REC_RERANK_PROMPT`.

When `use_llm=true`, the route loads `UserContext` once and hands it
to both `rank()` and `build_summary()` so the LLM path doesn't pay for
two Redis + Postgres round-trips on the same context.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from click_rec.api.routers._llm_helpers import build_summary, maybe_redis
from click_rec.db.base import get_sessionmaker
from click_rec.llm import load_llm_config, rerank_top_k
from click_rec.ranker import load_ranker_config, load_user_context, rank
from click_rec.ranker.schemas import RankedItemDTO

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/recommendations", tags=["recommendations"])


async def _session() -> AsyncIterator[AsyncSession]:  # pragma: no cover - thin FastAPI dep
    async with get_sessionmaker()() as session:
        yield session


SessionDep = Annotated[AsyncSession, Depends(_session)]


@router.get("", response_model=list[RankedItemDTO])
async def recommendations(
    session: SessionDep,
    user_id: Annotated[str, Query(min_length=1, max_length=128)],
    limit: Annotated[int, Query(ge=1, le=100)] = 10,
    use_llm: Annotated[bool, Query()] = False,
) -> list[RankedItemDTO]:
    redis_client = maybe_redis()

    user_ctx = None
    if use_llm:
        user_ctx = await load_user_context(
            user_id=user_id,
            session=session,
            redis_client=redis_client,
            cfg=load_ranker_config(),
        )

    try:
        hybrid = await rank(
            query=None,
            user_id=user_id,
            session=session,
            redis_client=redis_client,
            limit=max(limit, 20),
            user_ctx=user_ctx,
        )
    except Exception as exc:
        logger.exception(
            "ranker raised — returning 503",
            extra={"user_id": user_id},
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ranker unavailable",
        ) from exc

    if not use_llm or not hybrid:
        return hybrid[:limit]

    cfg_llm = load_llm_config()
    assert user_ctx is not None
    summary = build_summary(user_ctx=user_ctx, intent=None)

    reranked = await rerank_top_k(
        query="",
        candidates=hybrid,
        user_profile_summary=summary,
        cfg=cfg_llm,
    )
    return (reranked or hybrid)[:limit]
