"""`GET /search` — Phase 8a full route.

End-to-end wiring of the Phase 6 hybrid ranker plus the optional Phase 7
LLM re-rank, behind a `use_llm` flag. Defaults to top-10 (down from
Phase 6's interim 20) — matches plan §8.8a.

Failure-mode contract:
- LLM unavailable / times out → return the unmodified hybrid order. The
  re-rank helper already increments `llm_fallback_total` in that case.
- Redis unavailable (singleton not started) → run with `client=None`;
  ranker + LLM helpers all tolerate the missing client.
- Anything else escaping `rank()` → 503 with a logged traceback. The
  ranker fails open on dependency hiccups, so escapes here mean an
  embedder load failure or a real bug.

Intent extraction (`understand_query`) is appended to the user-profile
summary as a single string — the LLM treats it as soft context, not a
hard candidate filter. A wrong category from the LLM could wipe the
result set; treating intent as a re-rank hint preserves the
deterministic candidate pool.

When `use_llm=true`, the route loads `UserContext` once and hands it
to both `rank()` and `build_summary()` — without this, `rank()` would
load it internally and `build_summary()` would load it again, doubling
the Redis + Postgres round-trips on the LLM path.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from click_rec.api.routers._llm_helpers import build_summary, maybe_redis
from click_rec.db.base import get_sessionmaker
from click_rec.llm import (
    load_llm_config,
    rerank_top_k,
    understand_query,
)
from click_rec.ranker import load_ranker_config, load_user_context, rank
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
    limit: Annotated[int, Query(ge=1, le=100)] = 10,
    use_llm: Annotated[bool, Query()] = False,
) -> list[RankedItemDTO]:
    redis_client = maybe_redis()

    # On the LLM path we'll need `user_ctx` again to build the prompt
    # summary — load it once at the route and hand it down.
    user_ctx = None
    if use_llm:
        user_ctx = await load_user_context(
            user_id=user_id,
            session=session,
            redis_client=redis_client,
            cfg=load_ranker_config(),
        )

    try:
        # Hybrid is asked for ~2x the response limit so the LLM has
        # headroom to reorder. The Phase 7 LLMConfig.re_rank_input_k
        # caps how many actually go into the prompt.
        hybrid = await rank(
            query=q,
            user_id=user_id,
            session=session,
            redis_client=redis_client,
            limit=max(limit, 20),
            user_ctx=user_ctx,
        )
    except Exception as exc:
        logger.exception(
            "ranker raised — returning 503",
            extra={"query": q, "user_id": user_id},
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ranker unavailable",
        ) from exc

    if not use_llm or not hybrid:
        return hybrid[:limit]

    cfg_llm = load_llm_config()

    # Intent extraction has its own timeout + Redis cache + fail-open
    # contract. A None return is the expected "no signal" path.
    intent = await understand_query(q, redis_client=redis_client, cfg=cfg_llm)

    # `user_ctx` is non-None here because `use_llm=True` always pre-loads.
    assert user_ctx is not None
    summary = build_summary(user_ctx=user_ctx, intent=intent)

    reranked = await rerank_top_k(
        query=q,
        candidates=hybrid,
        user_profile_summary=summary,
        cfg=cfg_llm,
    )
    return (reranked or hybrid)[:limit]
