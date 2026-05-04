"""Hybrid ranker orchestration — `rank()`.

Pipeline shape:

    encode_query (executor) → BM25 + vector (gather) → bulk fetch + user
    context (gather) → features (gather pop + co_click; pure recency /
    personal / price_fit) → score → sort → top `limit`.

Each stage is timed via `ranker_latency_seconds.labels(stage=…)`.

LightGBM swap point — Phase 8b stretch goal. Replace the
`scorer.score(...)` call with `await rerank_with_lgbm(features,
model_handle)`. Keep the linear scorer as fallback when the model fails
to load. Training: persist replay clicks → LambdaRank objective →
serialise model → load at startup. Search "PHASE_8B_LGBM_SEAM" to
relocate the swap.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from sqlalchemy.ext.asyncio import AsyncSession

from click_rec.models.schemas import ItemDTO
from click_rec.ranker.candidates import (
    fetch_candidate_rows,
    generate_candidates,
)
from click_rec.ranker.config import RankerConfig, load_ranker_config
from click_rec.ranker.embedder import encode_query
from click_rec.ranker.features import UserContext, gather_features, load_user_context
from click_rec.ranker.schemas import RankedItemDTO
from click_rec.ranker.scorer import score as score_features
from click_rec.telemetry.metrics import (
    ranker_candidates_total,
    ranker_cold_start_total,
    ranker_empty_result_total,
    ranker_latency_seconds,
)

if TYPE_CHECKING:  # pragma: no cover
    import redis.asyncio as redis

logger = logging.getLogger(__name__)


def _candidate_to_item_dto(cand: object) -> ItemDTO:
    """Project a `RankingCandidate` into the public `ItemDTO`."""
    # Lazy import to avoid a circular at module import time.
    from click_rec.ranker.schemas import RankingCandidate

    assert isinstance(cand, RankingCandidate)
    return ItemDTO(
        id=cand.item_id,
        title=cand.title,
        description=cand.description,
        category=cand.category,
        brand=cand.brand,
        price=cand.price,
        created_at=cand.created_at,
        popularity_score=cand.popularity_score,
    )


async def rank(
    *,
    query: str | None,
    user_id: str | None,
    session: AsyncSession,
    redis_client: redis.Redis | None,
    limit: int = 20,
    cfg: RankerConfig | None = None,
    user_ctx: UserContext | None = None,
) -> list[RankedItemDTO]:
    """Rank items for a query, optionally personalised by `user_id`.

    Phase 8a: `query` may be None for the `/recommendations` path. With
    no query the pipeline:

    1. Loads the user context first (so `profile_vec` is available).
    2. Skips `encode_query` (the embed stage records 0).
    3. Hands `profile_vec` + `recent_categories` to `generate_candidates`
       which uses pgvector ANN over the profile vector (warm) or
       Redis top-sets (cold).

    `user_ctx` may be supplied by the caller to avoid a duplicate
    `load_user_context` round-trip when the route also needs the same
    context for downstream work (e.g. building the LLM re-rank profile
    summary). When omitted, the pipeline loads it itself.

    Fail-open contract:
    - No candidates → returns `[]`, increments `ranker_empty_result_total`.
    - Redis outage → cold-start path, never raises.
    - Embedder failure (query mode only) → bubbles as a 5xx; the route
      layer decides how to surface it.
    """
    cfg = cfg or load_ranker_config()

    total_start = time.monotonic()

    import asyncio

    if query is None:
        # No-query path: ensure user context (loaded by caller or here)
        # so its profile_vec / recent_categories are available for
        # candidate gen.
        ranker_latency_seconds.labels(stage="embed").observe(0.0)
        cand_start = time.monotonic()
        if user_ctx is None:
            user_ctx = await load_user_context(
                user_id=user_id,
                session=session,
                redis_client=redis_client,
                cfg=cfg,
            )
        raw_cands = await generate_candidates(
            query=None,
            qvec=None,
            session=session,
            cfg=cfg,
            profile_vec=user_ctx.profile_vec,
            recent_categories=user_ctx.recent_categories,
            redis_client=redis_client,
        )
        if not raw_cands:
            ranker_empty_result_total.inc()
            ranker_candidates_total.observe(0)
            ranker_latency_seconds.labels(stage="candidates").observe(
                time.monotonic() - cand_start
            )
            ranker_latency_seconds.labels(stage="total").observe(
                time.monotonic() - total_start
            )
            return []
        ranker_candidates_total.observe(len(raw_cands))
        cands = await fetch_candidate_rows(
            session, raw_cands, profile_vec=user_ctx.profile_vec
        )
        ranker_latency_seconds.labels(stage="candidates").observe(
            time.monotonic() - cand_start
        )
    else:
        # Query path: encode → BM25 + vector → gather context.
        enc_start = time.monotonic()
        qvec = await encode_query(query)
        ranker_latency_seconds.labels(stage="embed").observe(
            time.monotonic() - enc_start
        )

        cand_start = time.monotonic()

        # Run candidate generation and user-context load concurrently so
        # `profile_vec` is available by the time we fetch candidate rows.
        # Pushing the personal-score cosine into the fetch SQL lets us
        # drop the 384-float `embedding` column from the wire payload
        # (~150 KB / request → a single float per row).
        if user_ctx is None:
            from click_rec.db.base import get_sessionmaker

            async with get_sessionmaker()() as session_ctx:
                raw_cands, user_ctx = await asyncio.gather(
                    generate_candidates(
                        query=query,
                        qvec=qvec,
                        session=session,
                        cfg=cfg,
                        redis_client=redis_client,
                    ),
                    load_user_context(
                        user_id=user_id,
                        session=session_ctx,
                        redis_client=redis_client,
                        cfg=cfg,
                    ),
                )
        else:
            raw_cands = await generate_candidates(
                query=query,
                qvec=qvec,
                session=session,
                cfg=cfg,
                redis_client=redis_client,
            )

        if not raw_cands:
            ranker_empty_result_total.inc()
            ranker_candidates_total.observe(0)
            ranker_latency_seconds.labels(stage="total").observe(
                time.monotonic() - total_start
            )
            return []
        ranker_candidates_total.observe(len(raw_cands))

        cands = await fetch_candidate_rows(
            session, raw_cands, profile_vec=user_ctx.profile_vec
        )
        ranker_latency_seconds.labels(stage="candidates").observe(
            time.monotonic() - cand_start
        )

    if not cands:
        ranker_empty_result_total.inc()
        ranker_latency_seconds.labels(stage="total").observe(
            time.monotonic() - total_start
        )
        return []

    if not user_ctx.recent_item_ids:
        ranker_cold_start_total.inc()

    # 4. Feature extraction.
    feat_start = time.monotonic()
    features = await gather_features(
        cands=cands,
        user_ctx=user_ctx,
        redis_client=redis_client,
        session=session,
        cfg=cfg,
    )
    ranker_latency_seconds.labels(stage="features").observe(
        time.monotonic() - feat_start
    )

    # 5. Score + sort + top `limit`.
    # PHASE_8B_LGBM_SEAM — swap `score_features` for the trained
    # LambdaRank model here. The linear path stays as fallback.
    score_start = time.monotonic()
    scores, breakdowns = score_features(features, cfg)
    ranker_latency_seconds.labels(stage="score").observe(
        time.monotonic() - score_start
    )

    indexed = sorted(
        range(len(cands)), key=lambda i: scores[i], reverse=True
    )[:limit]

    out = [
        RankedItemDTO(
            item=_candidate_to_item_dto(cands[i]),
            score=scores[i],
            score_breakdown=breakdowns[i],
        )
        for i in indexed
    ]

    ranker_latency_seconds.labels(stage="total").observe(
        time.monotonic() - total_start
    )
    return out
