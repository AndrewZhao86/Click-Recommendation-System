"""Shared replay-eval helpers used by both `offline.py` and `llm.py`.

This module exists so the two CLIs can share `bm25_only_cfg`, the holdout
dance, the replay-triple loader, and the per-triple ranker scorer without
either CLI reaching into the other's private namespace. Underscore prefix
on the module name signals "internal to the eval package"; the symbols
themselves are public so the type checker stays happy.

Metric formulas (`ndcg_at_k`, `mrr_at_k`) live here too because they're
shared by both passes and have no other natural home.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

from click_rec.ranker import RankerConfig, rank
from click_rec.ranker.schemas import RankedItemDTO

logger = logging.getLogger(__name__)


# ============================================================ metric formulas


def _dcg(rels: list[float]) -> float:
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(rels))


def ndcg_at_k(rels: list[float], k: int) -> float:
    """Standard NDCG@K. Normalises against the ideal sort of the input."""
    if k <= 0 or not rels:
        return 0.0
    actual = _dcg(rels[:k])
    ideal = _dcg(sorted(rels, reverse=True)[:k])
    if ideal == 0.0:
        return 0.0
    return actual / ideal


def mrr_at_k(rels: list[float], k: int) -> float:
    """MRR@K — reciprocal rank of the first relevant hit in [0, k-1]."""
    for i, rel in enumerate(rels[:k]):
        if rel > 0:
            return 1.0 / (i + 1)
    return 0.0


# ============================================================ ranker config helpers


def bm25_only_cfg(cfg: RankerConfig) -> RankerConfig:
    """Zero every weight except BM25 — the baseline reference."""
    return replace(
        cfg,
        w_vector=0.0,
        w_popularity=0.0,
        w_recency=0.0,
        w_personal=0.0,
        w_co_click=0.0,
        w_price_fit=0.0,
    )


# ============================================================ holdout dance


async def holdout_remove(
    redis_client: Any, user_id: str, item_id: str
) -> float | None:
    """ZSCORE → ZREM. Returns the prior score (or None if absent / no Redis)."""
    if redis_client is None:
        return None
    key = f"user:{user_id}:recent_clicks"
    try:
        raw = await redis_client.zscore(key, item_id)
        if raw is None:
            return None
        prior = float(raw)
        await redis_client.zrem(key, item_id)
        return prior
    except Exception:
        logger.exception(
            "holdout: failed to remove %s from %s; eval may be biased",
            item_id,
            key,
        )
        return None


async def holdout_restore(
    redis_client: Any,
    user_id: str,
    item_id: str,
    prior_score: float | None,
) -> None:
    """ZADD back with the original score. No-op when nothing was removed."""
    if redis_client is None or prior_score is None:
        return
    key = f"user:{user_id}:recent_clicks"
    try:
        await redis_client.zadd(key, {item_id: prior_score})
    except Exception:
        logger.exception("holdout: failed to restore %s into %s", item_id, key)


# ============================================================ replay scorer


async def eval_replay_pass(
    *,
    session_factory: Any,
    redis_client: Any,
    cfg: RankerConfig,
    triples: list[tuple[str, str, str]],
    k: int,
) -> dict[str, float]:
    """Score the replay-captured (user, query, clicked) triples.

    Holds the clicked item out of the user's `recent_clicks` ZSET while
    ranking. Without this, `personal_score` averages in the held-out
    item's own embedding (it was published to Kafka and consumed before
    eval ran), which gives the hybrid path a self-similarity boost the
    BM25 baseline doesn't get — biasing the uplift number upward. The
    item's score is captured before ZREM and restored in `finally` so
    the user state isn't mutated across passes.
    """
    if not triples:
        return {"ndcg": 0.0, "mrr": 0.0, "n": 0.0}

    ndcgs: list[float] = []
    mrrs: list[float] = []
    for user_id, query, clicked_item_id in triples:
        prior_score = await holdout_remove(redis_client, user_id, clicked_item_id)
        try:
            async with session_factory() as session:
                results: list[RankedItemDTO] = await rank(
                    query=query,
                    user_id=user_id,
                    session=session,
                    redis_client=redis_client,
                    limit=k,
                    cfg=cfg,
                )
        finally:
            await holdout_restore(
                redis_client, user_id, clicked_item_id, prior_score
            )
        rels = [
            1.0 if r.item.id == clicked_item_id else 0.0 for r in results
        ]
        ndcgs.append(ndcg_at_k(rels, k))
        mrrs.append(mrr_at_k(rels, k))

    return {
        "ndcg": sum(ndcgs) / len(ndcgs),
        "mrr": sum(mrrs) / len(mrrs),
        "n": float(len(triples)),
    }


# ============================================================ data loading


def load_replay_triples(
    path: Path, num_users: int
) -> list[tuple[str, str, str]]:
    """Read replay-captured click events and hold out the LAST per user.

    Captures expected to be JSONL rows with at least:
    `{"user_id": "...", "query": "...", "item_id": "..."}`.
    """
    if not path.exists():
        logger.warning("replay eval log not found at %s — skipping replay pass", path)
        return []

    last_by_user: dict[str, tuple[str, str]] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            user_id = row.get("user_id")
            query = row.get("query")
            item_id = row.get("item_id")
            if not user_id or not query or not item_id:
                continue
            last_by_user[user_id] = (query, item_id)

    triples = [
        (uid, q, iid) for uid, (q, iid) in list(last_by_user.items())[:num_users]
    ]
    return triples


__all__ = [
    "bm25_only_cfg",
    "eval_replay_pass",
    "holdout_remove",
    "holdout_restore",
    "load_replay_triples",
    "mrr_at_k",
    "ndcg_at_k",
]
