"""`make eval-llm` harness — Hybrid baseline vs Hybrid+LLM, plus judge.

Three passes:

1. **Pass A (Hybrid baseline).** Delegate to `run_offline_eval` for
   the deterministic NDCG@10 / MRR@10 numbers Phase 6 already gates on.
2. **Pass B (Hybrid + LLM).** Same replay triples, but each query's
   top-20 hybrid candidates are re-scored with `rerank_top_k`. On
   timeout / parse_error, fall back to the unmodified hybrid order
   and increment `llm_fallback_total` (already done by the re-ranker).
3. **Pass C (LLM-as-judge).** For up to `--judge-sample` queries,
   ask `gemini-2.5-pro` to score the top-10 hybrid vs the top-10
   hybrid+LLM. Reported as a qualitative spot-check; the headline
   number is replay NDCG, not judge mean.

CLI exits 0 always (LLM uplift is qualitative — no hard gate). Negative
NDCG uplift logs a WARNING so it stays visible in CI logs. Mirrors
`run_offline_eval_cli`'s Redis + DB lifecycle ownership so callers can
invoke it as a one-shot script.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import logging
import time
from collections import Counter as CCounter
from pathlib import Path
from typing import Any

from click_rec.cache.redis_client import get_redis, start_redis, stop_redis
from click_rec.config import get_settings
from click_rec.db.base import dispose_engine, get_sessionmaker
from click_rec.eval._replay import (
    bm25_only_cfg,
    eval_replay_pass,
    holdout_remove,
    holdout_restore,
    load_replay_triples,
    mrr_at_k,
    ndcg_at_k,
)
from click_rec.llm import (
    LLMUnavailable,
    build_user_profile_summary,
    get_client,
    judge_ranking,
    load_llm_config,
    rerank_top_k,
)
from click_rec.ranker import RankerConfig, load_ranker_config, rank
from click_rec.ranker.features import load_user_context
from click_rec.ranker.schemas import RankedItemDTO

logger = logging.getLogger(__name__)


# ============================================================ helpers


async def _user_profile_summary(
    *,
    user_id: str,
    session_factory: Any,
    redis_client: Any,
    cfg: RankerConfig,
) -> str:
    """Build a prompt-ready profile summary from the user's recent clicks.

    Reads recent items from Redis via `load_user_context` (same path the
    ranker uses), then aggregates category / brand histograms by joining
    the recent ids back to the catalog.
    """
    async with session_factory() as session:
        ctx = await load_user_context(
            user_id=user_id,
            session=session,
            redis_client=redis_client,
            cfg=cfg,
        )
        if not ctx.recent_item_ids:
            return "(no recent activity)"

        from sqlalchemy import text

        rows = (
            (
                await session.execute(
                    text("SELECT category, brand FROM item WHERE id = ANY(:ids)"),
                    {"ids": ctx.recent_item_ids},
                )
            )
            .mappings()
            .all()
        )

    cats = [r["category"] for r in rows if r.get("category")]
    brands = [r["brand"] for r in rows if r.get("brand")]
    cat_top = [c for c, _ in CCounter(cats).most_common(5)]
    brand_top = [b for b, _ in CCounter(brands).most_common(5)]
    return build_user_profile_summary(
        recent_categories=cat_top,
        recent_brands=brand_top,
        avg_price=ctx.avg_price,
    )


async def _eval_llm_pass(
    *,
    session_factory: Any,
    redis_client: Any,
    ranker_cfg: RankerConfig,
    llm_cfg: Any,
    triples: list[tuple[str, str, str]],
    k: int,
) -> dict[str, float]:
    """Hybrid → LLM rerank → score. Tracks LLM fallback rate per-query."""
    if not triples:
        return {"ndcg": 0.0, "mrr": 0.0, "n": 0.0, "fallbacks": 0.0}

    ndcgs: list[float] = []
    mrrs: list[float] = []
    fallbacks = 0

    for user_id, query, clicked_item_id in triples:
        prior_score = await holdout_remove(redis_client, user_id, clicked_item_id)
        try:
            profile = await _user_profile_summary(
                user_id=user_id,
                session_factory=session_factory,
                redis_client=redis_client,
                cfg=ranker_cfg,
            )
            async with session_factory() as session:
                # Fetch a deeper pool so the re-ranker has room to reorder
                # without changing the deterministic semantics for callers
                # that pass `limit=k`.
                results: list[RankedItemDTO] = await rank(
                    query=query,
                    user_id=user_id,
                    session=session,
                    redis_client=redis_client,
                    limit=max(llm_cfg.re_rank_input_k, k),
                    cfg=ranker_cfg,
                )
            reordered = await rerank_top_k(
                query=query,
                candidates=results,
                user_profile_summary=profile,
                cfg=llm_cfg,
            )
            if reordered is None:
                fallbacks += 1
                final = results[:k]
            else:
                final = reordered[:k]
        finally:
            await holdout_restore(redis_client, user_id, clicked_item_id, prior_score)

        rels = [1.0 if r.item.id == clicked_item_id else 0.0 for r in final]
        ndcgs.append(ndcg_at_k(rels, k))
        mrrs.append(mrr_at_k(rels, k))

    return {
        "ndcg": sum(ndcgs) / len(ndcgs),
        "mrr": sum(mrrs) / len(mrrs),
        "n": float(len(triples)),
        "fallbacks": float(fallbacks),
    }


async def _eval_judge_pass(
    *,
    session_factory: Any,
    redis_client: Any,
    ranker_cfg: RankerConfig,
    llm_cfg: Any,
    triples: list[tuple[str, str, str]],
    k: int,
    sample: int,
) -> dict[str, float]:
    """Score top-10 hybrid vs top-10 hybrid+LLM with `gemini-2.5-pro`.

    `sample` caps the number of queries judged so a free-tier Pro quota
    (~5 RPM / ~100 RPD) doesn't get burned in CI. Returns mean scores
    plus the actual sample size (judge calls that returned None are
    excluded from the mean).
    """
    if not triples or sample <= 0:
        return {
            "judge_hybrid": 0.0,
            "judge_llm": 0.0,
            "judge_n": 0.0,
        }

    chosen = triples[:sample]
    hybrid_scores: list[float] = []
    llm_scores: list[float] = []
    pacing_s = max(0.0, llm_cfg.judge_pacing_seconds)
    judge_calls_made = 0

    async def _paced_judge(payload: list[dict[str, str]]) -> Any:
        """Sleep `pacing_s` before every judge call after the first.

        The free-tier `gemini-2.5-pro` budget is ~5 RPM and we issue two
        calls per query (hybrid + LLM); without pacing a sample of >2
        bursts past the limit and judge calls start failing.
        """
        nonlocal judge_calls_made
        if judge_calls_made > 0 and pacing_s > 0:
            await asyncio.sleep(pacing_s)
        judge_calls_made += 1
        return await judge_ranking(query=query, ranked_top_10=payload, cfg=llm_cfg)

    for user_id, query, _clicked in chosen:
        profile = await _user_profile_summary(
            user_id=user_id,
            session_factory=session_factory,
            redis_client=redis_client,
            cfg=ranker_cfg,
        )
        async with session_factory() as session:
            hybrid_top = await rank(
                query=query,
                user_id=user_id,
                session=session,
                redis_client=redis_client,
                limit=max(llm_cfg.re_rank_input_k, k),
                cfg=ranker_cfg,
            )
        reordered = await rerank_top_k(
            query=query,
            candidates=hybrid_top,
            user_profile_summary=profile,
            cfg=llm_cfg,
        )
        llm_top = (reordered or hybrid_top)[:k]

        hybrid_payload = [_to_judge_payload(r) for r in hybrid_top[:k]]
        llm_payload = [_to_judge_payload(r) for r in llm_top]

        v_hybrid = await _paced_judge(hybrid_payload)
        v_llm = await _paced_judge(llm_payload)
        if v_hybrid is not None:
            hybrid_scores.append(float(v_hybrid.relevance_score))
        if v_llm is not None:
            llm_scores.append(float(v_llm.relevance_score))

    return {
        "judge_hybrid": (sum(hybrid_scores) / len(hybrid_scores)) if hybrid_scores else 0.0,
        "judge_llm": (sum(llm_scores) / len(llm_scores)) if llm_scores else 0.0,
        "judge_n": float(min(len(hybrid_scores), len(llm_scores))),
    }


def _to_judge_payload(r: RankedItemDTO) -> dict[str, str]:
    return {
        "id": r.item.id,
        "title": r.item.title,
        "category": r.item.category,
        "brand": r.item.brand,
    }


# ============================================================ orchestration


async def run_eval_llm(
    *,
    session_factory: Any,
    redis_client: Any,
    num_users: int = 200,
    k: int = 10,
    judge_sample: int = 10,
    eval_log: Path | None = None,
) -> dict[str, float]:
    """Run Pass A (hybrid baseline) + Pass B (hybrid+LLM) + Pass C (judge).

    Pass C is skipped (returns 0 / 0 / 0) when the LLM is unavailable so
    the function still returns a well-shaped metrics dict for callers
    that want to assert on Pass A/B alone.
    """
    settings = get_settings()
    artifacts_dir = Path(settings.eval_output_dir)
    eval_log = eval_log or (artifacts_dir / "eval_clicks.jsonl")

    triples = load_replay_triples(eval_log, num_users)
    ranker_cfg = load_ranker_config()
    llm_cfg = load_llm_config()

    # Pass A — hybrid baseline (delegate to the existing replay scorer).
    bm25_cfg = bm25_only_cfg(ranker_cfg)
    pass_a = await eval_replay_pass(
        session_factory=session_factory,
        redis_client=redis_client,
        cfg=ranker_cfg,
        triples=triples,
        k=k,
    )
    pass_a_bm25 = await eval_replay_pass(
        session_factory=session_factory,
        redis_client=redis_client,
        cfg=bm25_cfg,
        triples=triples,
        k=k,
    )

    # Pass B — hybrid + LLM rerank.
    pass_b = await _eval_llm_pass(
        session_factory=session_factory,
        redis_client=redis_client,
        ranker_cfg=ranker_cfg,
        llm_cfg=llm_cfg,
        triples=triples,
        k=k,
    )

    # Pass C — judge spot-check. Skip if LLM unavailable.
    judge_skipped = False
    try:
        await get_client()
    except LLMUnavailable:
        judge_skipped = True
        pass_c = {"judge_hybrid": 0.0, "judge_llm": 0.0, "judge_n": 0.0}
    if not judge_skipped:
        pass_c = await _eval_judge_pass(
            session_factory=session_factory,
            redis_client=redis_client,
            ranker_cfg=ranker_cfg,
            llm_cfg=llm_cfg,
            triples=triples,
            k=k,
            sample=judge_sample,
        )

    # Uplift: Pass B vs Pass A.
    if pass_a["ndcg"] > 0:
        ndcg_uplift = (pass_b["ndcg"] - pass_a["ndcg"]) / pass_a["ndcg"] * 100.0
    else:
        ndcg_uplift = 0.0

    if pass_c["judge_hybrid"] > 0:
        judge_uplift = (
            (pass_c["judge_llm"] - pass_c["judge_hybrid"]) / pass_c["judge_hybrid"] * 100.0
        )
    else:
        judge_uplift = 0.0

    timeout_rate = (pass_b["fallbacks"] / pass_b["n"] * 100.0) if pass_b["n"] > 0 else 0.0

    return {
        "hybrid_ndcg@10": pass_a["ndcg"],
        "hybrid_mrr@10": pass_a["mrr"],
        "bm25_ndcg@10": pass_a_bm25["ndcg"],
        "bm25_mrr@10": pass_a_bm25["mrr"],
        "llm_ndcg@10": pass_b["ndcg"],
        "llm_mrr@10": pass_b["mrr"],
        "llm_ndcg_uplift_pct": ndcg_uplift,
        "llm_fallback_pct": timeout_rate,
        "judge_hybrid_score_mean": pass_c["judge_hybrid"],
        "judge_llm_score_mean": pass_c["judge_llm"],
        "judge_uplift_pct": judge_uplift,
        "judge_sample_size": pass_c["judge_n"],
        "replay_n": pass_a["n"],
        "judge_skipped": 1.0 if judge_skipped else 0.0,
    }


# ============================================================ CLI


def _print_markdown_table(results: dict[str, float]) -> None:
    print()
    print("| Metric         | BM25   | Hybrid | Hybrid+LLM | Uplift |")
    print("| -------------- | ------ | ------ | ---------- | ------ |")
    print(
        f"| NDCG@10        | {results['bm25_ndcg@10']:.4f} | "
        f"{results['hybrid_ndcg@10']:.4f} | "
        f"{results['llm_ndcg@10']:.4f}     | "
        f"{results['llm_ndcg_uplift_pct']:+.1f}% |"
    )
    print(
        f"| MRR@10         | {results['bm25_mrr@10']:.4f} | "
        f"{results['hybrid_mrr@10']:.4f} | "
        f"{results['llm_mrr@10']:.4f}     |        |"
    )
    print(
        f"| LLM fallback % |        |        | {results['llm_fallback_pct']:.1f}%      |        |"
    )
    if results["judge_sample_size"] > 0:
        print(
            f"| Judge mean     |        | "
            f"{results['judge_hybrid_score_mean']:.2f}   | "
            f"{results['judge_llm_score_mean']:.2f}       | "
            f"{results['judge_uplift_pct']:+.1f}% |"
        )
    elif results["judge_skipped"] > 0:
        print("| Judge mean     | skipped (LLM unavailable)              |")
    print()
    print(f"replay_n={int(results['replay_n'])}  judge_n={int(results['judge_sample_size'])}")


def _write_artifact(results: dict[str, float], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **results,
        "timestamp": int(time.time()),
        "ranker_config": dataclasses.asdict(load_ranker_config()),
        "llm_config": dataclasses.asdict(load_llm_config()),
    }
    output.write_text(json.dumps(payload, indent=2))
    logger.info("wrote eval artifact: %s", output)


async def run_eval_llm_cli(args: argparse.Namespace) -> int:
    """CLI driver — owns Redis + DB lifecycle."""
    settings = get_settings()
    output_arg: str | None = getattr(args, "output", None)
    artifacts_dir = Path(output_arg) if output_arg else Path(settings.eval_output_dir)
    if artifacts_dir.suffix == ".json":
        output_path = artifacts_dir
    else:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        output_path = artifacts_dir / f"eval_llm_{int(time.time())}.json"

    eval_log_path = Path(args.eval_log) if getattr(args, "eval_log", None) else None

    redis_client: Any = None
    redis_started = False
    try:
        await start_redis()
        redis_client = get_redis()
        redis_started = True
    except Exception:
        logger.warning(
            "redis unavailable for eval-llm — running without popularity / personal features"
        )

    try:
        sm = get_sessionmaker()
        results = await run_eval_llm(
            session_factory=sm,
            redis_client=redis_client,
            num_users=args.num_users,
            k=args.k,
            judge_sample=args.judge_sample,
            eval_log=eval_log_path,
        )
        _write_artifact(results, output_path)
        _print_markdown_table(results)

        if results["llm_ndcg_uplift_pct"] < 0:
            logger.warning(
                "LLM rerank degraded NDCG by %.1f%% — investigate prompt / fallback rate",
                -results["llm_ndcg_uplift_pct"],
            )
        # No hard gate — LLM uplift is qualitative.
        return 0
    finally:
        if redis_started:
            try:
                await stop_redis()
            except Exception:
                logger.exception("redis shutdown failed during eval-llm")
        await dispose_engine()


__all__ = [
    "run_eval_llm",
    "run_eval_llm_cli",
]
