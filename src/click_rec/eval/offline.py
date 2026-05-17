"""NDCG@K / MRR@K offline eval — gates the ≥15% NDCG-uplift bar.

Two passes:

1. **Replay gold (PRIMARY, gates the 15% bar).** Reads triples from
   `artifacts/eval_clicks.jsonl` (captured via the `--capture-eval-log`
   flag on `make replay`). For each `(user_id, query, clicked_item_id)`,
   ranks candidates and scores by clicked-item position. This is the
   click-through gold the headline metric is computed against.
2. **Hand gold (SPOT CHECK).** Iterates `tests/data/golden_queries.json`
   anonymously (`user_id=None`); a result is "relevant" iff its
   `category` is in the expected list **or** its `brand` is in the
   expected list. Qualitative-only — do not interpret as a
   production-quality number.

Both passes also run a BM25-only baseline (other weights zeroed) so the
uplift is computed against a known-trivial reference.

The CLI entrypoint owns Redis + DB lifecycle because the `rank()` hot
path calls `ZSCORE` on the Redis singleton — without `start_redis()`
the very first iteration would raise from `redis_client.get_redis()`.
Mirrors the singleton lifecycle that `scripts/replay_clicks.py` manages.

Shared helpers (replay scorer, holdout dance, BM25 weight zeroing,
metric formulas) live in `_replay.py` and are reused by `eval/llm.py`.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import time
from pathlib import Path
from typing import Any

from click_rec.cache.redis_client import get_redis, start_redis, stop_redis
from click_rec.config import get_settings
from click_rec.db.base import dispose_engine, get_sessionmaker
from click_rec.eval._replay import (
    bm25_only_cfg,
    eval_replay_pass,
    load_replay_triples,
    mrr_at_k,
    ndcg_at_k,
)
from click_rec.ranker import RankerConfig, load_ranker_config, rank

logger = logging.getLogger(__name__)


# ============================================================ hand-gold pass


async def _eval_hand_pass(
    *,
    session_factory: Any,
    redis_client: Any,
    cfg: RankerConfig,
    golden: list[dict[str, Any]],
    k: int,
) -> dict[str, float]:
    """Score the 20-query hand gold by category-or-brand match."""
    if not golden:
        return {"ndcg": 0.0, "mrr": 0.0, "n": 0.0}

    ndcgs: list[float] = []
    mrrs: list[float] = []
    for entry in golden:
        query = entry["query"]
        cats = set(entry.get("expected_categories") or [])
        brands = set(entry.get("expected_brands") or [])
        async with session_factory() as session:
            results = await rank(
                query=query,
                user_id=None,
                session=session,
                redis_client=redis_client,
                limit=k,
                cfg=cfg,
            )
        rels = [
            1.0 if (r.item.category in cats or r.item.brand in brands) else 0.0 for r in results
        ]
        ndcgs.append(ndcg_at_k(rels, k))
        mrrs.append(mrr_at_k(rels, k))

    return {
        "ndcg": sum(ndcgs) / len(ndcgs),
        "mrr": sum(mrrs) / len(mrrs),
        "n": float(len(golden)),
    }


# ============================================================ data loading


def _load_golden(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        logger.warning("golden_queries.json not found at %s — skipping pass 2", path)
        return []
    with path.open("r", encoding="utf-8") as fh:
        return list(json.load(fh))


# ============================================================ eval orchestration


async def run_offline_eval(
    *,
    session_factory: Any,
    redis_client: Any,
    num_users: int = 200,
    k: int = 10,
    eval_log: Path | None = None,
    golden_path: Path | None = None,
) -> dict[str, float]:
    """Run BM25-baseline + hybrid eval, return a metrics dict.

    Hybrid metrics use the YAML-loaded `RankerConfig`. Baseline metrics
    zero every weight except `w_bm25`. The headline `ndcg_uplift_pct` is
    computed against the replay pass when available, else the hand pass
    (with a `WARN: hand-gold uplift is qualitative only` log).
    """
    cfg_hybrid = load_ranker_config()
    cfg_bm25 = bm25_only_cfg(cfg_hybrid)

    settings = get_settings()
    artifacts_dir = Path(settings.eval_output_dir)
    eval_log = eval_log or (artifacts_dir / "eval_clicks.jsonl")
    if golden_path is None:
        golden_path = Path(__file__).resolve().parents[3] / "tests" / "data" / "golden_queries.json"

    triples = load_replay_triples(eval_log, num_users)
    golden = _load_golden(golden_path)

    replay_hybrid = await eval_replay_pass(
        session_factory=session_factory,
        redis_client=redis_client,
        cfg=cfg_hybrid,
        triples=triples,
        k=k,
    )
    replay_bm25 = await eval_replay_pass(
        session_factory=session_factory,
        redis_client=redis_client,
        cfg=cfg_bm25,
        triples=triples,
        k=k,
    )
    hand_hybrid = await _eval_hand_pass(
        session_factory=session_factory,
        redis_client=redis_client,
        cfg=cfg_hybrid,
        golden=golden,
        k=k,
    )
    hand_bm25 = await _eval_hand_pass(
        session_factory=session_factory,
        redis_client=redis_client,
        cfg=cfg_bm25,
        golden=golden,
        k=k,
    )

    # Headline uplift: replay if we have triples, else hand.
    if replay_hybrid["n"] > 0 and replay_bm25["ndcg"] > 0:
        uplift = (replay_hybrid["ndcg"] - replay_bm25["ndcg"]) / replay_bm25["ndcg"] * 100.0
        uplift_basis = "replay"
    elif hand_hybrid["n"] > 0 and hand_bm25["ndcg"] > 0:
        logger.warning(
            "no replay triples — falling back to hand-gold for uplift "
            "(QUALITATIVE only; do not interpret as a production number)"
        )
        uplift = (hand_hybrid["ndcg"] - hand_bm25["ndcg"]) / hand_bm25["ndcg"] * 100.0
        uplift_basis = "hand"
    else:
        uplift = 0.0
        uplift_basis = "none"

    return {
        "hybrid_ndcg@10": replay_hybrid["ndcg"],
        "hybrid_mrr@10": replay_hybrid["mrr"],
        "bm25_ndcg@10": replay_bm25["ndcg"],
        "bm25_mrr@10": replay_bm25["mrr"],
        "replay_n": replay_hybrid["n"],
        "hand_hybrid_ndcg@10": hand_hybrid["ndcg"],
        "hand_hybrid_mrr@10": hand_hybrid["mrr"],
        "hand_bm25_ndcg@10": hand_bm25["ndcg"],
        "hand_bm25_mrr@10": hand_bm25["mrr"],
        "hand_n": hand_hybrid["n"],
        "ndcg_uplift_pct": uplift,
        "uplift_basis": _basis_to_float(uplift_basis),
    }


def _basis_to_float(basis: str) -> float:
    """Encode the basis as a float so the metrics dict stays homogeneous."""
    return {"replay": 1.0, "hand": 2.0, "none": 0.0}.get(basis, 0.0)


# ============================================================ CLI


def _print_markdown_table(results: dict[str, float]) -> None:
    print()
    print("| Metric        | BM25-only | Hybrid | Uplift |")
    print("| ------------- | --------- | ------ | ------ |")
    print(
        f"| NDCG@10       | {results['bm25_ndcg@10']:.4f}   | "
        f"{results['hybrid_ndcg@10']:.4f} | {results['ndcg_uplift_pct']:+.1f}% |"
    )
    print(
        f"| MRR@10        | {results['bm25_mrr@10']:.4f}   | "
        f"{results['hybrid_mrr@10']:.4f} |        |"
    )
    print()
    print(
        f"| Hand-gold     | {results['hand_bm25_ndcg@10']:.4f}   | "
        f"{results['hand_hybrid_ndcg@10']:.4f} |        |"
    )
    print()
    print(f"replay_n={int(results['replay_n'])}  hand_n={int(results['hand_n'])}")


def _write_artifact(results: dict[str, float], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **results,
        "timestamp": int(time.time()),
        "config": dataclasses.asdict(load_ranker_config()),
    }
    output.write_text(json.dumps(payload, indent=2))
    logger.info("wrote eval artifact: %s", output)


async def run_offline_eval_cli(args: argparse.Namespace) -> int:
    """CLI driver — owns the Redis + DB lifecycle for a one-shot run."""
    settings = get_settings()
    artifacts_dir = Path(getattr(args, "output", None) or settings.eval_output_dir)
    if artifacts_dir.suffix == ".json":
        output_path = artifacts_dir
    else:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        output_path = artifacts_dir / f"eval_offline_{int(time.time())}.json"

    eval_log_path = Path(args.eval_log) if getattr(args, "eval_log", None) else None
    golden_path = Path(args.golden) if getattr(args, "golden", None) else None

    # Best-effort Redis startup — eval still runs without Redis (cold-start
    # path), it just exercises fewer features. A failure here logs and
    # continues; rank() will see `redis_client=None` and degrade.
    redis_client: Any = None
    redis_started = False
    try:
        await start_redis()
        redis_client = get_redis()
        redis_started = True
    except Exception:
        logger.warning(
            "redis unavailable for eval — running without popularity / personal Redis features"
        )

    try:
        sm = get_sessionmaker()
        results = await run_offline_eval(
            session_factory=sm,
            redis_client=redis_client,
            num_users=args.num_users,
            k=args.k,
            eval_log=eval_log_path,
            golden_path=golden_path,
        )
        _write_artifact(results, output_path)
        _print_markdown_table(results)

        if results["ndcg_uplift_pct"] < 15.0:
            logger.error(
                "FAIL: NDCG uplift %.1f%% below the 15%% bar",
                results["ndcg_uplift_pct"],
            )
            return 1
        return 0
    finally:
        if redis_started:
            try:
                await stop_redis()
            except Exception:
                logger.exception("redis shutdown failed during eval")
        await dispose_engine()


# Keep the public surface stable for tests that import these directly.
__all__ = [
    "mrr_at_k",
    "ndcg_at_k",
    "run_offline_eval",
    "run_offline_eval_cli",
]
