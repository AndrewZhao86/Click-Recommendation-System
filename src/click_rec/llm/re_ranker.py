"""LLM re-ranker — top-20 (hybrid) → top-10 with rationale.

This module is library-only in Phase 7: the eval harness
(`eval/llm.py`) and tests are the only callers. Phase 8a wires it into
`/search` behind the `use_llm` flag with the latency-budget
instrumentation around it.

Failure semantics: any LLM-side failure (timeout, parse, unavailable,
network) returns `None`. The caller is expected to fall back to the
unmodified hybrid order — the LLM is a feature, not the ranker.
"""

from __future__ import annotations

import logging

from click_rec.llm.client import (
    LLMError,
    LLMParseError,
    LLMTimeoutError,
    LLMUnavailable,
    get_client,
)
from click_rec.llm.config import LLMConfig, runtime_model
from click_rec.llm.prompts import RERANK_PROMPT
from click_rec.llm.schemas import RerankResult
from click_rec.ranker.schemas import RankedItemDTO
from click_rec.telemetry.metrics import llm_fallback_total

logger = logging.getLogger(__name__)

_USE_CASE = "re_rank"


def _format_candidate_block(candidates: list[RankedItemDTO]) -> str:
    """One line per candidate. Top-3 score features keep the prompt compact."""
    lines: list[str] = []
    for c in candidates:
        breakdown = c.score_breakdown or {}
        top_feats = sorted(breakdown.items(), key=lambda kv: kv[1], reverse=True)[:3]
        feat_str = ", ".join(f"{k}={v:.2f}" for k, v in top_feats) or "—"
        lines.append(
            f"{c.item.id} | {c.item.title} | {c.item.category} | "
            f"{c.item.brand} | ${c.item.price:.2f} | {feat_str}"
        )
    return "\n".join(lines)


def _reorder(candidates: list[RankedItemDTO], rerank: RerankResult) -> list[RankedItemDTO]:
    """Reorder candidates by `rerank.items.rank` and attach rationales.

    Items the LLM omitted (e.g. it returned only 8 of 10) are tail-padded
    in the original hybrid order. Items the LLM mentioned but that don't
    appear in the candidate pool are dropped — we never trust the model
    to materialise an item id.
    """
    by_id: dict[str, RankedItemDTO] = {c.item.id: c for c in candidates}

    sorted_picks = sorted(rerank.items, key=lambda it: it.rank)
    out: list[RankedItemDTO] = []
    seen: set[str] = set()
    for pick in sorted_picks:
        cand = by_id.get(pick.item_id)
        if cand is None or cand.item.id in seen:
            continue
        out.append(cand.model_copy(update={"llm_rationale": pick.rationale}))
        seen.add(cand.item.id)

    for cand in candidates:
        if cand.item.id in seen:
            continue
        out.append(cand)
        seen.add(cand.item.id)

    return out


async def rerank_top_k(
    *,
    query: str,
    candidates: list[RankedItemDTO],
    user_profile_summary: str | None,
    cfg: LLMConfig,
) -> list[RankedItemDTO] | None:
    """Re-rank up to `cfg.re_rank_input_k` hybrid candidates via Gemini.

    Returns the reordered + rationalised list (length up to `top_k`,
    plus tail-padded leftovers in original order) or `None` on any
    LLM-side failure. The caller falls back to the unmodified list.
    """
    if not candidates:
        return None

    head = candidates[: cfg.re_rank_input_k]
    candidate_block = _format_candidate_block(head)
    prompt = RERANK_PROMPT.format(
        query=query,
        top_k=cfg.re_rank_top_k,
        user_profile=user_profile_summary or "(no recent activity)",
        candidate_block=candidate_block,
    )

    try:
        client = await get_client()
    except LLMUnavailable:
        llm_fallback_total.labels(use_case=_USE_CASE, reason="unavailable").inc()
        return None

    try:
        result = await client.generate_json(
            model=runtime_model(),
            prompt=prompt,
            schema=RerankResult,
            timeout=cfg.re_rank_timeout,
            use_case=_USE_CASE,
        )
    except LLMTimeoutError:
        llm_fallback_total.labels(use_case=_USE_CASE, reason="timeout").inc()
        return None
    except LLMParseError:
        llm_fallback_total.labels(use_case=_USE_CASE, reason="parse_error").inc()
        return None
    except LLMError:
        llm_fallback_total.labels(use_case=_USE_CASE, reason="error").inc()
        return None

    rerank: RerankResult = result.parsed
    return _reorder(head, rerank)


def build_user_profile_summary(
    *,
    recent_categories: list[str],
    recent_brands: list[str],
    avg_price: float | None,
) -> str:
    """Compact profile summary suitable for inclusion in a prompt.

    Phase 7 callers (eval harness) construct this from `UserContext`;
    Phase 8a will source it from a Redis hash.
    """
    parts: list[str] = []
    if recent_categories:
        cats = ", ".join(recent_categories[:5])
        parts.append(f"recent categories: {cats}")
    if recent_brands:
        brands = ", ".join(recent_brands[:5])
        parts.append(f"recent brands: {brands}")
    if avg_price is not None:
        parts.append(f"avg price ~${avg_price:.0f}")
    return "; ".join(parts) if parts else "(no recent activity)"
