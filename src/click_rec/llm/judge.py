"""LLM-as-judge — `gemini-2.5-pro` rates a top-10 ranking 0..1.

Used by `eval/llm.py` only. Same-family judging (Pro grading Flash)
mitigates self-preference bias only partially; documented as a known
limitation in `phase7plan.md`. Replay-derived NDCG is the primary
quantitative signal — the judge score is qualitative.
"""

from __future__ import annotations

import logging

from click_rec.llm.client import (
    LLMError,
    LLMTimeoutError,
    LLMUnavailable,
    get_client,
)
from click_rec.llm.config import LLMConfig, judge_model
from click_rec.llm.prompts import JUDGE_PROMPT
from click_rec.llm.schemas import JudgeVerdict
from click_rec.telemetry.metrics import llm_fallback_total

logger = logging.getLogger(__name__)

_USE_CASE = "judge"


def _format_ranked_block(items: list[dict[str, str]]) -> str:
    return "\n".join(
        f"{i + 1} | {it.get('id', '?')} | {it.get('title', '?')} | "
        f"{it.get('category', '?')} | {it.get('brand', '?')}"
        for i, it in enumerate(items)
    )


async def judge_ranking(
    *,
    query: str,
    ranked_top_10: list[dict[str, str]],
    cfg: LLMConfig,
) -> JudgeVerdict | None:
    """Score a top-10 ranking against the query. None on any failure."""
    if not ranked_top_10:
        return None

    prompt = JUDGE_PROMPT.format(
        query=query, ranked_block=_format_ranked_block(ranked_top_10[:10])
    )

    try:
        client = await get_client()
    except LLMUnavailable:
        llm_fallback_total.labels(use_case=_USE_CASE, reason="unavailable").inc()
        return None

    try:
        result = await client.generate_json(
            model=judge_model(),
            prompt=prompt,
            schema=JudgeVerdict,
            timeout=cfg.judge_timeout,
            use_case=_USE_CASE,
        )
    except LLMTimeoutError:
        llm_fallback_total.labels(use_case=_USE_CASE, reason="timeout").inc()
        return None
    except LLMError:
        llm_fallback_total.labels(use_case=_USE_CASE, reason="error").inc()
        return None

    verdict: JudgeVerdict = result.parsed
    return verdict
