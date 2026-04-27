"""Pydantic v2 models for the structured outputs the Gemini SDK returns.

`response_schema=` on the SDK side rejects non-conforming output before
it reaches Python, so these double as both the generation contract and
the parse target. Keep them flat and bounded — `max_length` on lists
caps prompt-injection blast radius (a malicious query can't coerce a
1000-item rationale list).
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class PriceBias(str, Enum):
    low = "low"
    med = "med"
    high = "high"


class QueryIntent(BaseModel):
    """Structured rewrite of a free-text query.

    Phase 7 ships the parser; Phase 8a wires the result into candidate
    filters / re-rank prompts.
    """

    category: str | None = None
    attrs: list[str] = Field(default_factory=list, max_length=10)
    price_bias: PriceBias | None = None


class RerankItem(BaseModel):
    item_id: str
    # No upper bound on rank: `re_rank_top_k` is YAML-configurable, and
    # the surrounding `RerankResult.items` cap is the real injection-blast
    # bound. `ge=1` keeps 0 / negative ranks from sneaking in.
    rank: int = Field(ge=1)
    rationale: str = Field(max_length=200)


class RerankResult(BaseModel):
    # Cap is generous (>= LLMConfig.re_rank_input_k) so a configured
    # `re_rank_top_k` larger than the legacy 10 still validates. The cap
    # exists to bound prompt-injection blast radius, not to enforce K.
    items: list[RerankItem] = Field(min_length=1, max_length=50)


class JudgeVerdict(BaseModel):
    relevance_score: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(max_length=500)
