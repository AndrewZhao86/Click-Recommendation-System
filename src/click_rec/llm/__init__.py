"""Public surface for the Phase 7 Gemini LLM layer.

Importing from this package — rather than the underlying modules —
keeps the public contract stable when the internals get reshuffled
(e.g. a future `prompts/{use_case}.v2.txt` migration in Phase 8b).
"""

from click_rec.llm.client import (
    GeminiClient,
    LLMError,
    LLMParseError,
    LLMResult,
    LLMTimeoutError,
    LLMUnavailable,
    get_client,
    reset_for_tests,
)
from click_rec.llm.config import LLMConfig, judge_model, load_llm_config, runtime_model
from click_rec.llm.explain import explain_recommendation
from click_rec.llm.judge import judge_ranking
from click_rec.llm.query_understanding import understand_query
from click_rec.llm.re_ranker import build_user_profile_summary, rerank_top_k
from click_rec.llm.schemas import (
    JudgeVerdict,
    PriceBias,
    QueryIntent,
    RerankItem,
    RerankResult,
)

__all__ = [
    "GeminiClient",
    "JudgeVerdict",
    "LLMConfig",
    "LLMError",
    "LLMParseError",
    "LLMResult",
    "LLMTimeoutError",
    "LLMUnavailable",
    "PriceBias",
    "QueryIntent",
    "RerankItem",
    "RerankResult",
    "build_user_profile_summary",
    "explain_recommendation",
    "get_client",
    "judge_model",
    "judge_ranking",
    "load_llm_config",
    "rerank_top_k",
    "reset_for_tests",
    "runtime_model",
    "understand_query",
]
