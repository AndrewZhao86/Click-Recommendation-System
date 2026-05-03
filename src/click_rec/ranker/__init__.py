"""Public surface for the Phase 6 hybrid ranker.

`rank()` is the orchestration entrypoint; `RankedItemDTO` is the public
response schema; `load_ranker_config()` exposes the YAML loader for
callers that need to override weights at test time.

Phase 8a: `UserContext` and `load_user_context` are re-exported so
route handlers can build the LLM re-rank profile summary inline from
the same context that the pipeline already loaded.
"""

from click_rec.ranker.config import RankerConfig, load_ranker_config
from click_rec.ranker.features import UserContext, load_user_context
from click_rec.ranker.pipeline import rank
from click_rec.ranker.schemas import RankedItemDTO

__all__ = [
    "RankedItemDTO",
    "RankerConfig",
    "UserContext",
    "load_ranker_config",
    "load_user_context",
    "rank",
]
