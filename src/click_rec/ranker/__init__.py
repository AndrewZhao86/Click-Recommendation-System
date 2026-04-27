"""Public surface for the Phase 6 hybrid ranker.

`rank()` is the orchestration entrypoint; `RankedItemDTO` is the public
response schema; `load_ranker_config()` exposes the YAML loader for
callers that need to override weights at test time.
"""

from click_rec.ranker.config import RankerConfig, load_ranker_config
from click_rec.ranker.pipeline import rank
from click_rec.ranker.schemas import RankedItemDTO

__all__ = ["RankedItemDTO", "RankerConfig", "load_ranker_config", "rank"]
