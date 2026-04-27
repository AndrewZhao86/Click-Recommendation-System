"""Phase 6 offline eval — `make eval-offline`.

The harness lives in `click_rec.eval.offline`. Public entry points:
- `run_offline_eval(...)` — pure async function, returns a metrics dict.
- `run_offline_eval_cli(args)` — owns Redis + DB lifecycle so the CLI
  works without the FastAPI lifespan.
"""

from click_rec.eval.offline import (
    mrr_at_k,
    ndcg_at_k,
    run_offline_eval,
    run_offline_eval_cli,
)

__all__ = [
    "mrr_at_k",
    "ndcg_at_k",
    "run_offline_eval",
    "run_offline_eval_cli",
]
