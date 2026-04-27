"""Pure-function scorer.

Per-feature min-max → weighted sum → breakdown dict per candidate.

Three rules:

1. `None` in a feature column means "no signal for this candidate"
   (e.g. BM25 didn't match; embedding missing). It maps to 0.0
   *after* normalisation — never participates in the lo/hi calculation.
2. An all-`None` or constant column produces all 0.0 (no information →
   contributes nothing). This is what keeps the scorer deterministic
   even when a channel returns nothing.
3. The breakdown dict and the score are mathematically linked:
   `sum(breakdown[i].values()) == scores[i]` always (within float
   epsilon). Tested by `tests/test_ranker_scorer_unit.py`.
"""

from __future__ import annotations

from click_rec.ranker.config import RankerConfig

FEATURE_NAMES: tuple[str, ...] = (
    "bm25",
    "vector",
    "popularity",
    "recency",
    "personal",
    "co_click",
    "price_fit",
)

_WEIGHT_ATTR = {
    "bm25": "w_bm25",
    "vector": "w_vector",
    "popularity": "w_popularity",
    "recency": "w_recency",
    "personal": "w_personal",
    "co_click": "w_co_click",
    "price_fit": "w_price_fit",
}


def _min_max(xs: list[float | None]) -> list[float]:
    """Per-query min-max → [0, 1] with `None` → 0.

    Rules:
    - All-None column → all 0.0 (no signal at all).
    - Exactly one real value (rest None) → that hit gets 1.0, Nones get 0.0.
      A single-channel sparse match is itself the signal; collapsing it to
      0 silently erases the only candidate that channel surfaced.
    - Multiple real values that are all equal → all 0.0. A genuinely
      constant column carries no ranking information.
    """
    if not xs:
        return []
    real = [x for x in xs if x is not None]
    if not real:
        return [0.0 for _ in xs]
    if len(real) == 1:
        return [1.0 if x is not None else 0.0 for x in xs]
    lo = min(real)
    hi = max(real)
    if hi <= lo:
        return [0.0 for _ in xs]
    span = hi - lo
    return [0.0 if x is None else (x - lo) / span for x in xs]


def score(
    features: dict[str, list[float | None]], cfg: RankerConfig
) -> tuple[list[float], list[dict[str, float]]]:
    """Compute scores + breakdowns for a candidate batch.

    `features` must contain all 7 keys in `FEATURE_NAMES`. Each value is
    a list aligned with the candidate order. Returns
    `(scores, breakdowns)` aligned with the same order.
    """
    n = 0
    for name in FEATURE_NAMES:
        raw_col = features.get(name)
        if raw_col is None:
            raise KeyError(f"missing feature column: {name}")
        if n == 0:
            n = len(raw_col)
        elif len(raw_col) != n:
            raise ValueError(
                f"feature column length mismatch: {name} has {len(raw_col)} expected {n}"
            )

    if n == 0:
        return [], []

    normed: dict[str, list[float]] = {}
    for name in FEATURE_NAMES:
        normed[name] = _min_max(features[name])

    scores: list[float] = [0.0] * n
    breakdowns: list[dict[str, float]] = [{} for _ in range(n)]
    for name in FEATURE_NAMES:
        weight = float(getattr(cfg, _WEIGHT_ATTR[name]))
        normed_col = normed[name]
        for i in range(n):
            contrib = weight * normed_col[i]
            breakdowns[i][name] = contrib
            scores[i] += contrib
    return scores, breakdowns
