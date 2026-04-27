"""Pure-function tests for the Phase 6 scorer.

The scorer is deterministic by design — same features in, same scores
out — and these tests pin that contract. No fakes, no I/O.
"""

from __future__ import annotations

from click_rec.ranker.config import RankerConfig
from click_rec.ranker.scorer import FEATURE_NAMES, _min_max, score


def _all_features(values: dict[str, list[float | None]]) -> dict[str, list[float | None]]:
    """Fill in any missing feature columns with all-None of the same length."""
    n = max(len(v) for v in values.values())
    out: dict[str, list[float | None]] = {}
    for name in FEATURE_NAMES:
        out[name] = list(values.get(name, [None] * n))
    return out


def test_min_max_handles_all_none() -> None:
    assert _min_max([None, None, None]) == [0.0, 0.0, 0.0]


def test_min_max_handles_constant_column() -> None:
    """A constant column carries no ranking signal → all 0."""
    assert _min_max([0.5, 0.5, 0.5]) == [0.0, 0.0, 0.0]


def test_min_max_normalises_in_range() -> None:
    out = _min_max([0.0, 0.5, 1.0])
    assert out == [0.0, 0.5, 1.0]


def test_min_max_treats_none_as_zero_after_normalisation() -> None:
    out = _min_max([1.0, None, 3.0])
    assert out[0] == 0.0  # min
    assert out[1] == 0.0  # None → 0
    assert out[2] == 1.0  # max


def test_min_max_single_real_value_maps_to_one() -> None:
    """One channel returning one hit must not lose its signal.

    With the previous "constant column → 0" rule, a sparse BM25 query
    that matched a single item would zero out that item's BM25 contribution
    even though it was the only lexical signal in the candidate pool.
    """
    out = _min_max([None, 0.5, None, None])
    assert out == [0.0, 1.0, 0.0, 0.0]


def test_min_max_constant_multi_value_still_zero() -> None:
    """Multiple equal real values → genuinely no ranking signal → all 0."""
    assert _min_max([0.5, 0.5, 0.5, None]) == [0.0, 0.0, 0.0, 0.0]


def test_score_deterministic_given_fixed_features() -> None:
    """The verify-bullet test from plan §8 / phase6plan.md."""
    cfg = RankerConfig()
    features = _all_features(
        {
            "bm25": [0.1, 0.5, 0.9],
            "vector": [0.3, 0.6, 0.2],
        }
    )
    scores_a, breakdowns_a = score(features, cfg)
    scores_b, breakdowns_b = score(features, cfg)
    assert scores_a == scores_b
    assert breakdowns_a == breakdowns_b


def test_breakdown_sums_to_score() -> None:
    cfg = RankerConfig()
    features = _all_features(
        {
            "bm25": [0.1, 0.5, 0.9],
            "vector": [0.3, 0.6, 0.2],
            "popularity": [10.0, 5.0, 0.0],
            "recency": [0.8, 0.2, 0.6],
        }
    )
    scores, breakdowns = score(features, cfg)
    for s, b in zip(scores, breakdowns, strict=True):
        assert abs(s - sum(b.values())) < 1e-9


def test_cold_start_user_gets_zero_personal_contribution() -> None:
    """A `personal` column of all 0 must contribute exactly 0 to scores."""
    cfg = RankerConfig()
    features = _all_features(
        {
            "bm25": [0.1, 0.5, 0.9],
            "personal": [0.0, 0.0, 0.0],  # cold start
        }
    )
    _scores, breakdowns = score(features, cfg)
    for b in breakdowns:
        assert b["personal"] == 0.0


def test_breakdown_contains_all_seven_features() -> None:
    cfg = RankerConfig()
    features = _all_features({"bm25": [0.0, 1.0]})
    _scores, breakdowns = score(features, cfg)
    for b in breakdowns:
        assert set(b.keys()) == set(FEATURE_NAMES)


def test_score_handles_empty_input() -> None:
    cfg = RankerConfig()
    features = {name: [] for name in FEATURE_NAMES}
    scores, breakdowns = score(features, cfg)
    assert scores == []
    assert breakdowns == []


def test_score_raises_on_length_mismatch() -> None:
    cfg = RankerConfig()
    features = _all_features({"bm25": [0.1, 0.2]})
    features["vector"] = [0.5]  # wrong length
    import pytest

    with pytest.raises(ValueError):
        score(features, cfg)


def test_higher_bm25_outranks_lower_when_only_bm25_signal() -> None:
    """Sanity: with only BM25 weight active, BM25 ordering wins."""
    cfg = RankerConfig(
        w_bm25=1.0,
        w_vector=0.0,
        w_popularity=0.0,
        w_recency=0.0,
        w_personal=0.0,
        w_co_click=0.0,
        w_price_fit=0.0,
    )
    features = _all_features({"bm25": [0.1, 0.9, 0.5]})
    scores, _ = score(features, cfg)
    # The candidate with bm25=0.9 is the max; its normalised value is 1.0
    assert scores[1] > scores[2] > scores[0]
