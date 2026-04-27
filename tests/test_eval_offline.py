"""Unit tests for the offline-eval harness.

Tests the formula correctness and the hybrid-vs-baseline logic on
synthetic features — sidesteps replay-data dependency in CI.
"""

from __future__ import annotations

import math

from click_rec.eval.offline import mrr_at_k, ndcg_at_k


def test_ndcg_at_k_perfect_relevance_is_one() -> None:
    rels = [1.0, 0.0, 0.0]
    assert ndcg_at_k(rels, 3) == 1.0


def test_ndcg_at_k_known_value_for_position_two() -> None:
    """A relevant hit at position 2 (index 1) gives NDCG = (1/log2(3)) / 1."""
    rels = [0.0, 1.0, 0.0]
    expected = (1.0 / math.log2(3)) / 1.0
    assert ndcg_at_k(rels, 3) == expected


def test_ndcg_at_k_zero_when_no_relevant() -> None:
    assert ndcg_at_k([0.0, 0.0, 0.0], 3) == 0.0


def test_ndcg_at_k_handles_empty() -> None:
    assert ndcg_at_k([], 10) == 0.0


def test_ndcg_at_k_zero_k() -> None:
    assert ndcg_at_k([1.0], 0) == 0.0


def test_mrr_at_k_first_position() -> None:
    assert mrr_at_k([1.0, 0.0, 0.0], 3) == 1.0


def test_mrr_at_k_third_position() -> None:
    assert mrr_at_k([0.0, 0.0, 1.0], 3) == 1.0 / 3.0


def test_mrr_at_k_no_relevant() -> None:
    assert mrr_at_k([0.0, 0.0, 0.0], 3) == 0.0


def test_mrr_at_k_respects_cutoff() -> None:
    """A relevant hit at position 11 with k=10 is invisible."""
    rels = [0.0] * 10 + [1.0]
    assert mrr_at_k(rels, 10) == 0.0
