"""Smoke tests for `eval/llm.py`.

We don't run the full DB-backed eval here — that's the integration job
of `make eval-llm` against seeded Postgres. These assertions live at
the orchestration layer: shape of the metrics dict, fallback counting
when the rerank stub returns None.
"""

from __future__ import annotations

from typing import Any

import pytest

from click_rec.eval import llm as eval_llm_mod

_EXPECTED_KEYS = {
    "hybrid_ndcg@10",
    "hybrid_mrr@10",
    "bm25_ndcg@10",
    "bm25_mrr@10",
    "llm_ndcg@10",
    "llm_mrr@10",
    "llm_ndcg_uplift_pct",
    "llm_fallback_pct",
    "judge_hybrid_score_mean",
    "judge_llm_score_mean",
    "judge_uplift_pct",
    "judge_sample_size",
    "replay_n",
    "judge_skipped",
}


async def test_eval_llm_metric_keys_present(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    """No replay log → all keys still present, all zero."""

    async def fake_replay_pass(**_kwargs: Any) -> dict[str, float]:
        return {"ndcg": 0.0, "mrr": 0.0, "n": 0.0}

    async def fake_llm_pass(**_kwargs: Any) -> dict[str, float]:
        return {"ndcg": 0.0, "mrr": 0.0, "n": 0.0, "fallbacks": 0.0}

    async def fake_judge_pass(**_kwargs: Any) -> dict[str, float]:
        return {"judge_hybrid": 0.0, "judge_llm": 0.0, "judge_n": 0.0}

    from click_rec.llm.client import LLMUnavailable

    async def fake_get_client() -> Any:
        raise LLMUnavailable("test")

    monkeypatch.setattr(eval_llm_mod, "eval_replay_pass", fake_replay_pass)
    monkeypatch.setattr(eval_llm_mod, "_eval_llm_pass", fake_llm_pass)
    monkeypatch.setattr(eval_llm_mod, "_eval_judge_pass", fake_judge_pass)
    monkeypatch.setattr(eval_llm_mod, "get_client", fake_get_client)

    # No replay log file → triples=[]
    results = await eval_llm_mod.run_eval_llm(
        session_factory=None,
        redis_client=None,
        num_users=10,
        k=10,
        judge_sample=0,
        eval_log=tmp_path / "no-such.jsonl",
    )
    assert set(results.keys()) == _EXPECTED_KEYS
    # LLM unavailable → judge_skipped flag set.
    assert results["judge_skipped"] == 1.0


async def test_eval_llm_handles_rerank_fallbacks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """If half the queries fall back, llm_fallback_pct reflects it."""

    async def fake_replay_pass(**_kwargs: Any) -> dict[str, float]:
        return {"ndcg": 0.5, "mrr": 0.4, "n": 4.0}

    async def fake_llm_pass(**_kwargs: Any) -> dict[str, float]:
        return {"ndcg": 0.55, "mrr": 0.45, "n": 4.0, "fallbacks": 2.0}

    async def fake_judge_pass(**_kwargs: Any) -> dict[str, float]:
        return {"judge_hybrid": 0.0, "judge_llm": 0.0, "judge_n": 0.0}

    from click_rec.llm.client import LLMUnavailable

    async def fake_get_client() -> Any:
        raise LLMUnavailable("test")

    monkeypatch.setattr(eval_llm_mod, "eval_replay_pass", fake_replay_pass)
    monkeypatch.setattr(eval_llm_mod, "_eval_llm_pass", fake_llm_pass)
    monkeypatch.setattr(eval_llm_mod, "_eval_judge_pass", fake_judge_pass)
    monkeypatch.setattr(eval_llm_mod, "get_client", fake_get_client)

    results = await eval_llm_mod.run_eval_llm(
        session_factory=None,
        redis_client=None,
        num_users=10,
        k=10,
        judge_sample=0,
        eval_log=tmp_path / "no-such.jsonl",
    )
    assert results["llm_fallback_pct"] == 50.0
    assert results["llm_ndcg_uplift_pct"] == pytest.approx(10.0, rel=1e-2)
