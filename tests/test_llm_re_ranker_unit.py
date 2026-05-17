"""Unit tests for `rerank_top_k` reorder + fallback semantics."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

import pytest

from click_rec.llm import client as client_mod
from click_rec.llm.config import LLMConfig
from click_rec.llm.re_ranker import rerank_top_k
from click_rec.models.schemas import ItemDTO
from click_rec.ranker.schemas import RankedItemDTO


def _ranked(item_id: str, score: float = 0.5) -> RankedItemDTO:
    return RankedItemDTO(
        item=ItemDTO(
            id=item_id,
            title=f"Item {item_id}",
            description="",
            category="electronics/headphones",
            brand="Acme",
            price=99.99,
            created_at=datetime(2026, 4, 1, tzinfo=UTC),
            popularity_score=0.0,
        ),
        score=score,
        score_breakdown={
            "bm25": 0.3,
            "vector": 0.2,
            "popularity": 0.0,
            "recency": 0.0,
            "personal": 0.0,
            "co_click": 0.0,
            "price_fit": 0.0,
        },
    )


# ---------------------------------------------------------------- fakes


class _FakeUsage:
    prompt_token_count = 1
    candidates_token_count = 1


class _FakeResponse:
    def __init__(self, *, text: str | None = None) -> None:
        self.text = text
        self.parsed = None
        self.usage_metadata = _FakeUsage()
        self.candidates: list[Any] = []


class _FakeModels:
    def __init__(self, response: Any, *, sleep_s: float = 0.0) -> None:
        self.response = response
        self.sleep_s = sleep_s

    async def generate_content(self, **_kwargs: Any) -> Any:
        if self.sleep_s > 0:
            await asyncio.sleep(self.sleep_s)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class _FakeSdk:
    def __init__(self, models: _FakeModels) -> None:
        class _Aio:
            pass

        aio = _Aio()
        aio.models = models  # type: ignore[attr-defined]
        self.aio = aio


@pytest.fixture(autouse=True)
def _reset() -> None:
    client_mod.reset_for_tests()
    yield
    client_mod.reset_for_tests()


def _inject(response: Any, *, sleep_s: float = 0.0) -> None:
    sdk = _FakeSdk(_FakeModels(response, sleep_s=sleep_s))
    client_mod.set_client_for_tests(client_mod.GeminiClient(sdk=sdk))


# ---------------------------------------------------------------- tests


async def test_rerank_reorders_per_canned_response() -> None:
    """The LLM picks B then A — output must respect that order."""
    cands = [_ranked("a", 0.9), _ranked("b", 0.5), _ranked("c", 0.3)]
    canned = (
        '{"items": ['
        '{"item_id": "b", "rank": 1, "rationale": "best fit"},'
        '{"item_id": "a", "rank": 2, "rationale": "ok"}'
        "]}"
    )
    _inject(_FakeResponse(text=canned))

    out = await rerank_top_k(
        query="anything",
        candidates=cands,
        user_profile_summary=None,
        cfg=LLMConfig(re_rank_top_k=2, re_rank_input_k=20),
    )
    assert out is not None
    assert [r.item.id for r in out[:2]] == ["b", "a"]
    assert out[0].llm_rationale == "best fit"


async def test_rerank_pads_with_hybrid_leftovers() -> None:
    """Items the LLM omitted appear at the tail in original order."""
    cands = [_ranked("a", 0.9), _ranked("b", 0.5), _ranked("c", 0.3)]
    canned = '{"items": [{"item_id": "c", "rank": 1, "rationale": "zoom"}]}'
    _inject(_FakeResponse(text=canned))

    out = await rerank_top_k(
        query="x",
        candidates=cands,
        user_profile_summary=None,
        cfg=LLMConfig(re_rank_input_k=20),
    )
    assert out is not None
    assert [r.item.id for r in out] == ["c", "a", "b"]
    # Padded items have no rationale set.
    assert out[1].llm_rationale is None
    assert out[2].llm_rationale is None


async def test_rerank_timeout_returns_none() -> None:
    cands = [_ranked("a")]
    canned = '{"items": [{"item_id": "a", "rank": 1, "rationale": "x"}]}'
    _inject(_FakeResponse(text=canned), sleep_s=0.5)

    out = await rerank_top_k(
        query="x",
        candidates=cands,
        user_profile_summary=None,
        cfg=LLMConfig(re_rank_timeout=0.05),
    )
    assert out is None


async def test_rerank_parse_error_returns_none() -> None:
    cands = [_ranked("a")]
    _inject(_FakeResponse(text="not-json"))

    out = await rerank_top_k(
        query="x", candidates=cands, user_profile_summary=None, cfg=LLMConfig()
    )
    assert out is None


async def test_rationale_attached_to_dto() -> None:
    cands = [_ranked("a")]
    _inject(
        _FakeResponse(
            text='{"items": [{"item_id": "a", "rank": 1, "rationale": "matches your history"}]}'
        )
    )

    out = await rerank_top_k(
        query="x", candidates=cands, user_profile_summary="cats: shoes", cfg=LLMConfig()
    )
    assert out is not None
    assert out[0].llm_rationale == "matches your history"


async def test_empty_candidates_returns_none() -> None:
    out = await rerank_top_k(query="x", candidates=[], user_profile_summary=None, cfg=LLMConfig())
    assert out is None
