"""Ranker DTOs — the public response shape and an internal candidate row.

`RankedItemDTO` *composes* `ItemDTO` rather than subclassing so the
`GET /items/{id}` contract stays untouched. `RankingCandidate` is the
internal struct passed between candidates / features / scorer; it carries
the embedding (which `ItemDTO` deliberately does not expose) and the
per-channel raw scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from click_rec.models.schemas import ItemDTO


@dataclass(slots=True)
class RankingCandidate:
    """Internal row passed through the ranker pipeline.

    `bm25_raw` / `vector_raw` are `None` (not 0) when a channel did not
    surface this candidate — the scorer treats `None` as "no signal" and
    a real 0 as a real (low) score.
    """

    item_id: str
    title: str
    description: str
    category: str
    brand: str
    price: float
    created_at: datetime
    popularity_score: float
    embedding: list[float] | None
    bm25_raw: float | None
    vector_raw: float | None
    # Cosine similarity between user profile_vec and this candidate's
    # embedding, computed in SQL via pgvector at fetch time when
    # profile_vec is known. None when profile_vec was unavailable
    # (cold-start) or when the legacy in-Python path is used.
    personal_raw: float | None = None


class RankedItemDTO(BaseModel):
    """Public response shape for `GET /search`.

    Composes `ItemDTO` so the existing item DTO stays clean and the
    ranker can ship `score` + `score_breakdown` without re-shaping the
    catalog model. The breakdown enables the plan §8 step-4 debuggability
    bullet — every score reads back as a sum of weighted feature
    contributions, sanity-checkable in `jq`.
    """

    model_config = ConfigDict(from_attributes=True)

    item: ItemDTO
    score: float
    score_breakdown: dict[str, float]
    # Phase 7: populated by the LLM re-ranker. None when the LLM path is
    # off or fell back to hybrid order — keeps the Phase 6 response shape
    # forward-compatible.
    llm_rationale: str | None = None
