"""Pydantic v2 DTOs mirroring the versioned event schemas in `plan.md` §5.

These are the wire contracts for ingestion / replay / API I/O. They are intentionally
separate from the SQLAlchemy ORM classes: DTOs describe the JSON shape, ORMs describe
the persisted shape (e.g., `server_ts` is required on the ORM row, optional on the
client-submitted DTO because the API edge fills it in).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class _EventBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: UUID
    event_version: int = 1
    timestamp: datetime
    user_id: str
    session_id: str
    server_ts: datetime | None = None


class ClickEventDTO(_EventBase):
    event_type: Literal["click"] = "click"
    query: str | None = None
    item_id: str
    rank_position: int
    dwell_ms: int | None = None


class ImpressionEventDTO(_EventBase):
    event_type: Literal["impression"] = "impression"
    query: str
    result_ids: list[str]
    page: int = 1


class SearchEventDTO(_EventBase):
    event_type: Literal["search"] = "search"
    query: str
    filters: dict[str, Any] = Field(default_factory=dict)


class ItemDTO(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str
    description: str
    category: str
    brand: str
    price: float
    created_at: datetime
    popularity_score: float = 0.0


class UserDTO(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    segment: str
    country: str
    created_at: datetime
    last_active_at: datetime
