"""Request-side DTOs for the ingestion API.

These mirror [schemas.py](schemas.py) but relax two fields: `event_id` is an
optional client-supplied idempotency key (server generates UUIDv7 if missing)
and `server_ts` is not accepted from clients (it's always stamped at the API
edge). `extra="forbid"` mirrors the wire DTOs so unknown fields 422 early.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Final, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

MAX_BATCH_EVENTS: Final[int] = 500


class _EventRequestBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: UUID | None = None
    event_version: int = 1
    timestamp: datetime
    user_id: str
    session_id: str


class ClickEventRequest(_EventRequestBase):
    event_type: Literal["click"] = "click"
    query: str | None = None
    item_id: str
    rank_position: int
    dwell_ms: int | None = None


class ImpressionEventRequest(_EventRequestBase):
    event_type: Literal["impression"] = "impression"
    query: str
    result_ids: list[str]
    page: int = 1


class SearchEventRequest(_EventRequestBase):
    event_type: Literal["search"] = "search"
    query: str
    filters: dict[str, Any] = Field(default_factory=dict)


BatchEvent = Annotated[
    ClickEventRequest | ImpressionEventRequest | SearchEventRequest,
    Field(discriminator="event_type"),
]


class EventBatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    events: list[BatchEvent] = Field(
        ..., min_length=1, max_length=MAX_BATCH_EVENTS
    )
