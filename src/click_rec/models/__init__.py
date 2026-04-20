"""ORM + DTO re-exports. Importing this module registers every table on `Base.metadata`
— alembic autogenerate and the seeder both depend on that side-effect.
"""

from click_rec.db.base import Base
from click_rec.models.co_click import CoClick
from click_rec.models.events import ClickEvent, SearchQuery
from click_rec.models.item import EMBEDDING_DIM, Item
from click_rec.models.requests import (
    ClickEventRequest,
    EventBatchRequest,
    ImpressionEventRequest,
    SearchEventRequest,
)
from click_rec.models.schemas import (
    ClickEventDTO,
    ImpressionEventDTO,
    ItemDTO,
    SearchEventDTO,
    UserDTO,
)
from click_rec.models.user import USER_SEGMENTS, UserAccount

__all__ = [
    "Base",
    "EMBEDDING_DIM",
    "Item",
    "UserAccount",
    "USER_SEGMENTS",
    "ClickEvent",
    "SearchQuery",
    "CoClick",
    "ClickEventDTO",
    "ImpressionEventDTO",
    "SearchEventDTO",
    "ItemDTO",
    "UserDTO",
    "ClickEventRequest",
    "ImpressionEventRequest",
    "SearchEventRequest",
    "EventBatchRequest",
]
