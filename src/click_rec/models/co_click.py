from __future__ import annotations

from sqlalchemy import ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from click_rec.db.base import Base


class CoClick(Base):
    __tablename__ = "co_click"

    item_a: Mapped[str] = mapped_column(
        String,
        ForeignKey("item.id", ondelete="CASCADE"),
        primary_key=True,
    )
    item_b: Mapped[str] = mapped_column(
        String,
        ForeignKey("item.id", ondelete="CASCADE"),
        primary_key=True,
    )
    count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
