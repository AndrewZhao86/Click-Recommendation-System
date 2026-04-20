from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, String, func
from sqlalchemy import Enum as SAEnum
from sqlalchemy.orm import Mapped, mapped_column

from click_rec.db.base import Base

USER_SEGMENTS: tuple[str, ...] = (
    "bargain_hunter",
    "brand_loyalist",
    "category_explorer",
    "new_arrivals",
    "premium",
)


class UserAccount(Base):
    __tablename__ = "user_account"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    segment: Mapped[str] = mapped_column(
        SAEnum(*USER_SEGMENTS, name="user_segment"),
        nullable=False,
        index=True,
    )
    country: Mapped[str] = mapped_column(String(2), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    last_active_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
