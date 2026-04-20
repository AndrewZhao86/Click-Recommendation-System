"""phase2 initial schema

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-04-19

Creates the Phase 2 domain tables (item, user_account, click_event, search_query,
co_click), the user_segment ENUM, and a GIN index on item.tsv for Phase 6 BM25.
The ivfflat index on item.embedding is intentionally deferred to 0002 so it can
be built on a populated table with meaningful centroids.
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0001_initial_schema"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


USER_SEGMENTS = (
    "bargain_hunter",
    "brand_loyalist",
    "category_explorer",
    "new_arrivals",
    "premium",
)


def upgrade() -> None:
    user_segment = postgresql.ENUM(*USER_SEGMENTS, name="user_segment", create_type=True)
    user_segment.create(op.get_bind(), checkfirst=True)

    op.create_table(
        "user_account",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column(
            "segment",
            postgresql.ENUM(*USER_SEGMENTS, name="user_segment", create_type=False),
            nullable=False,
        ),
        sa.Column("country", sa.String(length=2), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "last_active_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index("ix_user_account_segment", "user_account", ["segment"])

    op.create_table(
        "item",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("title", sa.String(length=512), nullable=False),
        sa.Column("description", sa.Text(), nullable=False, server_default=""),
        sa.Column("category", sa.String(length=128), nullable=False),
        sa.Column("brand", sa.String(length=128), nullable=False),
        sa.Column("price", sa.Numeric(10, 2), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "popularity_score",
            sa.Float(),
            nullable=False,
            server_default="0",
        ),
        sa.Column("embedding", Vector(384), nullable=True),
        sa.Column(
            "tsv",
            postgresql.TSVECTOR(),
            sa.Computed(
                "to_tsvector('english', title || ' ' || coalesce(description, ''))",
                persisted=True,
            ),
            nullable=True,
        ),
    )
    op.create_index("ix_item_category", "item", ["category"])
    op.create_index("ix_item_brand", "item", ["brand"])
    op.execute("CREATE INDEX item_tsv_gin ON item USING gin (tsv)")

    op.create_table(
        "click_event",
        sa.Column("event_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "user_id",
            sa.String(),
            sa.ForeignKey("user_account.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "item_id",
            sa.String(),
            sa.ForeignKey("item.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("query", sa.String(length=512), nullable=True),
        sa.Column("rank_position", sa.Integer(), nullable=False),
        sa.Column("dwell_ms", sa.Integer(), nullable=True),
        sa.Column("client_ts", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "server_ts",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index("ix_click_event_user_id", "click_event", ["user_id"])
    op.create_index("ix_click_event_item_id", "click_event", ["item_id"])

    op.create_table(
        "search_query",
        sa.Column("event_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "user_id",
            sa.String(),
            sa.ForeignKey("user_account.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("session_id", sa.String(length=64), nullable=False),
        sa.Column("query", sa.String(length=512), nullable=False),
        sa.Column("filters", postgresql.JSONB(), nullable=True),
        sa.Column(
            "ts",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index("ix_search_query_user_id", "search_query", ["user_id"])
    op.create_index("ix_search_query_session_id", "search_query", ["session_id"])

    op.create_table(
        "co_click",
        sa.Column(
            "item_a",
            sa.String(),
            sa.ForeignKey("item.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "item_b",
            sa.String(),
            sa.ForeignKey("item.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("count", sa.Integer(), nullable=False, server_default="0"),
    )


def downgrade() -> None:
    op.drop_table("co_click")
    op.drop_index("ix_search_query_session_id", table_name="search_query")
    op.drop_index("ix_search_query_user_id", table_name="search_query")
    op.drop_table("search_query")
    op.drop_index("ix_click_event_item_id", table_name="click_event")
    op.drop_index("ix_click_event_user_id", table_name="click_event")
    op.drop_table("click_event")
    op.execute("DROP INDEX IF EXISTS item_tsv_gin")
    op.drop_index("ix_item_brand", table_name="item")
    op.drop_index("ix_item_category", table_name="item")
    op.drop_table("item")
    op.drop_index("ix_user_account_segment", table_name="user_account")
    op.drop_table("user_account")
    postgresql.ENUM(name="user_segment").drop(op.get_bind(), checkfirst=True)
