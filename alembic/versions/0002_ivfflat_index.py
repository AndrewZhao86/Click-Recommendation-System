"""ivfflat index on item.embedding

Revision ID: 0002_ivfflat_index
Revises: 0001_initial_schema
Create Date: 2026-04-19

Builds the pgvector ivfflat index on `item.embedding`. The seeder also creates
this index (idempotently) after bulk-loading items so centroids are learned from
real embeddings rather than an empty table. Declaring it here as well keeps
`alembic upgrade head` authoritative for anyone bootstrapping without the seeder.
"""
from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0002_ivfflat_index"
down_revision: str | Sequence[str] | None = "0001_initial_schema"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        "CREATE INDEX IF NOT EXISTS item_embedding_ivfflat "
        "ON item USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS item_embedding_ivfflat")
