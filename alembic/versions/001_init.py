"""Initial migration: create document_chunks table with pgvector support.

Revision ID: 001
Revises:
Create Date: 2026-05-14

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Enable pgvector extension
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # Create document_chunks table
    op.create_table(
        "document_chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("chunk_id", sa.String(36), unique=True, nullable=False, index=True),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("embedding", sa.String),  # Will be altered to VECTOR after extension is ready
        sa.Column("source_doc", sa.String(255), nullable=False, index=True),
        sa.Column("doc_version", sa.String(20), default="1.0"),
        sa.Column("section_path", sa.String(500), default=""),
        sa.Column("doc_type", sa.String(50), nullable=False, index=True),
        sa.Column("doc_subtype", sa.String(50), nullable=False, index=True),
        sa.Column("last_updated", sa.DateTime, nullable=False),
        sa.Column("language", sa.String(10), default="en"),
        sa.Column("outcome_score", sa.Float, default=0.0),
        sa.Column("summary", sa.Text, nullable=True),
        sa.Column("question_variants", postgresql.ARRAY(sa.Text), default=[]),
    )

    # Alter embedding column to VECTOR(384)
    op.execute("ALTER TABLE document_chunks ALTER COLUMN embedding TYPE vector(384) USING embedding::vector(384)")

    # Create HNSW index for fast approximate nearest neighbor search
    op.execute("""
        CREATE INDEX idx_chunks_embedding_hnsw
        ON document_chunks
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """)

    # Metadata indexes for filtered retrieval
    op.create_index("idx_chunks_doc_type", "document_chunks", ["doc_type"])
    op.create_index("idx_chunks_doc_subtype", "document_chunks", ["doc_subtype"])
    op.create_index("idx_chunks_last_updated", "document_chunks", ["last_updated"])
    op.create_index("idx_chunks_language", "document_chunks", ["language"])
    op.create_index("idx_chunks_outcome_score", "document_chunks", ["outcome_score"])

    # GIN index for full-text search (simple tokenizer for mixed-language content)
    op.execute("""
        CREATE INDEX idx_chunks_fts
        ON document_chunks
        USING GIN (to_tsvector('simple', content || ' ' || COALESCE(summary, '')))
    """)


def downgrade() -> None:
    op.drop_table("document_chunks")
    op.execute("DROP EXTENSION IF EXISTS vector")
