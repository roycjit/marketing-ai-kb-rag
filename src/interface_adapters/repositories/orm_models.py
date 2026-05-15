"""SQLAlchemy ORM models — infrastructure concern, not domain.

These models map the domain Chunk entity to a PostgreSQL table with pgvector.
The domain model remains framework-agnostic; this file is the translation layer.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import List, Optional

from sqlalchemy import ARRAY, Column, DateTime, Float, String, Text
from sqlalchemy.dialects.postgresql import UUID

from pgvector.sqlalchemy import Vector

from frameworks.config import EMBEDDING_DIMENSION
from frameworks.database import Base


class ChunkORM(Base):
    """Database representation of a document chunk."""

    __tablename__ = "document_chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    chunk_id = Column(String(255), unique=True, nullable=False, index=True)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(EMBEDDING_DIMENSION))
    source_doc = Column(String(255), nullable=False, index=True)
    doc_version = Column(String(20), default="1.0")
    section_path = Column(String(500), default="")
    doc_type = Column(String(50), nullable=False, index=True)
    doc_subtype = Column(String(50), nullable=False, index=True)
    last_updated = Column(DateTime, nullable=False, default=datetime.utcnow)
    language = Column(String(10), default="en")
    outcome_score = Column(Float, default=0.0)
    summary = Column(Text, nullable=True)
    question_variants = Column(ARRAY(Text), default=[])

    def __repr__(self) -> str:
        return f"<ChunkORM {self.chunk_id} {self.source_doc}>"
