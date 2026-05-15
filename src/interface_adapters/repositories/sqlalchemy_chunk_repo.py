"""Concrete ChunkRepository implementation using SQLAlchemy.

This adapter translates between the domain Chunk entity and the ORM model.
No business logic lives here — only persistence concerns.
"""

from typing import List, Optional

from sqlalchemy.orm import Session

from domain.models import Chunk
from domain.repositories import ChunkRepository
from interface_adapters.repositories.orm_models import ChunkORM


class SQLAlchemyChunkRepository(ChunkRepository):
    """SQLAlchemy-backed chunk persistence."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def save_all(self, chunks: List[Chunk]) -> None:
        """Persist chunks via bulk upsert (insert or update on conflict)."""
        orm_objects = [self._to_orm(chunk) for chunk in chunks]
        self._session.bulk_save_objects(orm_objects)
        self._session.commit()

    def get_by_id(self, chunk_id: str) -> Optional[Chunk]:
        """Fetch by domain chunk_id string."""
        orm = (
            self._session.query(ChunkORM)
            .filter(ChunkORM.chunk_id == chunk_id)
            .first()
        )
        return self._to_domain(orm) if orm else None

    def list_by_source(self, source_doc: str) -> List[Chunk]:
        """Fetch all chunks for a given source document."""
        orms = (
            self._session.query(ChunkORM)
            .filter(ChunkORM.source_doc == source_doc)
            .all()
        )
        return [self._to_domain(orm) for orm in orms]

    # ------------------------------------------------------------------ #
    # Translation helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _to_orm(chunk: Chunk) -> ChunkORM:
        return ChunkORM(
            chunk_id=chunk.chunk_id,
            content=chunk.content,
            embedding=chunk.embedding,
            source_doc=chunk.source_doc,
            doc_version=chunk.doc_version,
            section_path=chunk.section_path,
            doc_type=chunk.doc_type,
            doc_subtype=chunk.doc_subtype,
            last_updated=chunk.last_updated,
            language=chunk.language,
            outcome_score=chunk.outcome_score,
            summary=chunk.summary,
            question_variants=chunk.question_variants,
        )

    @staticmethod
    def _to_domain(orm: ChunkORM) -> Chunk:
        return Chunk(
            chunk_id=orm.chunk_id,
            content=orm.content,
            embedding=list(orm.embedding) if orm.embedding else [],
            source_doc=orm.source_doc,
            doc_version=orm.doc_version,
            section_path=orm.section_path,
            doc_type=orm.doc_type,
            doc_subtype=orm.doc_subtype,
            last_updated=orm.last_updated,
            language=orm.language,
            outcome_score=orm.outcome_score,
            summary=orm.summary,
            question_variants=list(orm.question_variants) if orm.question_variants else [],
        )
