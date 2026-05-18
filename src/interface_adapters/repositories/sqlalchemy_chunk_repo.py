"""Concrete ChunkRepository implementation using SQLAlchemy.

This adapter translates between the domain Chunk entity and the ORM model.
No business logic lives here — only persistence concerns.
"""

from __future__ import annotations

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from domain.exceptions import RepositoryError
from domain.models import Chunk
from domain.repositories import ChunkRepository
from interface_adapters.repositories.orm_models import ChunkORM


class SQLAlchemyChunkRepository(ChunkRepository):
    """SQLAlchemy-backed chunk persistence with upsert support."""

    def __init__(self, session: Session) -> None:
        """Initialize with a SQLAlchemy session."""
        self._session = session

    def save_all(self, chunks: list[Chunk]) -> None:
        """Persist chunks via bulk upsert (insert or update on conflict).

        Uses PostgreSQL ON CONFLICT for idempotent ingestion.
        Transaction rollback on failure to keep session clean.
        """
        if not chunks:
            return

        orm_objects = [self._to_orm(chunk) for chunk in chunks]

        try:
            # Use bulk_insert_mappings with upsert for idempotent saves
            mappings = [
                {
                    "chunk_id": obj.chunk_id,
                    "content": obj.content,
                    "embedding": obj.embedding,
                    "source_doc": obj.source_doc,
                    "doc_version": obj.doc_version,
                    "section_path": obj.section_path,
                    "doc_type": obj.doc_type,
                    "doc_subtype": obj.doc_subtype,
                    "last_updated": obj.last_updated,
                    "language": obj.language,
                    "outcome_score": obj.outcome_score,
                    "summary": obj.summary,
                    "question_variants": obj.question_variants,
                }
                for obj in orm_objects
            ]

            upsert_stmt = pg_insert(ChunkORM).values(mappings)
            upsert_stmt = upsert_stmt.on_conflict_do_update(
                index_elements=["chunk_id"],
                set_={
                    "content": upsert_stmt.excluded.content,
                    "embedding": upsert_stmt.excluded.embedding,
                    "doc_version": upsert_stmt.excluded.doc_version,
                    "section_path": upsert_stmt.excluded.section_path,
                    "doc_type": upsert_stmt.excluded.doc_type,
                    "doc_subtype": upsert_stmt.excluded.doc_subtype,
                    "last_updated": upsert_stmt.excluded.last_updated,
                    "language": upsert_stmt.excluded.language,
                    "outcome_score": upsert_stmt.excluded.outcome_score,
                    "summary": upsert_stmt.excluded.summary,
                    "question_variants": upsert_stmt.excluded.question_variants,
                },
            )
            self._session.execute(upsert_stmt)
            # NOTE: commit is intentionally NOT called here.
            # The caller (use case) controls the transaction boundary.
        except SQLAlchemyError as exc:
            self._session.rollback()
            raise RepositoryError(f"Failed to save chunks: {exc}") from exc

    def get_by_id(self, chunk_id: str) -> Chunk | None:
        """Fetch by domain chunk_id string."""
        try:
            orm = self._session.query(ChunkORM).filter(ChunkORM.chunk_id == chunk_id).first()
            return self._to_domain(orm) if orm else None
        except SQLAlchemyError as exc:
            raise RepositoryError(f"Failed to get chunk by id: {exc}") from exc

    def list_by_source(self, source_doc: str) -> list[Chunk]:
        """Fetch all chunks for a given source document."""
        try:
            orms = self._session.query(ChunkORM).filter(ChunkORM.source_doc == source_doc).all()
            return [self._to_domain(orm) for orm in orms]
        except SQLAlchemyError as exc:
            raise RepositoryError(f"Failed to list chunks by source: {exc}") from exc

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
