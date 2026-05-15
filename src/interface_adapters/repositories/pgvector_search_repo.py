"""Concrete SearchRepository implementation using PostgreSQL + pgvector.

Supports:
- Approximate nearest neighbor via HNSW index (cosine distance)
- Full-text keyword search via PostgreSQL tsvector
- Metadata filtering on all indexed columns
"""

from typing import List, Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

from domain.models import Chunk, SearchResult
from domain.repositories import SearchRepository
from interface_adapters.repositories.orm_models import ChunkORM


class PgVectorSearchRepository(SearchRepository):
    """Hybrid search backed by pgvector and PostgreSQL full-text search."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[dict] = None,
    ) -> List[SearchResult]:
        """Vector similarity search with optional metadata filters.

        Uses cosine distance operator (<=>) provided by pgvector.
        """
        # Build dynamic WHERE clause from filters
        where_clauses = []
        params: dict = {"embedding": query_embedding, "top_k": top_k}

        if filters:
            for key, value in filters.items():
                if hasattr(ChunkORM, key):
                    where_clauses.append(f"{key} = :{key}")
                    params[key] = value

        where_sql = " AND ".join(where_clauses) if where_clauses else "TRUE"

        # Cosine similarity = 1 - cosine distance
        sql = text(f"""
            SELECT
                chunk_id,
                content,
                source_doc,
                doc_version,
                section_path,
                doc_type,
                doc_subtype,
                last_updated,
                language,
                outcome_score,
                summary,
                question_variants,
                1 - (embedding <=> CAST(:embedding AS vector)) AS similarity
            FROM document_chunks
            WHERE {where_sql}
            ORDER BY embedding <=> CAST(:embedding AS vector)
            LIMIT :top_k
        """)

        rows = self._session.execute(sql, params).mappings().all()
        results: List[SearchResult] = []
        for row in rows:
            chunk = Chunk(
                chunk_id=row["chunk_id"],
                content=row["content"],
                embedding=[],
                source_doc=row["source_doc"],
                doc_version=row["doc_version"],
                section_path=row["section_path"],
                doc_type=row["doc_type"],
                doc_subtype=row["doc_subtype"],
                last_updated=row["last_updated"],
                language=row["language"],
                outcome_score=row["outcome_score"],
                summary=row["summary"],
                question_variants=list(row["question_variants"]) if row["question_variants"] else [],
            )
            results.append(SearchResult(chunk=chunk, similarity_score=float(row["similarity"])))

        return results

    def keyword_search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[dict] = None,
    ) -> List[SearchResult]:
        """Full-text keyword search using PostgreSQL tsvector.

        NOTE: This assumes a tsvector index exists on content + summary.
        The index should be created in a migration:
            CREATE INDEX idx_chunks_fts ON document_chunks
            USING GIN (to_tsvector('english', content || ' ' || COALESCE(summary, '')));
        """
        where_clauses = []
        params: dict = {"query": query, "top_k": top_k}

        if filters:
            for key, value in filters.items():
                if hasattr(ChunkORM, key):
                    where_clauses.append(f"{key} = :{key}")
                    params[key] = value

        where_sql = " AND ".join(where_clauses) if where_clauses else "TRUE"

        # Use plainto_tsquery for simple keyword queries
        sql = text(f"""
            SELECT
                chunk_id,
                content,
                source_doc,
                doc_version,
                section_path,
                doc_type,
                doc_subtype,
                last_updated,
                language,
                outcome_score,
                summary,
                question_variants,
                ts_rank(
                    to_tsvector('simple', content || ' ' || COALESCE(summary, '')),
                    plainto_tsquery('simple', :query)
                ) AS rank
            FROM document_chunks
            WHERE
                {where_sql}
                AND to_tsvector('simple', content || ' ' || COALESCE(summary, ''))
                    @@ plainto_tsquery('simple', :query)
            ORDER BY rank DESC
            LIMIT :top_k
        """)

        rows = self._session.execute(sql, params).mappings().all()
        results: List[SearchResult] = []
        for row in rows:
            chunk = Chunk(
                chunk_id=row["chunk_id"],
                content=row["content"],
                embedding=[],
                source_doc=row["source_doc"],
                doc_version=row["doc_version"],
                section_path=row["section_path"],
                doc_type=row["doc_type"],
                doc_subtype=row["doc_subtype"],
                last_updated=row["last_updated"],
                language=row["language"],
                outcome_score=row["outcome_score"],
                summary=row["summary"],
                question_variants=list(row["question_variants"]) if row["question_variants"] else [],
            )
            results.append(
                SearchResult(
                    chunk=chunk,
                    similarity_score=0.0,
                    keyword_score=float(row["rank"]),
                )
            )

        return results
