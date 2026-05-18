"""Concrete SearchRepository implementation using PostgreSQL + pgvector.

Supports:
- Approximate nearest neighbor via HNSW index (cosine distance)
- Full-text keyword search via stored PostgreSQL tsvector
- Metadata filtering on all indexed columns
"""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.orm import Session

from domain.models import Chunk, SearchResult
from domain.repositories import SearchRepository

# Explicit allowlist of filterable columns to prevent SQL injection
_FILTERABLE_COLUMNS: frozenset[str] = frozenset(
    {
        "chunk_id",
        "source_doc",
        "doc_version",
        "doc_type",
        "doc_subtype",
        "language",
    }
)


class PgVectorSearchRepository(SearchRepository):
    """Hybrid search backed by pgvector and PostgreSQL full-text search."""

    def __init__(self, session: Session) -> None:
        """Initialize with a SQLAlchemy session."""
        self._session = session

    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        """Vector similarity search with optional metadata filters.

        Uses cosine distance operator (<=>) provided by pgvector.
        """
        # Build dynamic WHERE clause from filters
        where_clauses = []
        params: dict = {"embedding": query_embedding, "top_k": top_k}

        if filters:
            for key, value in filters.items():
                if key in _FILTERABLE_COLUMNS:
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
        results: list[SearchResult] = []
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
                question_variants=(
                    list(row["question_variants"]) if row["question_variants"] else []
                ),
            )
            results.append(SearchResult(chunk=chunk, similarity_score=float(row["similarity"])))

        return results

    def keyword_search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        """Full-text keyword search using PostgreSQL tsvector.

        Queries the indexed ``to_tsvector`` expression directly. The initial
        migration creates a GIN index on this expression, so PostgreSQL can
        use it for fast full-text retrieval without a stored column.
        """
        where_clauses = []
        params: dict = {"query": query, "top_k": top_k}

        if filters:
            for key, value in filters.items():
                if key in _FILTERABLE_COLUMNS:
                    where_clauses.append(f"{key} = :{key}")
                    params[key] = value

        where_sql = " AND ".join(where_clauses) if where_clauses else "TRUE"

        # The GIN index from migration 001 covers this exact expression.
        tsvector_expr = "to_tsvector('simple', content || ' ' || COALESCE(summary, ''))"

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
                ts_rank({tsvector_expr}, plainto_tsquery('simple', :query)) AS rank
            FROM document_chunks
            WHERE
                {where_sql}
                AND {tsvector_expr} @@ plainto_tsquery('simple', :query)
            ORDER BY rank DESC
            LIMIT :top_k
        """)

        rows = self._session.execute(sql, params).mappings().all()
        results: list[SearchResult] = []
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
                question_variants=(
                    list(row["question_variants"]) if row["question_variants"] else []
                ),
            )
            results.append(
                SearchResult(
                    chunk=chunk,
                    similarity_score=0.0,
                    keyword_score=float(row["rank"]),
                )
            )

        return results
