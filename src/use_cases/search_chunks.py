"""Hybrid search use case — orchestrates keyword + semantic + RRF + re-ranking."""

from __future__ import annotations

import structlog

from domain.models import SearchResult
from domain.repositories import SearchRepository
from domain.services import outcome_rerank, rrf_fuse
from domain.validation import sanitize_query
from interface_adapters.embeddings.cross_encoder_client import CrossEncoderReranker
from interface_adapters.embeddings.sentence_transformer_client import (
    SentenceTransformerEmbedder,
)

logger = structlog.get_logger(__name__)


class HybridSearchUseCase:
    """Execute hybrid search: semantic + keyword in parallel, fuse with RRF, re-rank."""

    def __init__(
        self,
        search_repository: SearchRepository,
        embedder: SentenceTransformerEmbedder,
        reranker: CrossEncoderReranker | None = None,
    ) -> None:
        """Initialize with search repository, embedder, and optional reranker."""
        self._search_repo = search_repository
        self._embedder = embedder
        self._reranker = reranker

    def execute(
        self,
        query: str,
        top_k: int = 10,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        """Run hybrid search pipeline.

        Args:
            query: Raw user query string.
            top_k: Maximum results to return after fusion and re-ranking.
            filters: Optional metadata filters (e.g., {"language": "en"}).

        Returns:
            Re-ranked list of search results with citations.
        """
        safe_query = sanitize_query(query)
        logger.info("hybrid_search_started", query=safe_query, top_k=top_k, filters=filters)

        # 1. Semantic search
        query_embedding = self._embedder.embed_single(safe_query)
        semantic_results = self._search_repo.similarity_search(
            query_embedding=query_embedding,
            top_k=top_k * 2,  # Over-fetch to give RRF more candidates
            filters=filters,
        )
        logger.debug("semantic_results", count=len(semantic_results))

        # 2. Keyword search
        keyword_results = self._search_repo.keyword_search(
            query=safe_query,
            top_k=top_k * 2,
            filters=filters,
        )
        logger.debug("keyword_results", count=len(keyword_results))

        # 3. RRF fusion
        fused = rrf_fuse(semantic_results, keyword_results, k=60)
        logger.debug("fused_results", count=len(fused))

        # 4. Outcome re-ranking
        reranked: list[SearchResult] = outcome_rerank(fused, top_k=top_k)
        logger.info("hybrid_search_complete", returned=len(reranked))

        # 5. Optional cross-encoder re-ranking (more accurate, slower)
        if self._reranker is not None and reranked:
            reranked = self._reranker.rerank(safe_query, reranked, top_k=top_k)
            logger.info("cross_encoder_rerank_complete", returned=len(reranked))

        return reranked
