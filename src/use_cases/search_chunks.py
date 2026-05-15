"""Hybrid search use case — orchestrates keyword + semantic + RRF + re-ranking."""

from typing import List, Optional

import structlog

from domain.models import SearchResult
from domain.repositories import SearchRepository
from domain.services import outcome_rerank, rrf_fuse
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
    ) -> None:
        self._search_repo = search_repository
        self._embedder = embedder

    def execute(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[dict] = None,
    ) -> List[SearchResult]:
        """Run hybrid search pipeline.

        Args:
            query: Raw user query string.
            top_k: Maximum results to return after fusion and re-ranking.
            filters: Optional metadata filters (e.g., {"language": "en"}).

        Returns:
            Re-ranked list of search results with citations.
        """
        logger.info("hybrid_search_started", query=query, top_k=top_k, filters=filters)

        # 1. Semantic search
        query_embedding = self._embedder.embed_single(query)
        semantic_results = self._search_repo.similarity_search(
            query_embedding=query_embedding,
            top_k=top_k * 2,  # Over-fetch to give RRF more candidates
            filters=filters,
        )
        logger.debug("semantic_results", count=len(semantic_results))

        # 2. Keyword search
        keyword_results = self._search_repo.keyword_search(
            query=query,
            top_k=top_k * 2,
            filters=filters,
        )
        logger.debug("keyword_results", count=len(keyword_results))

        # 3. RRF fusion
        fused = rrf_fuse(semantic_results, keyword_results, k=60)
        logger.debug("fused_results", count=len(fused))

        # 4. Outcome re-ranking
        reranked = outcome_rerank(fused, top_k=top_k)
        logger.info("hybrid_search_complete", returned=len(reranked))

        return reranked
