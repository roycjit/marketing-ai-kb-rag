"""Cross-encoder re-ranking for retrieved search results.

Cross-encoders score query-document pairs with higher accuracy than
bi-encoders (embedding cosine similarity), but are slower. They are
ideal for re-ranking the top-k candidates from hybrid search.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from domain.models import SearchResult

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """Re-rank search results using a cross-encoder model.

    The model is loaded once and reused. It scores (query, passage) pairs
    and returns results sorted by relevance.
    """

    def __init__(self, model_name: str | None = None) -> None:
        """Initialize with optional model override.

        Args:
            model_name: HuggingFace cross-encoder model name.
        """
        self._model_name = model_name or _DEFAULT_MODEL
        self._model: CrossEncoder | None = None

    def _load(self) -> CrossEncoder:
        """Lazy-load the cross-encoder model."""
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._model_name)
        return self._model

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Re-rank results by cross-encoder relevance scores.

        Args:
            query: Original user query.
            results: Candidate results from hybrid search.
            top_k: Maximum results to return. If None, returns all re-ranked.

        Returns:
            Results sorted by cross-encoder score (highest first).
        """
        if not results:
            return []

        model = self._load()
        pairs = [(query, r.chunk.content) for r in results]
        scores = model.predict(pairs, show_progress_bar=False)

        # Attach cross-encoder scores and sort descending
        scored = [
            SearchResult(
                chunk=r.chunk,
                similarity_score=float(score),
                keyword_score=r.keyword_score,
            )
            for r, score in zip(results, scores, strict=True)
        ]
        scored.sort(key=lambda sr: sr.similarity_score, reverse=True)

        if top_k is not None:
            return scored[:top_k]
        return scored
