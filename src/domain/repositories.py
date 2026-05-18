"""Abstract repository interfaces.

The domain defines the contract; interface adapters provide the implementation.
"""

from abc import ABC, abstractmethod

from domain.models import Chunk, SearchResult


class ChunkRepository(ABC):
    """Persistence interface for document chunks."""

    @abstractmethod
    def save_all(self, chunks: list[Chunk]) -> None:
        """Persist multiple chunks in a single transaction."""
        ...

    @abstractmethod
    def get_by_id(self, chunk_id: str) -> Chunk | None:
        """Fetch a single chunk by its UUID."""
        ...

    @abstractmethod
    def list_by_source(self, source_doc: str) -> list[Chunk]:
        """Retrieve all chunks originating from a given document."""
        ...


class SearchRepository(ABC):
    """Retrieval interface for vector and keyword search."""

    @abstractmethod
    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        """Execute approximate nearest-neighbor search via vector index.

        Args:
            query_embedding: Dense vector of the query.
            top_k: Maximum results to return.
            filters: Optional metadata filters (e.g., {"doc_type": "case_study"}).

        Returns:
            Ranked list of search results with similarity scores.
        """
        ...

    @abstractmethod
    def keyword_search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        """Execute full-text / BM25 keyword search.

        Args:
            query: Raw user query string.
            top_k: Maximum results to return.
            filters: Optional metadata filters.

        Returns:
            Ranked list of search results with keyword scores.
        """
        ...
