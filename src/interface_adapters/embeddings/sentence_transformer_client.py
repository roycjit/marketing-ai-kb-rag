"""Local embedding client using sentence-transformers."""

from typing import List

from sentence_transformers import SentenceTransformer

from frameworks.config import EMBEDDING_DIMENSION, EMBEDDING_MODEL


class SentenceTransformerEmbedder:
    """Generate dense vector embeddings via local sentence-transformers model.

    The model is loaded once and reused across calls for efficiency.
    """

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or EMBEDDING_MODEL
        self._model: SentenceTransformer | None = None

    def _load(self) -> SentenceTransformer:
        """Lazy-load the model."""
        if self._model is None:
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts.

        Args:
            texts: List of strings to embed.

        Returns:
            List of embedding vectors (one per input text).
        """
        if not texts:
            return []
        model = self._load()
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return [emb.tolist() for emb in embeddings]

    def embed_single(self, text: str) -> List[float]:
        """Embed a single text."""
        results = self.embed([text])
        return results[0]

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return EMBEDDING_DIMENSION
