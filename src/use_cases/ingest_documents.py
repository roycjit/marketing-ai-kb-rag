"""Ingestion use case — orchestrates the pipeline from raw files to database.

This is the application boundary: it knows about parsers, chunkers, embedders,
and repositories, but contains no business logic of its own.
"""

from __future__ import annotations

from pathlib import Path

import structlog

from domain.models import Chunk
from domain.repositories import ChunkRepository
from domain.services import chunk_document, compute_outcome_score
from interface_adapters.embeddings.sentence_transformer_client import (
    SentenceTransformerEmbedder,
)
from interface_adapters.parsers.markdown_parser import MarkdownParser

logger = structlog.get_logger(__name__)

# Batch size for embedding generation to balance memory and speed.
_EMBED_BATCH_SIZE = 32


class IngestDocumentsUseCase:
    """Orchestrate document ingestion: parse → chunk → score → embed → save."""

    def __init__(
        self,
        parser: MarkdownParser,
        embedder: SentenceTransformerEmbedder,
        repository: ChunkRepository,
    ) -> None:
        """Initialize with parser, embedder, and repository."""
        self._parser = parser
        self._embedder = embedder
        self._repository = repository

    def execute(self, source_dir: Path) -> int:
        """Ingest all markdown files from a directory.

        Args:
            source_dir: Path to directory containing .md files.

        Returns:
            Number of chunks successfully ingested.
        """
        logger.info("ingestion_started", source_dir=str(source_dir))

        # 1. Parse
        try:
            documents = self._parser.parse_directory(source_dir)
        except Exception as exc:
            logger.error("ingestion_parse_failed", error=str(exc))
            raise

        logger.info("documents_parsed", count=len(documents))

        # 2. Chunk with per-document error handling
        all_chunks: list[Chunk] = []
        failed_docs = 0
        for doc in documents:
            try:
                chunks = chunk_document(
                    source_doc=doc.source_doc,
                    title=doc.title,
                    body=doc.body,
                    metadata=doc.metadata,
                )
                all_chunks.extend(chunks)
            except Exception as exc:
                failed_docs += 1
                logger.error("chunk_document_failed", source_doc=doc.source_doc, error=str(exc))
                continue

        logger.info("chunks_created", count=len(all_chunks), failed_docs=failed_docs)

        if not all_chunks:
            logger.warning("no_chunks_created")
            return 0

        # 3. Compute outcome scores
        scored_chunks = [
            chunk.model_copy(update={"outcome_score": compute_outcome_score(chunk)})
            for chunk in all_chunks
        ]

        # 4. Embed in batches
        texts = [c.content for c in scored_chunks]
        embeddings: list[list[float]] = []
        for i in range(0, len(texts), _EMBED_BATCH_SIZE):
            batch = texts[i : i + _EMBED_BATCH_SIZE]
            try:
                batch_embeddings = self._embedder.embed(batch)
                embeddings.extend(batch_embeddings)
                logger.debug("embedding_batch", batch=i // _EMBED_BATCH_SIZE + 1)
            except Exception as exc:
                logger.error(
                    "embedding_batch_failed", batch=i // _EMBED_BATCH_SIZE + 1, error=str(exc)
                )
                raise

        # 5. Attach embeddings to chunks
        final_chunks = [
            chunk.model_copy(update={"embedding": emb})
            for chunk, emb in zip(scored_chunks, embeddings, strict=True)
        ]

        # 6. Persist
        try:
            self._repository.save_all(final_chunks)
            logger.info("ingestion_complete", chunks_saved=len(final_chunks))
        except Exception as exc:
            logger.error("ingestion_persist_failed", error=str(exc))
            raise

        return len(final_chunks)
