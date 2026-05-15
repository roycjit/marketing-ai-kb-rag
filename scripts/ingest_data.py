#!/usr/bin/env python3
"""CLI entry point for ingesting blog export directories into the RAG database."""

import argparse
import sys
from pathlib import Path

from frameworks.config import (
    HEYFLOW_EXPORT_DIR,
    PERSPECTIVE_EXPORT_DIR,
)
from frameworks.database import SessionLocal
from interface_adapters.embeddings.sentence_transformer_client import (
    SentenceTransformerEmbedder,
)
from interface_adapters.parsers.markdown_parser import MarkdownParser
from interface_adapters.repositories.sqlalchemy_chunk_repo import (
    SQLAlchemyChunkRepository,
)
from use_cases.ingest_documents import IngestDocumentsUseCase


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ingest markdown blog exports into the funnel RAG database."
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Path to directory containing .md files",
    )
    parser.add_argument(
        "--heyflow",
        action="store_true",
        help=f"Shorthand for --source {HEYFLOW_EXPORT_DIR}",
    )
    parser.add_argument(
        "--perspective",
        action="store_true",
        help=f"Shorthand for --source {PERSPECTIVE_EXPORT_DIR}",
    )
    args = parser.parse_args()

    # Resolve source directory
    if args.heyflow:
        source_dir = HEYFLOW_EXPORT_DIR
    elif args.perspective:
        source_dir = PERSPECTIVE_EXPORT_DIR
    elif args.source:
        source_dir = Path(args.source)
    else:
        parser.error("Must specify one of: --source, --heyflow, --perspective")
        return 1

    if not source_dir.exists():
        print(f"ERROR: Source directory does not exist: {source_dir}", file=sys.stderr)
        return 1

    # Wire dependencies
    db = SessionLocal()
    try:
        repository = SQLAlchemyChunkRepository(db)
        use_case = IngestDocumentsUseCase(
            parser=MarkdownParser(),
            embedder=SentenceTransformerEmbedder(),
            repository=repository,
        )
        count = use_case.execute(source_dir)
        print(f"\n✓ Ingested {count} chunks from {source_dir}")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
