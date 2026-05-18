"""Markdown parser with frontmatter support.

Extracts structured documents from markdown files, including
metadata, title inference, and body content.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import frontmatter

from domain.validation import validate_file_size


@dataclass
class ParsedDocument:
    """A single parsed markdown document."""

    source_doc: str
    title: str
    body: str
    metadata: dict[str, Any]


class MarkdownParser:
    """Parse markdown files into structured documents."""

    def parse_file(self, file_path: Path) -> ParsedDocument:
        """Parse a single markdown file.

        Args:
            file_path: Path to the markdown file.

        Returns:
            ParsedDocument with extracted metadata and content.
        """
        file_size = os.path.getsize(file_path)
        validate_file_size(file_size)
        return self._parse_file(file_path)

    def parse_directory(self, directory: Path) -> list[ParsedDocument]:
        """Parse all .md files in a directory.

        Args:
            directory: Path to directory containing markdown files.

        Returns:
            List of parsed documents.

        Raises:
            ValidationError: If a file exceeds size limits or path is not a directory.
        """
        if not directory.is_dir():
            raise ValueError(f"Not a directory: {directory}")
        """Parse all .md files in a directory.

        Args:
            directory: Path to directory containing markdown files.

        Returns:
            List of parsed documents.

        Raises:
            ValidationError: If a file exceeds size limits.
        """
        documents: list[ParsedDocument] = []
        for file_path in sorted(directory.glob("*.md")):
            # Validate file size before reading
            file_size = os.path.getsize(file_path)
            validate_file_size(file_size)

            doc = self._parse_file(file_path)
            if doc:
                documents.append(doc)
        return documents

    def _parse_file(self, file_path: Path) -> ParsedDocument:
        """Parse a single markdown file."""
        content = file_path.read_text(encoding="utf-8")

        try:
            post = frontmatter.loads(content)
        except Exception:
            # Malformed frontmatter — treat entire file as body
            post = frontmatter.Post(content)

        metadata = post.metadata or {}
        body = post.content.strip()
        raw_title = metadata.get("title")
        fallback = self._infer_title(body) or file_path.stem
        title = raw_title if isinstance(raw_title, str) else fallback

        return ParsedDocument(
            source_doc=file_path.name,
            title=title,
            body=body,
            metadata=metadata,
        )

    @staticmethod
    def _infer_title(body: str) -> str | None:
        """Extract the first H1 heading as title."""
        for line in body.splitlines():
            stripped = line.strip()
            if stripped.startswith("# "):
                return stripped[2:].strip()
        return None
