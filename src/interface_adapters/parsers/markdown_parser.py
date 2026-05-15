"""Markdown document parser — extracts frontmatter and body."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import frontmatter


@dataclass(frozen=True)
class ParsedDocument:
    """Result of parsing a single markdown file."""

    source_doc: str
    title: str
    body: str
    metadata: Dict


class MarkdownParser:
    """Parse markdown blog exports with YAML frontmatter."""

    def __init__(self, encoding: str = "utf-8") -> None:
        self._encoding = encoding

    def parse_file(self, path: Path) -> ParsedDocument:
        """Parse a single markdown file.

        Args:
            path: Path to .md file.

        Returns:
            ParsedDocument with title, body, and metadata.
        """
        raw = path.read_text(encoding=self._encoding)
        try:
            post = frontmatter.loads(raw)
            metadata = dict(post.metadata)
            body = post.content
        except Exception:
            # Malformed frontmatter (e.g., unquoted colons in URLs).
            # Fall back to plain markdown — no metadata.
            metadata = {}
            body = raw

        title = metadata.get("title", path.stem.replace("-", " ").title())
        # Include parent directory to disambiguate same-named files across sources
        source_doc = f"{path.parent.name}/{path.name}"
        return ParsedDocument(
            source_doc=source_doc,
            title=title,
            body=body,
            metadata=metadata,
        )

    def parse_directory(self, directory: Path) -> List[ParsedDocument]:
        """Parse all markdown files in a directory (non-recursive).

        Args:
            directory: Path to directory containing .md files.

        Returns:
            List of ParsedDocument, sorted by source_doc name.
        """
        if not directory.is_dir():
            raise ValueError(f"Not a directory: {directory}")

        files = sorted(directory.glob("*.md"))
        return [self.parse_file(f) for f in files]
