"""Tests for MarkdownParser."""

from pathlib import Path

import pytest

from interface_adapters.parsers.markdown_parser import MarkdownParser


@pytest.fixture
def sample_md(tmp_path: Path) -> Path:
    content = """---
title: Best Funnel Builders for Solar
date: 2026-03-10
category: Guides & best practices
---

# Best Funnel Builders for Solar

## Why Solar Needs Special Funnels

Solar is a high-consideration purchase.

## Key Features

- Mobile-first design
- TCPA compliance
- Built-in calculations
"""
    fpath = tmp_path / "solar-guide.md"
    fpath.write_text(content, encoding="utf-8")
    return fpath


class TestParseFile:
    def test_extracts_frontmatter(self, sample_md: Path):
        parser = MarkdownParser()
        doc = parser.parse_file(sample_md)

        assert doc.title == "Best Funnel Builders for Solar"
        assert doc.source_doc == "solar-guide.md"
        assert doc.metadata["date"] == "2026-03-10"
        assert doc.metadata["category"] == "Guides & best practices"

    def test_extracts_body(self, sample_md: Path):
        parser = MarkdownParser()
        doc = parser.parse_file(sample_md)

        assert "Why Solar Needs Special Funnels" in doc.body
        assert "Mobile-first design" in doc.body
        assert "---" not in doc.body  # Frontmatter delimiter stripped

    def test_uses_filename_as_fallback_title(self, tmp_path: Path):
        fpath = tmp_path / "no-frontmatter.md"
        fpath.write_text("# Just a title\n\nSome content.")

        parser = MarkdownParser()
        doc = parser.parse_file(fpath)
        assert doc.title == "No Frontmatter"


class TestParseDirectory:
    def test_parses_all_markdown_files(self, tmp_path: Path):
        (tmp_path / "a.md").write_text("# A")
        (tmp_path / "b.md").write_text("# B")
        (tmp_path / "skip.txt").write_text("not markdown")

        parser = MarkdownParser()
        docs = parser.parse_directory(tmp_path)

        assert len(docs) == 2
        assert {d.source_doc for d in docs} == {"a.md", "b.md"}

    def test_raises_on_non_directory(self, tmp_path: Path):
        parser = MarkdownParser()
        with pytest.raises(ValueError, match="Not a directory"):
            parser.parse_directory(tmp_path / "file.md")
