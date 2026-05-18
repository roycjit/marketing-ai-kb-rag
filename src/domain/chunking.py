"""Token-aware document chunking with sliding-window overlap.

Replaces the legacy word-count chunker with a tokenizer-aware pipeline
that respects the embedding model's max_seq_length and preserves
semantic boundaries via overlap.
"""

from __future__ import annotations

import re
import uuid
from datetime import UTC, datetime

# Runtime import for type annotations (transformers is a project dependency)
from transformers import PreTrainedTokenizerFast

from domain.models import Chunk

# Token budget for the embedding model (paraphrase-multilingual-MiniLM-L12-v2
# has max_seq_length == 128). We leave headroom for special tokens.
_TARGET_TOKENS = 100
_MAX_TOKENS = 120
_OVERLAP_TOKENS = 20

# Namespace for deterministic UUID5 generation.
_CHUNK_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


def chunk_document(
    source_doc: str,
    title: str,
    body: str,
    metadata: dict,
    doc_version: str = "1.0",
    tokenizer: TokenizerWrapper | None = None,
) -> list[Chunk]:
    """Split a parsed document into token-aware chunks with overlap.

    Strategy:
        1. Split by headers (##, ###) — preserves structural boundaries.
        2. Tokenize each section using the embedding model's tokenizer.
        3. Split token stream into ranges of ~100 tokens with 20-token overlap.
        4. Decode each range back to text.
        5. Generate deterministic UUID5 chunk IDs from content hash.

    Args:
        source_doc: Filename (used for traceability).
        title: Document title from frontmatter.
        body: Markdown body text.
        metadata: Parsed frontmatter dict.
        doc_version: Version string.
        tokenizer: Optional tokenizer wrapper. Created on first call if None.

    Returns:
        List of Chunk entities ready for embedding.
    """
    if tokenizer is None:
        tokenizer = TokenizerWrapper()

    doc_subtype = _infer_doc_subtype(source_doc, metadata)
    doc_type = doc_subtype
    language = _infer_language(body)
    last_updated = _parse_date(metadata.get("date", ""))

    sections = _split_by_headers(body)
    chunks: list[Chunk] = []

    for _section_idx, (header_path, section_text) in enumerate(sections):
        if not section_text.strip():
            continue

        # Preserve tables as single chunks when they fit
        if _contains_table(section_text):
            table_chunks = _split_table_section(section_text, tokenizer)
        else:
            table_chunks = [section_text]

        for text in table_chunks:
            token_ranges = tokenizer.split_with_overlap(
                text,
                target_tokens=_TARGET_TOKENS,
                overlap_tokens=_OVERLAP_TOKENS,
                max_tokens=_MAX_TOKENS,
            )

            for _range_idx, (start_tok, _end_tok, chunk_text) in enumerate(token_ranges):
                # Deterministic ID: UUID5 from content hash
                chunk_id = str(
                    uuid.uuid5(
                        _CHUNK_NAMESPACE,
                        f"{source_doc}:{doc_version}:{start_tok}:{hash(chunk_text) & 0xFFFFFFFF}",
                    )
                )
                section_path = f"{title} > {header_path}" if header_path else title

                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        content=chunk_text.strip(),
                        embedding=[],
                        source_doc=source_doc,
                        doc_version=doc_version,
                        section_path=section_path,
                        doc_type=doc_type,
                        doc_subtype=doc_subtype,
                        last_updated=last_updated,
                        language=language,
                        outcome_score=0.0,
                        summary=_extract_summary(chunk_text),
                        question_variants=[],
                    )
                )

    return chunks


class TokenizerWrapper:
    """Lazy-loaded tokenizer that mirrors the embedding model's vocabulary."""

    def __init__(self, model_name: str | None = None) -> None:
        """Initialize with optional model name override.

        Args:
            model_name: HuggingFace tokenizer name. Defaults to the project
                embedding model.
        """
        from frameworks.config import EMBEDDING_MODEL

        self._model_name = model_name or EMBEDDING_MODEL
        self._tokenizer: "PreTrainedTokenizerFast | None" = None  # noqa: UP037

    def _load(self) -> "PreTrainedTokenizerFast":  # noqa: UP037
        """Lazy-load the tokenizer (lightweight — no model weights)."""
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            # We intentionally chunk long sequences ourselves; suppress the
            # false-positive "sequence length > model_max_length" warning.
            self._tokenizer.model_max_length = int(1e30)
        return self._tokenizer

    def count_tokens(self, text: str) -> int:
        """Return the number of tokens for a given text."""
        tok = self._load()
        return len(tok.encode(text, add_special_tokens=False))

    def split_with_overlap(
        self,
        text: str,
        target_tokens: int = _TARGET_TOKENS,
        overlap_tokens: int = _OVERLAP_TOKENS,
        max_tokens: int = _MAX_TOKENS,
    ) -> list[tuple[int, int, str]]:
        """Split text into overlapping token ranges.

        Returns:
            List of (start_token_idx, end_token_idx, decoded_text) tuples.
        """
        tok = self._load()
        encoded = tok.encode(text, add_special_tokens=False)
        total_tokens = len(encoded)

        if total_tokens <= max_tokens:
            return [(0, total_tokens, text)]

        ranges: list[tuple[int, int, str]] = []
        step = target_tokens - overlap_tokens
        start = 0

        while start < total_tokens:
            end = min(start + target_tokens, total_tokens)
            chunk_ids = encoded[start:end]
            chunk_text = tok.decode(chunk_ids, skip_special_tokens=True)
            ranges.append((start, end, chunk_text))

            if end == total_tokens:
                break
            start += step

        return ranges


# --------------------------------------------------------------------------- #
# Legacy helpers (kept for backward compatibility, moved here from services.py)
# --------------------------------------------------------------------------- #


_SUBTYPE_PRECEDENCE = ["explainer", "comparison", "guide", "case_study"]


def _infer_doc_subtype(source_doc: str, metadata: dict) -> str:
    """Infer document subtype from filename and frontmatter."""
    lower_name = source_doc.lower()
    category = str(metadata.get("category", "")).lower()

    if lower_name.startswith("case-study") or "case study" in category:
        return "case_study"
    if any(k in lower_name for k in ["guide", "best-practice", "best-practices"]):
        return "guide"
    if any(k in lower_name for k in ["alternative", "comparison", "vs-", "versus"]):
        return "comparison"
    return "explainer"


def _infer_language(text: str) -> str:
    """Simple heuristic: if >5% of words are German-specific, mark as de."""
    german_markers = {
        "die",
        "der",
        "das",
        "und",
        "ist",
        "für",
        "mit",
        "von",
        "zu",
        "den",
        "auf",
        "ein",
        "eine",
        "sich",
        "nicht",
        "als",
        "auch",
        "werden",
        "bei",
        "nach",
        "aus",
        "dass",
        "kann",
        "mehr",
        "über",
        "sind",
        "wie",
        "einen",
    }
    words = re.findall(r"\b\w+\b", text.lower())
    if not words:
        return "en"
    german_count = sum(1 for w in words if w in german_markers)
    return "de" if german_count / len(words) > 0.05 else "en"


def _parse_date(date_val: object) -> datetime:
    """Parse ISO-8601 date string (or date object) with fallback to epoch."""
    if not date_val:
        return datetime(1970, 1, 1, tzinfo=UTC)
    if isinstance(date_val, datetime):
        return date_val.replace(tzinfo=date_val.tzinfo or UTC)
    if hasattr(date_val, "year"):
        y: int = date_val.year  # type: ignore[attr-defined]
        m: int = date_val.month  # type: ignore[attr-defined]
        d: int = date_val.day  # type: ignore[attr-defined]
        return datetime(y, m, d, tzinfo=UTC)
    if isinstance(date_val, str):
        for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(date_val, fmt).replace(tzinfo=UTC)
            except ValueError:
                continue
    return datetime(1970, 1, 1, tzinfo=UTC)


def _split_by_headers(body: str) -> list[tuple[str, str]]:
    """Split markdown body by ## and ### headers.

    Preserves any text that appears before the first header as an
    un-named introductory section.
    """
    pattern = re.compile(r"^(#{2,3})\s+(.+)$", re.MULTILINE)
    matches = list(pattern.finditer(body))

    if not matches:
        return [("", body)]

    sections: list[tuple[str, str]] = []

    # Capture text before the first header (if any)
    first_match_start = matches[0].start()
    if first_match_start > 0:
        intro = body[:first_match_start].strip()
        if intro:
            sections.append(("", intro))

    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        header_text = match.group(2).strip()
        section_text = body[start:end].strip()
        sections.append((header_text, section_text))

    return sections


def _contains_table(text: str) -> bool:
    """Detect markdown table by pipe characters."""
    lines = text.splitlines()
    pipe_lines = [line for line in lines if line.strip().startswith("|")]
    return len(pipe_lines) >= 2


def _split_table_section(text: str, tokenizer: TokenizerWrapper) -> list[str]:
    """Keep tables as single chunks if under token limit, else split at row boundaries."""
    if tokenizer.count_tokens(text) <= _MAX_TOKENS:
        return [text]

    # Split by blank lines first (separates table from surrounding text).
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    result: list[str] = []
    for para in paragraphs:
        # If this paragraph is a table and still too long, split into rows.
        if _contains_table(para) and tokenizer.count_tokens(para) > _MAX_TOKENS:
            rows = [r.strip() for r in para.splitlines() if r.strip()]
            if len(rows) >= 2:
                # Keep header + separator together, then add data rows while they fit.
                header = rows[0]
                separator = rows[1]
                base = f"{header}\n{separator}"
                current = base
                for row in rows[2:]:
                    candidate = f"{current}\n{row}"
                    if tokenizer.count_tokens(candidate) > _MAX_TOKENS:
                        result.append(current)
                        current = f"{base}\n{row}"
                    else:
                        current = candidate
                if current:
                    result.append(current)
            else:
                result.append(para)
        else:
            result.append(para)

    return result


def _extract_summary(text: str) -> str:
    """Extract a simple summary: first sentence or first 200 chars."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    if sentences and len(sentences[0]) > 20:
        return sentences[0][:300]
    return text[:300].strip()
