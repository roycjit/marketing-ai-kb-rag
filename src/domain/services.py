"""Pure domain services — no I/O, no framework dependencies.

These functions encapsulate business rules that don't naturally belong
to a single entity.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from domain.models import Chunk

# Document subtype hierarchy for conflict resolution.
# Higher index = higher authority.
_SUBTYPE_PRECEDENCE = ["explainer", "comparison", "guide", "case_study"]

# Rough word count target per chunk.
_TARGET_CHUNK_WORDS = 250
_MAX_CHUNK_WORDS = 500


def compute_outcome_score(chunk: Chunk) -> float:
    """Derive a quality signal from document metadata.

    Case studies with explicit metrics score highest.
    Guides and comparisons score moderately.
    Generic explainers score lowest.

    Returns:
        Float in [0.0, 1.0].
    """
    base_score = {
        "case_study": 0.9,
        "guide": 0.6,
        "comparison": 0.5,
        "explainer": 0.3,
    }.get(chunk.doc_subtype, 0.3)

    # Small recency bonus: documents from the last 12 months get +0.1
    if chunk.last_updated:
        age_days = (datetime.utcnow() - chunk.last_updated).days
        if age_days < 365:
            base_score = min(1.0, base_score + 0.1)

    return round(base_score, 2)


def resolve_conflicts(chunks: List[Chunk]) -> List[Chunk]:
    """When multiple chunks cover the same topic, prefer authoritative sources.

    Resolution rules (in order):
    1. Higher subtype precedence (case_study > guide > comparison > explainer)
    2. More recent last_updated
    3. Higher outcome_score

    Args:
        chunks: Potentially overlapping chunks from different documents.

    Returns:
        Deduplicated list with conflicts resolved.
    """
    if not chunks:
        return []

    def _sort_key(c: Chunk) -> tuple:
        precedence = _SUBTYPE_PRECEDENCE.index(c.doc_subtype) if c.doc_subtype in _SUBTYPE_PRECEDENCE else -1
        return (
            precedence,
            c.last_updated.timestamp() if c.last_updated else 0,
            c.outcome_score,
        )

    # Group by source_doc and section_path to detect overlap
    seen: dict[str, Chunk] = {}
    for chunk in chunks:
        key = f"{chunk.source_doc}::{chunk.section_path}"
        if key not in seen:
            seen[key] = chunk
        else:
            existing = seen[key]
            if _sort_key(chunk) > _sort_key(existing):
                seen[key] = chunk

    return list(seen.values())


# --------------------------------------------------------------------------- #
# Retrieval Services
# --------------------------------------------------------------------------- #


def rrf_fuse(
    semantic_results: List["SearchResult"],
    keyword_results: List["SearchResult"],
    k: int = 60,
) -> List["SearchResult"]:
    """Fuse semantic and keyword results via Reciprocal Rank Fusion.

    RRF_score(d) = sum_m 1 / (k + rank_m(d))

    Documents present in both result sets receive boosted scores.

    Args:
        semantic_results: Ranked list from vector similarity search.
        keyword_results: Ranked list from full-text keyword search.
        k: RRF constant (standard value = 60).

    Returns:
        Fused and re-ranked list of search results.
    """
    from domain.models import SearchResult

    scores: dict[str, float] = {}
    result_map: dict[str, SearchResult] = {}

    for rank, sr in enumerate(semantic_results, start=1):
        cid = sr.chunk.chunk_id
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
        result_map[cid] = sr

    for rank, sr in enumerate(keyword_results, start=1):
        cid = sr.chunk.chunk_id
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
        if cid not in result_map:
            result_map[cid] = sr

    # Sort by RRF score descending
    sorted_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    return [
        SearchResult(
            chunk=result_map[cid].chunk,
            similarity_score=scores[cid],
            keyword_score=result_map[cid].keyword_score,
        )
        for cid in sorted_ids
    ]


def outcome_rerank(results: List["SearchResult"], top_k: int = 10) -> List["SearchResult"]:
    """Re-rank fused results using document-encoded outcome signals.

    Boosts documents by:
    1. Outcome score (case studies with metrics)
    2. Recency (documents from last 12 months)
    3. Document subtype precedence (case_study > guide > comparison > explainer)

    The boost is additive to the RRF score to preserve the fused ranking
    while surfacing higher-quality content.
    """
    from domain.models import SearchResult

    def _boost(sr: SearchResult) -> float:
        base = sr.similarity_score
        chunk = sr.chunk

        # Outcome score boost: up to +0.15
        base += chunk.outcome_score * 0.15

        # Recency boost: +0.05 for documents < 365 days old
        if chunk.last_updated:
            age_days = (datetime.utcnow() - chunk.last_updated).days
            if age_days < 365:
                base += 0.05

        # Subtype boost
        subtype_boost = {
            "case_study": 0.10,
            "guide": 0.05,
            "comparison": 0.02,
            "explainer": 0.0,
        }.get(chunk.doc_subtype, 0.0)
        base += subtype_boost

        return base

    boosted = sorted(results, key=_boost, reverse=True)
    return boosted[:top_k]


# --------------------------------------------------------------------------- #
# Chunking Service
# --------------------------------------------------------------------------- #


def chunk_document(
    source_doc: str,
    title: str,
    body: str,
    metadata: dict,
    doc_version: str = "1.0",
) -> List[Chunk]:
    """Split a parsed document into semantically coherent chunks.

    Strategy:
    1. Split by headers (##, ###) — each section is a candidate chunk.
    2. If a section exceeds _MAX_CHUNK_WORDS, split by paragraphs.
    3. If a paragraph exceeds _MAX_CHUNK_WORDS, split by sentences.
    4. Preserve table blocks as single chunks where possible.
    5. Include header ancestry in section_path.

    Args:
        source_doc: Filename (used for traceability).
        title: Document title from frontmatter.
        body: Markdown body text.
        metadata: Parsed frontmatter dict.
        doc_version: Version string.

    Returns:
        List of Chunk entities ready for embedding.
    """
    doc_subtype = _infer_doc_subtype(source_doc, metadata)
    doc_type = doc_subtype  # Simplified: type == subtype for MVP
    language = _infer_language(body)
    last_updated = _parse_date(metadata.get("date", ""))

    sections = _split_by_headers(body)
    chunks: List[Chunk] = []

    for section_idx, (header_path, section_text) in enumerate(sections):
        if not section_text.strip():
            continue

        # If section contains a table, try to keep it whole
        if _contains_table(section_text):
            sub_chunks = _split_table_section(section_text)
        else:
            sub_chunks = _split_by_length(section_text)

        for idx, text in enumerate(sub_chunks):
            header_slug = header_path.replace(' ', '-').replace('**', '') if header_path else 'intro'
            # Truncate slug so chunk_id stays well under 255 chars
            header_slug = header_slug[:60]
            chunk_id = f"{source_doc.replace('.md', '')}--{section_idx}--{header_slug}--{idx}"
            section_path = f"{title} > {header_path}" if header_path else title

            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    content=text.strip(),
                    embedding=[],  # populated later by embedder
                    source_doc=source_doc,
                    doc_version=doc_version,
                    section_path=section_path,
                    doc_type=doc_type,
                    doc_subtype=doc_subtype,
                    last_updated=last_updated,
                    language=language,
                    outcome_score=0.0,  # populated later
                    summary=_extract_summary(text),
                    question_variants=[],
                )
            )

    return chunks


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
    """Simple heuristic: if >20% of words are German-specific, mark as de."""
    # Common German words that rarely appear in English
    german_markers = {
        "die", "der", "das", "und", "ist", "für", "mit", "von", "zu", "den",
        "auf", "ein", "eine", "sich", "nicht", "als", "auch", "werden", "bei",
        "nach", "aus", "dass", "kann", "mehr", "über", "sind", "wie", "einen",
    }
    words = re.findall(r"\b\w+\b", text.lower())
    if not words:
        return "en"
    german_count = sum(1 for w in words if w in german_markers)
    return "de" if german_count / len(words) > 0.05 else "en"


def _parse_date(date_val) -> datetime:
    """Parse ISO-8601 date string (or date object) with fallback to epoch."""
    if not date_val:
        return datetime(1970, 1, 1, tzinfo=timezone.utc)
    # frontmatter/yaml may already parse dates into date/datetime objects
    if isinstance(date_val, datetime):
        return date_val.replace(tzinfo=date_val.tzinfo or timezone.utc)
    if hasattr(date_val, "year"):  # date object
        return datetime(date_val.year, date_val.month, date_val.day, tzinfo=timezone.utc)
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(date_val, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return datetime(1970, 1, 1, tzinfo=timezone.utc)


def _split_by_headers(body: str) -> List[Tuple[str, str]]:
    """Split markdown body by ## and ### headers.

    Returns list of (header_path, section_text) tuples.
    """
    # Match ## or ### headers
    pattern = re.compile(r"^(#{2,3})\s+(.+)$", re.MULTILINE)
    matches = list(pattern.finditer(body))

    if not matches:
        return [("", body)]

    sections: List[Tuple[str, str]] = []
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
    pipe_lines = [l for l in lines if l.strip().startswith("|")]
    return len(pipe_lines) >= 2


def _split_table_section(text: str) -> List[str]:
    """Keep tables as single chunks if under size limit, else split at table boundaries."""
    words = len(text.split())
    if words <= _MAX_CHUNK_WORDS:
        return [text]
    # Split by double newline to separate table from surrounding text
    parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    return _merge_small_parts(parts)


def _split_by_length(text: str) -> List[str]:
    """Split text by paragraphs, then by sentences if paragraphs are too long."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    result: List[str] = []

    for para in paragraphs:
        word_count = len(para.split())
        if word_count <= _MAX_CHUNK_WORDS:
            result.append(para)
        else:
            # Split by sentences
            sentences = re.split(r"(?<=[.!?])\s+", para)
            current = ""
            for sent in sentences:
                if len((current + " " + sent).split()) > _TARGET_CHUNK_WORDS and current:
                    result.append(current.strip())
                    current = sent
                else:
                    current = (current + " " + sent).strip()
            if current:
                result.append(current)

    return _merge_small_parts(result)


def _merge_small_parts(parts: List[str]) -> List[str]:
    """Merge consecutive small parts until they reach target size."""
    if not parts:
        return []

    merged: List[str] = []
    current = parts[0]
    for part in parts[1:]:
        if len((current + "\n\n" + part).split()) <= _TARGET_CHUNK_WORDS:
            current = current + "\n\n" + part
        else:
            merged.append(current)
            current = part
    merged.append(current)
    return merged


def _extract_summary(text: str) -> str:
    """Extract a simple summary: first sentence or first 200 chars."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    if sentences and len(sentences[0]) > 20:
        return sentences[0][:300]
    return text[:300].strip()
