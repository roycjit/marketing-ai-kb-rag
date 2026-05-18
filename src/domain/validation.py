"""Input validation and sanitization utilities.

These helpers guard against prompt injection, oversized inputs,
and malformed data at the domain boundary.
"""

from __future__ import annotations

import re
from typing import Final

from domain.exceptions import ValidationError

# --------------------------------------------------------------------------- #
# Limits
# --------------------------------------------------------------------------- #

MAX_BRIEF_LENGTH: Final[int] = 10_000
MAX_QUERY_LENGTH: Final[int] = 500
MAX_FILE_SIZE_MB: Final[int] = 10
MAX_FILE_SIZE_BYTES: Final[int] = MAX_FILE_SIZE_MB * 1024 * 1024

# Chars that could break string formatting or be used for prompt injection
_CONTROL_CHAR_PATTERN: Final[re.Pattern[str]] = re.compile(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]")

# Simple prompt-injection heuristics (defense-in-depth, not foolproof)
_INJECTION_PATTERNS: Final[list[re.Pattern[str]]] = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"ignore\s+(the\s+)?(above|prior)\s+(instructions|prompt)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(a\s+)?", re.IGNORECASE),
    re.compile(r"new\s+instruction\s*:", re.IGNORECASE),
    re.compile(r"system\s*:\s*", re.IGNORECASE),
    re.compile(r"<\|im_start\|>", re.IGNORECASE),
    re.compile(r"<\|system\|>", re.IGNORECASE),
]

# Escape braces to prevent .format() crashes
_FORMAT_ESCAPE_PATTERN: Final[re.Pattern[str]] = re.compile(r"\{(?!\w+\}|\{)")


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def sanitize_brief(brief: str) -> str:
    """Validate and sanitize a campaign brief.

    Args:
        brief: Raw user input.

    Returns:
        Cleaned brief string.

    Raises:
        ValidationError: If the brief violates length or safety rules.
    """
    if not brief or not brief.strip():
        raise ValidationError("Brief cannot be empty.")

    if len(brief) > MAX_BRIEF_LENGTH:
        raise ValidationError(
            f"Brief exceeds maximum length of {MAX_BRIEF_LENGTH} characters "
            f"(received {len(brief)})."
        )

    # Strip control characters
    cleaned = _CONTROL_CHAR_PATTERN.sub("", brief)

    # Check for obvious injection patterns
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(cleaned):
            raise ValidationError(
                "Brief contains potentially unsafe content. "
                "Please rephrase without instructions directed at the AI."
            )

    # Escape unmatched braces to prevent .format() KeyError/ValueError
    cleaned = _FORMAT_ESCAPE_PATTERN.sub("{{", cleaned)
    cleaned = cleaned.replace("}", "}}") if "}" in cleaned and "{{" not in cleaned else cleaned

    return cleaned.strip()


def sanitize_query(query: str) -> str:
    """Validate and sanitize a search query.

    Args:
        query: Raw user query.

    Returns:
        Cleaned query string.

    Raises:
        ValidationError: If the query is empty or too long.
    """
    if not query or not query.strip():
        raise ValidationError("Query cannot be empty.")

    if len(query) > MAX_QUERY_LENGTH:
        raise ValidationError(
            f"Query exceeds maximum length of {MAX_QUERY_LENGTH} characters "
            f"(received {len(query)})."
        )

    cleaned = _CONTROL_CHAR_PATTERN.sub("", query)
    return cleaned.strip()


def validate_file_size(size_bytes: int) -> None:
    """Ensure a file is within acceptable size limits.

    Args:
        size_bytes: File size in bytes.

    Raises:
        ValidationError: If the file is too large.
    """
    if size_bytes > MAX_FILE_SIZE_BYTES:
        raise ValidationError(
            f"File size {size_bytes / (1024 * 1024):.1f} MB exceeds "
            f"maximum of {MAX_FILE_SIZE_MB} MB."
        )
