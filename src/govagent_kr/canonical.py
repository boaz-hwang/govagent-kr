import hashlib
import json
import unicodedata
from typing import Any

from .constants import CANONICALIZATION_VERSION


class CanonicalizationError(ValueError):
    """Raised when text cannot be represented by quote-c14n/v1."""


def _reject_invalid_scalar_text(value: str) -> None:
    if "\x00" in value:
        raise CanonicalizationError("NUL is not allowed")
    for char in value:
        if 0xD800 <= ord(char) <= 0xDFFF:
            raise CanonicalizationError("unpaired surrogate is not allowed")


def canonicalize_quote(value: str) -> str:
    """Apply the deliberately narrow quote-c14n/v1 transform."""

    if not isinstance(value, str):
        raise CanonicalizationError("quote must be a string")
    _reject_invalid_scalar_text(value)
    normalized = value.replace("\r\n", "\n").replace("\r", "\n")
    normalized = unicodedata.normalize("NFC", normalized)
    normalized = "\n".join(line.rstrip(" \t") for line in normalized.split("\n"))
    if normalized == "":
        raise CanonicalizationError("canonical quote must not be empty")
    return normalized


def quote_sha256(value: str) -> str:
    return hashlib.sha256(canonicalize_quote(value).encode("utf-8")).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    """Stable JSON encoding used for replay receipts and hashes."""

    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def content_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def controlling_event_hash(value: dict[str, Any]) -> str:
    return content_sha256(canonical_json_bytes(value))


__all__ = [
    "CANONICALIZATION_VERSION",
    "CanonicalizationError",
    "canonical_json_bytes",
    "canonicalize_quote",
    "content_sha256",
    "controlling_event_hash",
    "quote_sha256",
]
