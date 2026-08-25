from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from .constants import MAX_SUBMISSION_BYTES
from .models import Submission


class StrictJSONError(ValueError):
    """A deterministic, user-caused submission parsing failure."""


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise StrictJSONError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise StrictJSONError(f"non-finite JSON number is not allowed: {value}")


def _validate_scalar_text(value: Any) -> None:
    if isinstance(value, str):
        if "\x00" in value:
            raise StrictJSONError("NUL is not allowed")
        if any(0xD800 <= ord(char) <= 0xDFFF for char in value):
            raise StrictJSONError("unpaired surrogate is not allowed")
    elif isinstance(value, list):
        for child in value:
            _validate_scalar_text(child)
    elif isinstance(value, dict):
        for key, child in value.items():
            _validate_scalar_text(key)
            _validate_scalar_text(child)


def parse_strict_json_object(raw: str | bytes, *, max_bytes: int = MAX_SUBMISSION_BYTES) -> dict:
    if isinstance(raw, str):
        try:
            encoded = raw.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise StrictJSONError("submission is not valid Unicode scalar text") from exc
        text = raw
    elif isinstance(raw, bytes):
        encoded = raw
        try:
            text = raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise StrictJSONError("submission bytes are not UTF-8") from exc
    else:
        raise StrictJSONError("submission must be str or bytes")
    if len(encoded) > max_bytes:
        raise StrictJSONError(f"submission exceeds {max_bytes} bytes")
    try:
        value = json.loads(
            text,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except StrictJSONError:
        raise
    except (json.JSONDecodeError, RecursionError) as exc:
        raise StrictJSONError("submission is not one valid JSON value") from exc
    if not isinstance(value, dict):
        raise StrictJSONError("submission root must be a JSON object")
    _validate_scalar_text(value)
    return value


def parse_submission(raw: str | bytes) -> Submission:
    try:
        return Submission.model_validate(parse_strict_json_object(raw), strict=True)
    except ValidationError as exc:
        raise StrictJSONError("submission schema is invalid") from exc


__all__ = ["StrictJSONError", "parse_strict_json_object", "parse_submission"]
