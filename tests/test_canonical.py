import hashlib
import unicodedata

import pytest

from govagent_kr.canonical import (
    CanonicalizationError,
    canonical_json_bytes,
    canonicalize_quote,
    quote_sha256,
)


def test_canonicalization_applies_only_documented_equivalences() -> None:
    composed = "제25조\n사회적기업"
    decomposed = unicodedata.normalize("NFD", composed)

    variants = [
        composed,
        "제25조\r\n사회적기업",
        "제25조\r사회적기업",
        "제25조 \t\n사회적기업\t",
        decomposed,
    ]

    assert all(canonicalize_quote(value) == composed for value in variants)
    assert len({quote_sha256(value) for value in variants}) == 1


@pytest.mark.parametrize(
    "changed",
    [
        " 제25조\n사회적기업",  # leading space
        "제25조\n사회적 기업",  # internal space
        "제25조.\n사회적기업",  # punctuation
        "제26조\n사회적기업",  # number
        "제25조 사회적기업",  # newline removed
        "제25조\n사회적기업\n",  # newline added
        "제25조\n사회적기업\u00a0",  # NBSP is not ASCII line-end whitespace
        "제25조\n사회적\u200b기업",  # zero-width space
        "제２５조\n사회적기업",  # NFKC-only compatibility change
        "사회적기업",  # partial quote
        "설명: 제25조\n사회적기업",  # prefix added
    ],
)
def test_semantic_or_compatibility_changes_do_not_canonicalize_away(changed: str) -> None:
    original = "제25조\n사회적기업"

    assert canonicalize_quote(changed) != original
    assert quote_sha256(changed) != quote_sha256(original)


@pytest.mark.parametrize("invalid", ["", " \t", "\x00", "본문\x00", "\ud800"])
def test_invalid_quote_text_is_rejected(invalid: str) -> None:
    with pytest.raises(CanonicalizationError):
        canonicalize_quote(invalid)


def test_non_string_quote_is_rejected_without_coercion() -> None:
    with pytest.raises(CanonicalizationError, match="must be a string"):
        canonicalize_quote(25)  # type: ignore[arg-type]


def test_canonicalization_is_idempotent_and_hashes_utf8_bytes() -> None:
    raw = "사회적기업 \t\r\n제25조\t"
    canonical = canonicalize_quote(raw)

    assert canonicalize_quote(canonical) == canonical
    assert quote_sha256(raw) == hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def test_canonical_json_bytes_is_stable_and_does_not_escape_hangul() -> None:
    assert canonical_json_bytes({"나": 2, "가": 1}) == '{"가":1,"나":2}\n'.encode()
