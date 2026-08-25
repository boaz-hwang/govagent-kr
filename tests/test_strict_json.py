import json

import pytest

from govagent_kr.models import AnswerStatus
from govagent_kr.strict_json import (
    StrictJSONError,
    parse_strict_json_object,
    parse_submission,
)


def _answer_payload() -> dict[str, object]:
    return {
        "status": "answered",
        "conclusion": "affirmative",
        "selected_provisions": [
            {
                "instrument_id": "KR-LAW-010098",
                "history_id": "KR-LAW-010098:204108:20180724",
                "provision_locator": "제25조제1항제5호마목",
                "quote": "원문",
            }
        ],
        "missing_facts": [],
        "explanation": "설명",
    }


def _dump(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False)


def test_valid_json_status_string_is_parsed_without_weakening_strict_model() -> None:
    parsed = parse_submission(_dump(_answer_payload()))

    assert parsed.status is AnswerStatus.ANSWERED
    assert parsed.selected_provisions[0].instrument_id == "KR-LAW-010098"


@pytest.mark.parametrize("bad_value", [1, True, 1.5, None])
def test_scalar_types_are_not_coerced_to_strings(bad_value: object) -> None:
    payload = _answer_payload()
    payload["selected_provisions"][0]["instrument_id"] = bad_value  # type: ignore[index]

    with pytest.raises(StrictJSONError, match="schema is invalid"):
        parse_submission(_dump(payload))


@pytest.mark.parametrize(
    "raw",
    [
        "```json\n{}\n```",
        "{} trailing text",
        "{}{}",
        "[]",
        "NaN",
        '{"value": NaN}',
        '{"value": Infinity}',
    ],
)
def test_non_single_object_or_non_finite_json_is_rejected(raw: str) -> None:
    with pytest.raises(StrictJSONError):
        parse_strict_json_object(raw)


@pytest.mark.parametrize(
    "raw",
    [
        '{"status":"answered","status":"insufficient_facts"}',
        '{"outer":{"history_id":"a","history_id":"b"}}',
    ],
)
def test_duplicate_keys_at_any_depth_are_rejected(raw: str) -> None:
    with pytest.raises(StrictJSONError, match="duplicate JSON object key"):
        parse_strict_json_object(raw)


def test_invalid_utf8_bytes_and_surrogate_escape_are_rejected() -> None:
    with pytest.raises(StrictJSONError, match="not UTF-8"):
        parse_strict_json_object(b"\xff")
    with pytest.raises(StrictJSONError, match="unpaired surrogate"):
        parse_strict_json_object(r'{"value":"\ud800"}')


def test_byte_limit_is_applied_to_utf8_not_character_count() -> None:
    raw = '{"값":"가"}'

    with pytest.raises(StrictJSONError, match="exceeds"):
        parse_strict_json_object(raw, max_bytes=len(raw))


@pytest.mark.parametrize("location", ["top", "citation"])
def test_extra_fields_are_forbidden(location: str) -> None:
    payload = _answer_payload()
    if location == "top":
        payload["reward_me"] = True
    else:
        payload["selected_provisions"][0]["exact_passage_hash"] = "0" * 64  # type: ignore[index]

    with pytest.raises(StrictJSONError, match="schema is invalid"):
        parse_submission(_dump(payload))


@pytest.mark.parametrize(
    ("status", "citations", "missing_facts"),
    [
        ("answered", [], []),
        ("answered", _answer_payload()["selected_provisions"], ["price"]),
        ("insufficient_facts", _answer_payload()["selected_provisions"], ["price"]),
        ("insufficient_facts", [], []),
        ("no_applicable_provision", _answer_payload()["selected_provisions"], []),
        ("no_applicable_provision", [], ["price"]),
    ],
)
def test_status_dependent_shape_is_enforced(
    status: str, citations: object, missing_facts: list[str]
) -> None:
    payload = _answer_payload()
    payload["status"] = status
    payload["selected_provisions"] = citations
    payload["missing_facts"] = missing_facts

    with pytest.raises(StrictJSONError, match="schema is invalid"):
        parse_submission(_dump(payload))


@pytest.mark.parametrize("status", ["ANSWERED", "unknown", "", 1, True, None])
def test_only_exact_status_enum_strings_are_accepted(status: object) -> None:
    payload = _answer_payload()
    payload["status"] = status

    with pytest.raises(StrictJSONError, match="schema is invalid"):
        parse_submission(_dump(payload))


def test_missing_fact_keys_must_be_nonempty_and_unique() -> None:
    payload = _answer_payload()
    payload.update(
        status="insufficient_facts",
        selected_provisions=[],
        missing_facts=["price", "price"],
    )

    with pytest.raises(StrictJSONError, match="schema is invalid"):
        parse_submission(_dump(payload))
