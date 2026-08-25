import json
import unicodedata

import pytest

from govagent_kr.canonical import quote_sha256
from govagent_kr.dataset import load_gold
from govagent_kr.fixtures import FixtureStore
from govagent_kr.models import EpisodeOutcome, GoldCitation, GoldRecord
from govagent_kr.scorer import (
    current_version_ablation_submission,
    gold_submission,
    grade_submission,
    replay_grade_bytes,
)


@pytest.fixture(scope="module")
def store() -> FixtureStore:
    return FixtureStore()


@pytest.fixture(scope="module")
def gold_records() -> list[GoldRecord]:
    return load_gold()


@pytest.fixture(scope="module")
def single_answer_gold(gold_records: list[GoldRecord]) -> GoldRecord:
    return next(record for record in gold_records if len(record.required_citations) == 1)


@pytest.fixture(scope="module")
def multiple_answer_gold(gold_records: list[GoldRecord]) -> GoldRecord:
    return next(record for record in gold_records if len(record.required_citations) > 1)


def _payload(raw: bytes) -> dict[str, object]:
    return json.loads(raw)


def _raw(payload: dict[str, object]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()


def _optional_citation(gold: GoldRecord, store: FixtureStore) -> tuple[GoldCitation, dict]:
    required = gold.required_citations[0]
    for provision in store.iter_provisions():
        identity = (
            provision["instrument_id"],
            provision["history_id"],
            provision["provision_locator"],
        )
        same_version = identity[:2] == (required.instrument_id, required.history_id)
        if same_version and identity != required.triple():
            citation = GoldCitation(
                instrument_id=provision["instrument_id"],
                history_id=provision["history_id"],
                provision_locator=provision["provision_locator"],
                exact_passage_hash=provision["exact_passage_hash"],
            )
            selected = {
                "instrument_id": provision["instrument_id"],
                "history_id": provision["history_id"],
                "provision_locator": provision["provision_locator"],
                "quote": provision["passage"],
            }
            return citation, selected
    raise AssertionError("fixture needs another provision in the required history")


def test_fixture_oracle_passes_all_twenty_items(
    gold_records: list[GoldRecord], store: FixtureStore
) -> None:
    results = [grade_submission(gold_submission(gold, store), gold, store) for gold in gold_records]

    assert len(results) == 20
    assert all(result.core_success == 1 for result in results)
    assert all(result.failure_codes == [] for result in results)


def test_current_version_ablation_exposes_at_least_three_temporal_traps(
    gold_records: list[GoldRecord], store: FixtureStore
) -> None:
    results = [
        grade_submission(current_version_ablation_submission(gold, store), gold, store)
        for gold in gold_records
    ]

    assert sum(result.core_success == 0 for result in results) >= 3


def test_required_and_explicitly_allowed_citations_pass(
    single_answer_gold: GoldRecord, store: FixtureStore
) -> None:
    optional, optional_selected = _optional_citation(single_answer_gold, store)
    gold_with_allowed = single_answer_gold.model_copy(update={"allowed_citations": [optional]})
    payload = _payload(gold_submission(single_answer_gold, store))
    payload["selected_provisions"].append(optional_selected)  # type: ignore[union-attr]

    result = grade_submission(_raw(payload), gold_with_allowed, store)

    assert result.core_success == 1
    assert result.metrics.required_provision_recall == 1.0
    assert result.metrics.selected_provision_precision == 1.0


def test_allowed_only_does_not_replace_a_required_citation(
    single_answer_gold: GoldRecord, store: FixtureStore
) -> None:
    optional, optional_selected = _optional_citation(single_answer_gold, store)
    gold_with_allowed = single_answer_gold.model_copy(update={"allowed_citations": [optional]})
    payload = _payload(gold_submission(single_answer_gold, store))
    payload["selected_provisions"] = [optional_selected]

    result = grade_submission(_raw(payload), gold_with_allowed, store)

    assert result.core_success == 0
    assert result.failure_codes == ["REQUIRED_CITATION_MISSING"]


def test_unlisted_overcitation_turns_a_correct_answer_into_failure(
    single_answer_gold: GoldRecord, store: FixtureStore
) -> None:
    _, extra = _optional_citation(single_answer_gold, store)
    payload = _payload(gold_submission(single_answer_gold, store))
    payload["selected_provisions"].append(extra)  # type: ignore[union-attr]

    result = grade_submission(_raw(payload), single_answer_gold, store)

    assert result.core_success == 0
    assert "UNEXPECTED_CITATION" in result.failure_codes
    assert "REQUIRED_CITATION_MISSING" not in result.failure_codes


def test_duplicate_and_canonically_equivalent_duplicate_citations_fail(
    single_answer_gold: GoldRecord, store: FixtureStore
) -> None:
    payload = _payload(gold_submission(single_answer_gold, store))
    selected = payload["selected_provisions"][0]  # type: ignore[index]
    duplicate = dict(selected)
    duplicate["quote"] = unicodedata.normalize("NFD", duplicate["quote"])
    assert quote_sha256(selected["quote"]) == quote_sha256(duplicate["quote"])
    payload["selected_provisions"].append(duplicate)  # type: ignore[union-attr]

    result = grade_submission(_raw(payload), single_answer_gold, store)

    assert result.core_success == 0
    assert result.failure_codes == ["DUPLICATE_CITATION"]


def test_same_identity_with_two_different_quotes_is_duplicate_and_quote_mismatch(
    single_answer_gold: GoldRecord, store: FixtureStore
) -> None:
    payload = _payload(gold_submission(single_answer_gold, store))
    altered = dict(payload["selected_provisions"][0])  # type: ignore[index]
    altered["quote"] += "."
    payload["selected_provisions"].append(altered)  # type: ignore[union-attr]

    result = grade_submission(_raw(payload), single_answer_gold, store)

    assert result.core_success == 0
    assert "DUPLICATE_CITATION" in result.failure_codes
    assert "UNEXPECTED_CITATION" in result.failure_codes
    assert "QUOTE_HASH_MISMATCH" in result.failure_codes


def test_partial_or_modified_quote_fails_full_passage_hash(
    single_answer_gold: GoldRecord, store: FixtureStore
) -> None:
    payload = _payload(gold_submission(single_answer_gold, store))
    selected = payload["selected_provisions"][0]  # type: ignore[index]
    selected["quote"] = selected["quote"][:-1]

    result = grade_submission(_raw(payload), single_answer_gold, store)

    assert result.core_success == 0
    assert "REQUIRED_CITATION_MISSING" in result.failure_codes
    assert "UNEXPECTED_CITATION" in result.failure_codes
    assert "QUOTE_HASH_MISMATCH" in result.failure_codes


def test_documented_unicode_and_line_end_normalization_still_passes(
    single_answer_gold: GoldRecord, store: FixtureStore
) -> None:
    payload = _payload(gold_submission(single_answer_gold, store))
    selected = payload["selected_provisions"][0]  # type: ignore[index]
    quote = unicodedata.normalize("NFD", selected["quote"])
    selected["quote"] = quote.replace("\n", " \t\r\n") + " \t"

    result = grade_submission(_raw(payload), single_answer_gold, store)

    assert result.core_success == 1
    assert result.metrics.quote_integrity == 1.0


def test_empty_after_canonicalization_is_schema_failure(
    single_answer_gold: GoldRecord, store: FixtureStore
) -> None:
    payload = _payload(gold_submission(single_answer_gold, store))
    payload["selected_provisions"][0]["quote"] = " \t"  # type: ignore[index]

    result = grade_submission(_raw(payload), single_answer_gold, store)

    assert result.core_success == 0
    assert result.failure_codes == ["SCHEMA_INVALID"]


def test_exact_passage_from_wrong_history_does_not_cross_match(
    gold_records: list[GoldRecord], store: FixtureStore
) -> None:
    provisions = list(store.iter_provisions())
    match = next(
        (gold, citation, wrong)
        for gold in gold_records
        for citation in gold.required_citations
        for wrong in provisions
        if wrong["history_id"] != citation.history_id
        and wrong["provision_locator"] == citation.provision_locator
        and wrong["exact_passage_hash"] == citation.exact_passage_hash
    )
    gold, citation, wrong = match
    payload = _payload(gold_submission(gold, store))
    selected = payload["selected_provisions"]
    target_index = next(
        index
        for index, item in enumerate(selected)  # type: ignore[arg-type]
        if item["history_id"] == citation.history_id  # type: ignore[index]
        and item["provision_locator"] == citation.provision_locator  # type: ignore[index]
    )
    selected[target_index] = {  # type: ignore[index]
        "instrument_id": wrong["instrument_id"],
        "history_id": wrong["history_id"],
        "provision_locator": wrong["provision_locator"],
        "quote": wrong["passage"],
    }

    result = grade_submission(_raw(payload), gold, store)

    assert result.core_success == 0
    assert "REQUIRED_CITATION_MISSING" in result.failure_codes
    assert "UNEXPECTED_CITATION" in result.failure_codes
    assert result.metrics.quote_integrity == 1.0


def test_forbidden_current_history_is_reported(
    single_answer_gold: GoldRecord, store: FixtureStore
) -> None:
    current = next(
        provision
        for provision in store.iter_provisions()
        if provision["history_id"] == store.latest_history_id
    )
    payload = _payload(gold_submission(single_answer_gold, store))
    payload["selected_provisions"].append(  # type: ignore[union-attr]
        {
            "instrument_id": current["instrument_id"],
            "history_id": current["history_id"],
            "provision_locator": current["provision_locator"],
            "quote": current["passage"],
        }
    )

    result = grade_submission(_raw(payload), single_answer_gold, store)

    assert result.core_success == 0
    assert "FORBIDDEN_HISTORY" in result.failure_codes
    assert "UNEXPECTED_CITATION" in result.failure_codes


def test_unknown_source_is_a_scored_failure_not_an_infrastructure_error(
    single_answer_gold: GoldRecord, store: FixtureStore
) -> None:
    payload = _payload(gold_submission(single_answer_gold, store))
    payload["selected_provisions"].append(  # type: ignore[union-attr]
        {
            "instrument_id": "KR-LAW-DOES-NOT-EXIST",
            "history_id": "missing-history",
            "provision_locator": "제999조",
            "quote": "존재하지 않는 원문",
        }
    )

    result = grade_submission(_raw(payload), single_answer_gold, store)

    assert result.episode_outcome is EpisodeOutcome.COMPLETED
    assert result.analysis_eligible is True
    assert result.environment_valid is True
    assert result.core_success == 0
    assert "UNKNOWN_SOURCE" in result.failure_codes


def test_correct_and_incorrect_insufficient_fact_abstention(
    gold_records: list[GoldRecord], store: FixtureStore
) -> None:
    gold = next(record for record in gold_records if record.expected_status == "insufficient_facts")
    correct_payload = _payload(gold_submission(gold, store))
    wrong_payload = {**correct_payload, "missing_facts": ["unrelated_fact"]}

    correct = grade_submission(_raw(correct_payload), gold, store)
    wrong = grade_submission(_raw(wrong_payload), gold, store)

    assert correct.core_success == 1
    assert correct.metrics.correct_abstention == 1.0
    assert wrong.core_success == 0
    assert wrong.failure_codes == ["MISSING_FACTS_MISMATCH"]
    assert wrong.metrics.correct_abstention == 0.0


def test_blanket_abstention_does_not_pass_an_answered_item(
    single_answer_gold: GoldRecord, store: FixtureStore
) -> None:
    payload = {
        "status": "insufficient_facts",
        "conclusion": "indeterminate",
        "selected_provisions": [],
        "missing_facts": ["unrelated_fact"],
        "explanation": "기권",
    }

    result = grade_submission(_raw(payload), single_answer_gold, store)

    assert result.core_success == 0
    assert "STATUS_MISMATCH" in result.failure_codes


def test_opposite_structured_conclusion_fails_even_with_exact_citations(
    single_answer_gold: GoldRecord, store: FixtureStore
) -> None:
    payload = _payload(gold_submission(single_answer_gold, store))
    payload["conclusion"] = (
        "negative" if single_answer_gold.expected_conclusion == "affirmative" else "affirmative"
    )

    result = grade_submission(_raw(payload), single_answer_gold, store)

    assert result.core_success == 0
    assert result.failure_codes == ["CONCLUSION_MISMATCH"]


def test_no_applicable_provision_submission_can_pass_without_citations(
    gold_records: list[GoldRecord], store: FixtureStore
) -> None:
    gold = next(
        record for record in gold_records if record.expected_status == "no_applicable_provision"
    )

    result = grade_submission(gold_submission(gold, store), gold, store)

    assert result.core_success == 1
    assert result.metrics.correct_abstention == 1.0
    assert result.metrics.required_provision_recall is None
    assert result.metrics.selected_provision_precision is None


@pytest.mark.parametrize(
    "raw",
    [
        b"not json",
        b"```json\n{}\n```",
        b'{"status":"answered","status":"insufficient_facts"}',
        b'{"status":"answered","selected_provisions":[],"missing_facts":[],"explanation":""}',
    ],
)
def test_malformed_or_invalid_submission_is_deterministic_schema_failure(
    raw: bytes, single_answer_gold: GoldRecord, store: FixtureStore
) -> None:
    result = grade_submission(raw, single_answer_gold, store)

    assert result.episode_outcome is EpisodeOutcome.COMPLETED
    assert result.analysis_eligible is True
    assert result.core_success == 0
    assert result.failure_codes == ["SCHEMA_INVALID"]


@pytest.mark.parametrize(
    "outcome",
    [
        EpisodeOutcome.PROVIDER_ERROR,
        EpisodeOutcome.TOOL_OR_ENV_ERROR,
        EpisodeOutcome.SANDBOX_ERROR,
        EpisodeOutcome.GRADER_ERROR,
    ],
)
def test_infrastructure_and_environment_outcomes_are_not_graded(
    outcome: EpisodeOutcome, single_answer_gold: GoldRecord, store: FixtureStore
) -> None:
    result = grade_submission(b"not json", single_answer_gold, store, episode_outcome=outcome)

    assert result.episode_outcome is outcome
    assert result.analysis_eligible is False
    assert result.environment_valid is False
    assert result.core_success is None
    assert result.attempt_success == 0
    assert result.failure_codes == []


@pytest.mark.parametrize(
    ("outcome", "failure_code"),
    [
        (EpisodeOutcome.SUT_BUDGET_EXHAUSTED, "SUT_BUDGET_EXHAUSTED"),
        (EpisodeOutcome.SUT_NO_SUBMISSION, "SUT_NO_SUBMISSION"),
    ],
)
def test_sut_noncompletion_is_counted_as_attempt_failure(
    outcome: EpisodeOutcome,
    failure_code: str,
    single_answer_gold: GoldRecord,
    store: FixtureStore,
) -> None:
    result = grade_submission(b"not json", single_answer_gold, store, episode_outcome=outcome)

    assert result.analysis_eligible is True
    assert result.environment_valid is True
    assert result.core_success == 0
    assert result.attempt_success == 0
    assert result.failure_codes == [failure_code]


def test_replay_bytes_ignore_citation_order_and_are_stable(
    multiple_answer_gold: GoldRecord, store: FixtureStore
) -> None:
    first = _payload(gold_submission(multiple_answer_gold, store))
    second = dict(first)
    second["selected_provisions"] = list(reversed(first["selected_provisions"]))  # type: ignore[arg-type]

    first_grade = replay_grade_bytes(_raw(first), multiple_answer_gold, store)
    second_grade = replay_grade_bytes(_raw(second), multiple_answer_gold, store)

    assert first_grade == second_grade
    assert first_grade == replay_grade_bytes(_raw(first), multiple_answer_gold, store)
