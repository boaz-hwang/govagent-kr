from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from .canonical import CanonicalizationError, canonical_json_bytes, quote_sha256
from .fixtures import FixtureStore
from .models import (
    AnswerStatus,
    EpisodeOutcome,
    GoldRecord,
    GradeMetrics,
    GradeResult,
    Submission,
)
from .strict_json import StrictJSONError, parse_submission

FAILURE_CODE_ORDER = (
    "SCHEMA_INVALID",
    "STATUS_MISMATCH",
    "CONCLUSION_MISMATCH",
    "MISSING_FACTS_MISMATCH",
    "DUPLICATE_CITATION",
    "REQUIRED_CITATION_MISSING",
    "UNEXPECTED_CITATION",
    "FORBIDDEN_HISTORY",
    "QUOTE_HASH_MISMATCH",
    "UNKNOWN_SOURCE",
    "SUT_BUDGET_EXHAUSTED",
    "SUT_NO_SUBMISSION",
)


def _ordered_failure_codes(codes: Iterable[str]) -> list[str]:
    unique = set(codes)
    unknown = unique - set(FAILURE_CODE_ORDER)
    if unknown:
        raise ValueError(f"unknown failure codes: {sorted(unknown)}")
    return [code for code in FAILURE_CODE_ORDER if code in unique]


def _empty_metrics() -> GradeMetrics:
    return GradeMetrics(
        required_provision_recall=None,
        selected_provision_precision=None,
        temporal_history_accuracy=None,
        quote_integrity=None,
        forbidden_current_version_rate=None,
        correct_abstention=None,
    )


def _terminal_outcome(outcome: EpisodeOutcome) -> GradeResult | None:
    infrastructure = {
        EpisodeOutcome.PROVIDER_ERROR,
        EpisodeOutcome.TOOL_OR_ENV_ERROR,
        EpisodeOutcome.SANDBOX_ERROR,
        EpisodeOutcome.GRADER_ERROR,
    }
    if outcome in infrastructure:
        return GradeResult(
            episode_outcome=outcome,
            analysis_eligible=False,
            core_success=None,
            attempt_success=0,
            failure_codes=[],
            metrics=_empty_metrics(),
            environment_valid=False,
            selected_tuples=[],
        )
    if outcome == EpisodeOutcome.SUT_BUDGET_EXHAUSTED:
        return GradeResult(
            episode_outcome=outcome,
            analysis_eligible=True,
            core_success=0,
            attempt_success=0,
            failure_codes=["SUT_BUDGET_EXHAUSTED"],
            metrics=_empty_metrics(),
            environment_valid=True,
            selected_tuples=[],
        )
    if outcome == EpisodeOutcome.SUT_NO_SUBMISSION:
        return GradeResult(
            episode_outcome=outcome,
            analysis_eligible=True,
            core_success=0,
            attempt_success=0,
            failure_codes=["SUT_NO_SUBMISSION"],
            metrics=_empty_metrics(),
            environment_valid=True,
            selected_tuples=[],
        )
    return None


def _schema_failure(outcome: EpisodeOutcome = EpisodeOutcome.COMPLETED) -> GradeResult:
    return GradeResult(
        episode_outcome=outcome,
        analysis_eligible=True,
        core_success=0,
        attempt_success=0,
        failure_codes=["SCHEMA_INVALID"],
        metrics=_empty_metrics(),
        environment_valid=True,
        selected_tuples=[],
    )


def grade_submission(
    raw_submission: str | bytes,
    gold: GoldRecord,
    store: FixtureStore,
    *,
    episode_outcome: EpisodeOutcome = EpisodeOutcome.COMPLETED,
) -> GradeResult:
    terminal = _terminal_outcome(episode_outcome)
    if terminal is not None:
        return terminal
    if episode_outcome != EpisodeOutcome.COMPLETED:
        raise ValueError(f"unsupported episode outcome: {episode_outcome}")
    try:
        submission = parse_submission(raw_submission)
    except StrictJSONError:
        return _schema_failure()
    return grade_parsed_submission(submission, gold, store)


def grade_parsed_submission(
    submission: Submission,
    gold: GoldRecord,
    store: FixtureStore,
) -> GradeResult:
    failures: set[str] = set()
    if submission.status != gold.expected_status:
        failures.add("STATUS_MISMATCH")
    if submission.conclusion != gold.expected_conclusion:
        failures.add("CONCLUSION_MISMATCH")

    expected_missing = set(gold.required_missing_fact_keys)
    allowed_missing = set(gold.allowed_missing_fact_keys)
    submitted_missing = set(submission.missing_facts)
    if submission.status == AnswerStatus.INSUFFICIENT_FACTS and (
        not expected_missing.issubset(submitted_missing)
        or not submitted_missing.issubset(expected_missing | allowed_missing)
    ):
        failures.add("MISSING_FACTS_MISMATCH")

    required = {citation.tuple4() for citation in gold.required_citations}
    allowed = {citation.tuple4() for citation in gold.allowed_citations}
    accepted = required | allowed
    accepted_identities = {item[:3] for item in accepted}

    selected_tuples: list[tuple[str, str, str, str]] = []
    selected_identities: list[tuple[str, str, str]] = []
    quote_integrity_count = 0
    unknown_source = False
    quote_mismatch = False
    try:
        for selected in submission.selected_provisions:
            identity = selected.triple()
            passage_hash = quote_sha256(selected.quote)
            tuple4 = (*identity, passage_hash)
            selected_identities.append(identity)
            selected_tuples.append(tuple4)
            provision = store.maybe_get_provision(selected.history_id, selected.provision_locator)
            if provision is None or provision["instrument_id"] != selected.instrument_id:
                unknown_source = True
            elif provision["exact_passage_hash"] == passage_hash:
                quote_integrity_count += 1
            if identity in accepted_identities and tuple4 not in accepted:
                quote_mismatch = True
    except CanonicalizationError:
        return _schema_failure()

    if len(selected_identities) != len(set(selected_identities)) or len(selected_tuples) != len(
        set(selected_tuples)
    ):
        failures.add("DUPLICATE_CITATION")
    selected = set(selected_tuples)
    if not required.issubset(selected):
        failures.add("REQUIRED_CITATION_MISSING")
    if not selected.issubset(accepted):
        failures.add("UNEXPECTED_CITATION")
    if any(identity[1] in set(gold.forbidden_history_ids) for identity in selected_identities):
        failures.add("FORBIDDEN_HISTORY")
    if quote_mismatch:
        failures.add("QUOTE_HASH_MISMATCH")
    if unknown_source:
        failures.add("UNKNOWN_SOURCE")

    selected_count = len(selected_tuples)
    required_recall = len(required & selected) / len(required) if required else None
    selected_precision = len(selected & accepted) / selected_count if selected_count else None
    temporal_accuracy = (
        sum(identity in accepted_identities for identity in selected_identities) / selected_count
        if selected_count
        else None
    )
    quote_integrity = quote_integrity_count / selected_count if selected_count else None
    forbidden_rate = (
        sum(identity[1] in set(gold.forbidden_history_ids) for identity in selected_identities)
        / selected_count
        if selected_count
        else None
    )
    abstention_expected = gold.expected_status != AnswerStatus.ANSWERED
    correct_abstention = None
    if abstention_expected:
        correct_abstention = float(
            submission.status == gold.expected_status
            and not submission.selected_provisions
            and "MISSING_FACTS_MISMATCH" not in failures
        )
    failure_codes = _ordered_failure_codes(failures)
    core_success = 0 if failure_codes else 1
    return GradeResult(
        episode_outcome=EpisodeOutcome.COMPLETED,
        analysis_eligible=True,
        core_success=core_success,
        attempt_success=core_success,
        failure_codes=failure_codes,
        metrics=GradeMetrics(
            required_provision_recall=required_recall,
            selected_provision_precision=selected_precision,
            temporal_history_accuracy=temporal_accuracy,
            quote_integrity=quote_integrity,
            forbidden_current_version_rate=forbidden_rate,
            correct_abstention=correct_abstention,
        ),
        environment_valid=True,
        selected_tuples=sorted(selected_tuples),
    )


def gold_submission(gold: GoldRecord, store: FixtureStore) -> bytes:
    if gold.expected_status == AnswerStatus.ANSWERED:
        selected = []
        for citation in gold.required_citations:
            provision = store.get_provision(citation.history_id, citation.provision_locator)
            selected.append(
                {
                    "instrument_id": citation.instrument_id,
                    "history_id": citation.history_id,
                    "provision_locator": citation.provision_locator,
                    "quote": provision["passage"],
                }
            )
        missing_facts: list[str] = []
    elif gold.expected_status == AnswerStatus.INSUFFICIENT_FACTS:
        selected = []
        missing_facts = list(gold.required_missing_fact_keys)
    else:
        selected = []
        missing_facts = []
    return canonical_json_bytes(
        {
            "status": gold.expected_status.value,
            "conclusion": gold.expected_conclusion.value,
            "selected_provisions": selected,
            "missing_facts": missing_facts,
            "explanation": "fixture oracle replay",
        }
    )


def current_version_ablation_submission(gold: GoldRecord, store: FixtureStore) -> bytes:
    if gold.expected_status != AnswerStatus.ANSWERED:
        return gold_submission(gold, store)
    selected: list[dict[str, Any]] = []
    for citation in gold.required_citations:
        current = store.maybe_get_provision(store.latest_history_id, citation.provision_locator)
        if current is None:
            original = store.get_provision(citation.history_id, citation.provision_locator)
            history_id = citation.history_id
            passage = original["passage"]
        else:
            history_id = store.latest_history_id
            passage = current["passage"]
        selected.append(
            {
                "instrument_id": citation.instrument_id,
                "history_id": history_id,
                "provision_locator": citation.provision_locator,
                "quote": passage,
            }
        )
    return canonical_json_bytes(
        {
            "status": gold.expected_status.value,
            "conclusion": gold.expected_conclusion.value,
            "selected_provisions": selected,
            "missing_facts": [],
            "explanation": "current-version ablation diagnostic",
        }
    )


def replay_grade_bytes(
    raw_submission: str | bytes,
    gold: GoldRecord,
    store: FixtureStore,
    *,
    episode_outcome: EpisodeOutcome = EpisodeOutcome.COMPLETED,
) -> bytes:
    result = grade_submission(
        raw_submission,
        gold,
        store,
        episode_outcome=episode_outcome,
    )
    return canonical_json_bytes(result.model_dump(mode="json"))


def gold_from_json(raw: str | bytes) -> GoldRecord:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return GoldRecord.model_validate(json.loads(raw), strict=True)


__all__ = [
    "FAILURE_CODE_ORDER",
    "current_version_ablation_submission",
    "gold_submission",
    "grade_parsed_submission",
    "grade_submission",
    "replay_grade_bytes",
]
