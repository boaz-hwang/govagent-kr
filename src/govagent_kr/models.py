from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .constants import (
    MAX_EXPLANATION_CHARS,
    MAX_MISSING_FACTS,
    MAX_SELECTED_PROVISIONS,
    SCHEMA_VERSION,
    TIMEZONE,
)

NonEmptyStr = Annotated[str, Field(min_length=1)]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class ControllingEvent(StrictModel):
    kind: NonEmptyStr
    date: Annotated[str, Field(pattern=r"^\d{4}-\d{2}-\d{2}$")]
    timezone: Literal["Asia/Seoul"] = TIMEZONE

    @model_validator(mode="after")
    def valid_calendar_date(self) -> ControllingEvent:
        date.fromisoformat(self.date)
        return self


class TaskInput(StrictModel):
    task: Literal["kr-law-temporal-citation"]
    question: NonEmptyStr
    facts: Annotated[list[NonEmptyStr], Field(min_length=1, max_length=32)]
    instrument_id: NonEmptyStr
    controlling_event: ControllingEvent


class Variant(StrEnum):
    PRE_EFFECTIVE = "pre_effective"
    POST_EFFECTIVE = "post_effective"
    CURRENT_DECOY = "current_decoy"
    INSUFFICIENT_FACTS = "insufficient_facts"


class CaseMetadata(StrictModel):
    boundary_id: NonEmptyStr
    variant: Variant
    effective_at: Annotated[str, Field(pattern=r"^\d{4}-\d{2}-\d{2}$")]
    candidate_gold: bool = True


class CaseRecord(StrictModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    item_id: NonEmptyStr
    family_id: NonEmptyStr
    metadata: CaseMetadata
    input: TaskInput


class CitationIdentity(StrictModel):
    instrument_id: NonEmptyStr
    history_id: NonEmptyStr
    provision_locator: NonEmptyStr

    def triple(self) -> tuple[str, str, str]:
        return (self.instrument_id, self.history_id, self.provision_locator)


class GoldCitation(CitationIdentity):
    exact_passage_hash: Sha256

    def tuple4(self) -> tuple[str, str, str, str]:
        return (*self.triple(), self.exact_passage_hash)


class SelectedProvision(CitationIdentity):
    quote: NonEmptyStr


class AnswerStatus(StrEnum):
    ANSWERED = "answered"
    INSUFFICIENT_FACTS = "insufficient_facts"
    NO_APPLICABLE_PROVISION = "no_applicable_provision"


class Conclusion(StrEnum):
    AFFIRMATIVE = "affirmative"
    NEGATIVE = "negative"
    INDETERMINATE = "indeterminate"


class ReviewStatus(StrEnum):
    PENDING = "pending"
    ADJUDICATED = "adjudicated"
    EXCLUDED = "excluded"


class GoldReview(StrictModel):
    status: ReviewStatus = ReviewStatus.PENDING
    reviewer_1: str | None = None
    reviewer_2: str | None = None
    reviewer_1_verdict: Literal["correct", "incorrect", "indeterminate"] | None = None
    reviewer_2_verdict: Literal["correct", "incorrect", "indeterminate"] | None = None
    resolution: str | None = None

    @model_validator(mode="after")
    def adjudicated_has_two_reviewers(self) -> GoldReview:
        if self.status == ReviewStatus.ADJUDICATED:
            values = (
                self.reviewer_1,
                self.reviewer_2,
                self.reviewer_1_verdict,
                self.reviewer_2_verdict,
                self.resolution,
            )
            if any(value is None or not value.strip() for value in values):
                raise ValueError("adjudicated gold requires two reviewer verdicts and resolution")
            assert self.reviewer_1 is not None
            assert self.reviewer_2 is not None
            if self.reviewer_1.strip().casefold() == self.reviewer_2.strip().casefold():
                raise ValueError("gold reviewers must be independent people")
        return self


class GoldRecord(StrictModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    item_id: NonEmptyStr
    expected_status: AnswerStatus
    expected_conclusion: Conclusion
    required_citations: list[GoldCitation]
    allowed_citations: list[GoldCitation]
    forbidden_history_ids: list[NonEmptyStr]
    required_missing_fact_keys: list[NonEmptyStr] = Field(default_factory=list)
    allowed_missing_fact_keys: list[NonEmptyStr] = Field(default_factory=list)
    controlling_event_hash: Sha256
    adjudication_note: str
    review: GoldReview


class Submission(StrictModel):
    status: AnswerStatus
    conclusion: Conclusion
    selected_provisions: Annotated[
        list[SelectedProvision], Field(max_length=MAX_SELECTED_PROVISIONS)
    ]
    missing_facts: Annotated[list[NonEmptyStr], Field(max_length=MAX_MISSING_FACTS)]
    explanation: Annotated[str, Field(max_length=MAX_EXPLANATION_CHARS)]

    @field_validator("status", mode="before")
    @classmethod
    def parse_json_status(cls, value: Any) -> Any:
        """Accept an exact JSON enum string without weakening other strict fields."""

        if type(value) is str:
            try:
                return AnswerStatus(value)
            except ValueError:
                return value
        return value

    @field_validator("conclusion", mode="before")
    @classmethod
    def parse_json_conclusion(cls, value: Any) -> Any:
        """Accept an exact JSON enum string without weakening other strict fields."""

        if type(value) is str:
            try:
                return Conclusion(value)
            except ValueError:
                return value
        return value

    @model_validator(mode="after")
    def enforce_status_shape(self) -> Submission:
        if len(set(self.missing_facts)) != len(self.missing_facts):
            raise ValueError("missing_facts must be unique stable keys")
        if self.status == AnswerStatus.ANSWERED:
            if not self.selected_provisions:
                raise ValueError("answered requires at least one selected provision")
            if self.missing_facts:
                raise ValueError("answered must not include missing facts")
            if self.conclusion == Conclusion.INDETERMINATE:
                raise ValueError("answered requires an affirmative or negative conclusion")
        elif self.status == AnswerStatus.INSUFFICIENT_FACTS:
            if self.selected_provisions:
                raise ValueError("insufficient_facts must not include citations")
            if not self.missing_facts:
                raise ValueError("insufficient_facts requires at least one stable fact key")
            if self.conclusion != Conclusion.INDETERMINATE:
                raise ValueError("insufficient_facts requires an indeterminate conclusion")
        else:
            if self.selected_provisions or self.missing_facts:
                raise ValueError(
                    "no_applicable_provision must not include citations or missing facts"
                )
            if self.conclusion != Conclusion.NEGATIVE:
                raise ValueError("no_applicable_provision requires a negative conclusion")
        return self


class EpisodeOutcome(StrEnum):
    COMPLETED = "completed"
    PROVIDER_ERROR = "provider_error"
    TOOL_OR_ENV_ERROR = "tool_or_env_error"
    SANDBOX_ERROR = "sandbox_error"
    GRADER_ERROR = "grader_error"
    SUT_BUDGET_EXHAUSTED = "sut_budget_exhausted"
    SUT_NO_SUBMISSION = "sut_no_submission"


SUT_TERMINAL_ERROR_GRADES: dict[str, tuple[EpisodeOutcome, str]] = {
    "RuntimeError('sandbox agent exited 22: model request budget exhausted')": (
        EpisodeOutcome.SUT_BUDGET_EXHAUSTED,
        "SUT_BUDGET_EXHAUSTED",
    ),
    "RuntimeError('sandbox agent exited 23: tool call budget exhausted')": (
        EpisodeOutcome.SUT_BUDGET_EXHAUSTED,
        "SUT_BUDGET_EXHAUSTED",
    ),
    "RuntimeError('sandbox agent exited 24: model returned no final submission')": (
        EpisodeOutcome.SUT_NO_SUBMISSION,
        "SUT_NO_SUBMISSION",
    ),
}


class GradeMetrics(StrictModel):
    required_provision_recall: float | None
    selected_provision_precision: float | None
    temporal_history_accuracy: float | None
    quote_integrity: float | None
    forbidden_current_version_rate: float | None
    correct_abstention: float | None


class GradeResult(StrictModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    episode_outcome: EpisodeOutcome
    analysis_eligible: bool
    core_success: Literal[0, 1] | None
    attempt_success: Literal[0, 1]
    failure_codes: list[str]
    metrics: GradeMetrics
    environment_valid: bool
    selected_tuples: list[tuple[str, str, str, str]]


class HumanTaskVerdict(StrEnum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    INDETERMINATE = "indeterminate"


class RootCause(StrEnum):
    SUT_ERROR = "SUT_ERROR"
    TASK_AMBIGUITY = "TASK_AMBIGUITY"
    GOLD_ERROR = "GOLD_ERROR"
    GRADER_ERROR = "GRADER_ERROR"
    TOOL_OR_ENV_ERROR = "TOOL_OR_ENV_ERROR"
    INFRA_ERROR = "INFRA_ERROR"


class TranscriptReview(StrictModel):
    run_id: NonEmptyStr
    reviewer_id: NonEmptyStr
    review_stage: Literal["primary", "secondary", "adjudication"]
    grader_visible: bool
    human_task_verdict: HumanTaskVerdict
    grader_agrees: bool | Literal["not_applicable"]
    environment_valid: bool
    sut_error: bool | Literal["indeterminate"]
    labels: list[RootCause]
    primary_cause: RootCause | None
    note: str


def model_as_dict(model: BaseModel) -> dict[str, Any]:
    return model.model_dump(mode="json")
