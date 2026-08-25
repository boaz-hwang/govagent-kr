from __future__ import annotations

import importlib.metadata
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from inspect_ai.dataset import Sample
from inspect_ai.log import EvalError, EvalRevision
from inspect_ai.model import ChatMessageUser, GenerateConfig, ModelOutput
from inspect_ai.scorer import Score

from govagent_kr import report, review, task
from govagent_kr.canonical import canonical_json_bytes, content_sha256
from govagent_kr.constants import PACKAGE_ROOT, PROFILE_ID, PROFILE_VERSION
from govagent_kr.dataset import load_cases, load_gold
from govagent_kr.freeze import FreezeError
from govagent_kr.models import (
    EpisodeOutcome,
    GoldReview,
    GradeMetrics,
    GradeResult,
    ReviewStatus,
    RootCause,
)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(canonical_json_bytes(record) for record in records))


def _completed_grade(core_success: int = 1) -> dict[str, Any]:
    failure_codes = [] if core_success == 1 else ["SCHEMA_INVALID"]
    return GradeResult(
        episode_outcome=EpisodeOutcome.COMPLETED,
        analysis_eligible=True,
        core_success=core_success,
        attempt_success=core_success,
        failure_codes=failure_codes,
        metrics=GradeMetrics(
            required_provision_recall=None,
            selected_provision_precision=None,
            temporal_history_accuracy=None,
            quote_integrity=None,
            forbidden_current_version_rate=None,
            correct_abstention=None,
        ),
        environment_valid=True,
        selected_tuples=[],
    ).model_dump(mode="json")


def _scored_sample(metadata: dict[str, Any] | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        id="item",
        epoch=1,
        limit=None,
        error=None,
        scores={"govagent_strict_scorer": Score(value=1, metadata=metadata or _completed_grade())},
    )


@pytest.mark.parametrize(
    ("message", "outcome", "failure_code"),
    [
        (
            "RuntimeError('sandbox agent exited 22: model request budget exhausted')",
            "sut_budget_exhausted",
            "SUT_BUDGET_EXHAUSTED",
        ),
        (
            "RuntimeError('sandbox agent exited 23: tool call budget exhausted')",
            "sut_budget_exhausted",
            "SUT_BUDGET_EXHAUSTED",
        ),
        (
            "RuntimeError('sandbox agent exited 24: model returned no final submission')",
            "sut_no_submission",
            "SUT_NO_SUBMISSION",
        ),
    ],
)
def test_sample_grade_classifies_only_exact_sut_terminal_errors(
    message: str, outcome: str, failure_code: str
) -> None:
    sample = SimpleNamespace(
        id="item",
        epoch=1,
        error=EvalError(message=message, traceback="trace", traceback_ansi="trace"),
    )

    grade = report._sample_grade(sample)

    assert grade["episode_outcome"] == outcome
    assert grade["core_success"] == 0
    assert grade["attempt_success"] == 0
    assert grade["analysis_eligible"] is True
    assert grade["failure_codes"] == [failure_code]
    assert review._automatic_grade(sample)["core_success"] == 0


def test_sample_grade_does_not_promote_error_text_containing_exit_substring() -> None:
    sample = SimpleNamespace(
        id="item",
        epoch=1,
        error=EvalError(
            message="ProviderError('upstream said exited 22 while rejecting a request')",
            traceback="trace",
            traceback_ansi="trace",
        ),
    )

    grade = report._sample_grade(sample)

    assert grade["episode_outcome"] is None
    assert grade["attempt_state"] == "excluded_error"
    assert grade["error_classification"] == "unclassified"
    assert grade["core_success"] is None
    assert grade["analysis_eligible"] is False


def test_sample_grade_rejects_coerced_or_inconsistent_metadata() -> None:
    sample = _scored_sample()
    assert report._sample_grade(sample)["core_success"] == 1

    coerced = _completed_grade()
    coerced["attempt_success"] = True
    sample = _scored_sample(coerced)
    with pytest.raises(report.FinalReportError, match="required coercion"):
        report._sample_grade(sample)

    inconsistent = _completed_grade()
    inconsistent["failure_codes"] = ["SCHEMA_INVALID"]
    sample = _scored_sample(inconsistent)
    with pytest.raises(report.FinalReportError, match="failure codes"):
        report._sample_grade(sample)


def test_sample_limit_cannot_be_reported_as_a_completed_grade() -> None:
    sample = _scored_sample()
    sample.limit = SimpleNamespace(type="message", limit=80.0)

    with pytest.raises(report.FinalReportError, match="halted by an Inspect sample limit"):
        report._sample_grade(sample)

    assert review._automatic_grade(sample)["core_success"] is None


def _analysis_freeze() -> dict[str, Any]:
    model_config = {
        "generate": {
            **task.REQUIRED_GENERATE_CONFIG,
            "temperature": 0,
            "reasoning_effort": "high",
        },
        "model_args": {},
    }
    return {
        "profile_id": PROFILE_ID,
        "profile_version": PROFILE_VERSION,
        "freeze_kind": "analysis_freeze",
        "analysis_eligible": True,
        "freeze_id": "f" * 64,
        "requested_model": "mockllm/model",
        "model_config": model_config,
        "dataset_review": {"analysis_ready": True, "pending_human_review": 0},
        "git": {"commit": "a" * 40, "gov_agent_worktree_status": ""},
        "runtime": {"inspect_ai": importlib.metadata.version("inspect-ai")},
        "container": {"image_id": "sha256:" + "c" * 64},
    }


def _frozen_run() -> tuple[SimpleNamespace, dict[str, Any], list[SimpleNamespace]]:
    freeze = _analysis_freeze()
    normalized = task.normalize_frozen_model_config(freeze["model_config"])
    metadata = {
        "freeze_id": freeze["freeze_id"],
        "analysis_eligible": True,
        "candidate_gold_allowed": False,
        "frozen_requested_model": freeze["requested_model"],
        "frozen_model_config": normalized,
        "frozen_model_config_sha256": task.frozen_model_config_sha256(freeze["model_config"]),
        "frozen_container_image_id": freeze["container"]["image_id"],
        "model_request_budget": 12,
        "tool_call_budget": 24,
        "network": "none",
        "provider_web_search": False,
        "provider_code_execution": False,
        "remote_mcp": False,
    }
    config = SimpleNamespace(
        limit=None,
        sample_id=None,
        sample_shuffle=None,
        epochs=3,
        epochs_reducer=[],
        approval=None,
        notification=None,
        fail_on_error=False,
        continue_on_fail=False,
        score_on_error=False,
        retry_on_error=None,
        message_limit=80,
        token_limit=None,
        token_limit_type=None,
        turn_limit=None,
        time_limit=180,
        working_limit=None,
        cost_limit=None,
        max_samples=1,
        max_dataset_memory=None,
        max_tasks=None,
        max_subprocesses=None,
        max_sandboxes=None,
        log_samples=True,
        log_realtime=True,
        log_images=True,
        log_model_api=True,
        log_buffer=1,
        log_shared=None,
        score_display=True,
        sandbox_cleanup=True,
        acp_server=None,
    )
    eval_spec = SimpleNamespace(
        model=freeze["requested_model"],
        task="govagent_kr",
        task_version=PROFILE_VERSION,
        task_file="src/govagent_kr/task.py",
        task_registry_name="govagent_kr",
        solver=None,
        solver_args=None,
        solver_args_passed={},
        dataset=SimpleNamespace(
            name="govagent-kr-v0.1",
            location=None,
            samples=20,
            sample_ids=[f"item-{item}" for item in range(20)],
            shuffled=False,
        ),
        config=config,
        sandbox=SimpleNamespace(type="docker", config=str(task.DEFAULT_FROZEN_COMPOSE_PATH)),
        scorers=[
            SimpleNamespace(
                name="govagent_strict_scorer",
                options={
                    "fixture_root": str((PACKAGE_ROOT / "fixtures/local-contract-decree").resolve())
                },
            )
        ],
        revision=EvalRevision(type="git", origin="git@example", commit="a" * 7, dirty=False),
        packages={"inspect_ai": freeze["runtime"]["inspect_ai"]},
        model_generate_config=GenerateConfig(),
        model_base_url=None,
        model_args={},
        model_roles=None,
        metadata=metadata,
    )
    plan = SimpleNamespace(
        name="plan",
        finish=None,
        steps=[
            SimpleNamespace(
                solver="govagent_sandbox_agent",
                params={
                    "max_model_requests": 12,
                    "max_tool_calls": 24,
                    "timeout_seconds": 175,
                },
            )
        ],
        config=GenerateConfig.model_validate(normalized["generate"]),
    )
    samples = [
        SimpleNamespace(
            id=f"item-{item}",
            epoch=epoch,
            uuid=f"uuid-{item}-{epoch}",
            invalidation=None,
            error_retries=None,
            model_fallbacks=None,
            token_limit=None,
            token_limit_type=None,
            token_limit_usage=None,
            limit=None,
        )
        for item in range(20)
        for epoch in (1, 2, 3)
    ]
    log = SimpleNamespace(
        status="success",
        invalidated=False,
        error=None,
        eval=eval_spec,
        plan=plan,
    )
    return log, freeze, samples


def test_frozen_run_accepts_exact_20_by_3_and_bound_model_config() -> None:
    log, freeze, samples = _frozen_run()
    cases = {f"item-{item}": object() for item in range(20)}

    pairs = report._validate_frozen_run(log, freeze, samples, cases)

    assert len(pairs) == 60


def test_frozen_run_accepts_matching_cli_generation_values_and_empty_solver_args() -> None:
    log, freeze, samples = _frozen_run()
    log.eval.model_generate_config = GenerateConfig(
        max_connections=1,
        max_retries=0,
        timeout=120,
        attempt_timeout=120,
    )
    cases = {f"item-{item}": object() for item in range(20)}

    assert len(report._validate_frozen_run(log, freeze, samples, cases)) == 60


def test_frozen_run_allows_inspect_repo_dirty_flag_from_unrelated_siblings() -> None:
    log, freeze, samples = _frozen_run()
    log.eval.revision.dirty = True
    cases = {f"item-{item}": object() for item in range(20)}

    assert len(report._validate_frozen_run(log, freeze, samples, cases)) == 60


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda log: setattr(log.eval, "task_registry_name", "other"), "task file"),
        (lambda log: setattr(log.eval, "solver_args_passed", {"x": 1}), "solver override"),
        (lambda log: setattr(log.eval.dataset, "shuffled", True), "dataset identity"),
        (lambda log: setattr(log.eval.config, "token_limit", 10), "run controls"),
        (lambda log: setattr(log.eval, "model_roles", {"critic": {}}), "model roles"),
        (
            lambda log: setattr(log.eval.revision, "type", "hg"),
            "source revision",
        ),
    ],
)
def test_frozen_run_rejects_additional_identity_and_limit_mutations(
    mutation: Any, message: str
) -> None:
    log, freeze, samples = _frozen_run()
    mutation(log)
    cases = {f"item-{item}": object() for item in range(20)}

    with pytest.raises(report.FinalReportError, match=message):
        report._validate_frozen_run(log, freeze, samples, cases)


def test_frozen_run_rejects_actual_model_fallback_use() -> None:
    log, freeze, samples = _frozen_run()
    samples[0].model_fallbacks = [SimpleNamespace(fallback_model="other/model")]
    cases = {f"item-{item}": object() for item in range(20)}

    with pytest.raises(report.FinalReportError, match="fallback"):
        report._validate_frozen_run(log, freeze, samples, cases)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda log: setattr(log, "status", "cancelled"), "successful run"),
        (lambda log: setattr(log.eval.config, "epochs", 1), "run controls"),
        (
            lambda log: setattr(log.plan.steps[0], "solver", "generate"),
            "solver plan",
        ),
        (
            lambda log: setattr(log.eval, "model_args", {"reasoning": "low"}),
            "model arguments",
        ),
        (
            lambda log: setattr(log.eval.metadata, "freeze_id", "wrong"),
            "task metadata",
        ),
    ],
)
def test_frozen_run_rejects_protocol_mutations(mutation: Any, message: str) -> None:
    log, freeze, samples = _frozen_run()
    cases = {f"item-{item}": object() for item in range(20)}
    if message == "task metadata":
        log.eval.metadata["freeze_id"] = "wrong"
    else:
        mutation(log)

    with pytest.raises(report.FinalReportError, match=message):
        report._validate_frozen_run(log, freeze, samples, cases)


def _review_sample(*, error: EvalError | None = None) -> SimpleNamespace:
    score = Score(value=1, metadata=_completed_grade())
    return SimpleNamespace(
        id="item",
        epoch=1,
        uuid="sample-uuid",
        input=[ChatMessageUser(content="question")],
        messages=[ChatMessageUser(content="question")],
        output=ModelOutput.from_content(model="mockllm/model", content="answer"),
        error=error,
        scores=None if error else {"govagent_strict_scorer": score},
        total_time=1.0,
        working_time=0.5,
        model_usage={},
        turn_count=1,
        limit=None,
    )


def _identity_and_public(sample: Any, log_file: str, run_id: str) -> tuple[dict, dict]:
    blind_id = "R-blind"
    public = review._safe_sample_payload(sample, blind_id, "mockllm/model")
    automatic = review._automatic_grade(sample)
    identity = {
        "blind_id": blind_id,
        "log_file": log_file,
        "eval_run_id": run_id,
        "sample_id": sample.id,
        "epoch": sample.epoch,
        "sample_uuid": sample.uuid,
        "identity_hmac_sha256": "a" * 64,
        "public_sample_sha256": content_sha256(canonical_json_bytes(public)),
        "automatic_grade_sha256": content_sha256(canonical_json_bytes(automatic)),
    }
    return identity, public


def test_review_source_binding_checks_run_uuid_transcript_and_grade(tmp_path: Path) -> None:
    sample = _review_sample()
    identity, public = _identity_and_public(sample, "run.eval", "run-id")
    _write_jsonl(tmp_path / "blind-items.jsonl", [public])

    automatic = report._validate_review_source_binding(
        log_file="run.eval",
        run_id="run-id",
        run_model="mockllm/model",
        samples_by_pair={("item", 1): sample},
        identities={"R-blind": identity},
        review_bundle_dir=tmp_path,
    )
    assert automatic["R-blind"] == review._automatic_grade(sample)

    changed = deepcopy(identity)
    changed["eval_run_id"] = "other-run"
    with pytest.raises(report.FinalReportError, match="source identity"):
        report._validate_review_source_binding(
            log_file="run.eval",
            run_id="run-id",
            run_model="mockllm/model",
            samples_by_pair={("item", 1): sample},
            identities={"R-blind": changed},
            review_bundle_dir=tmp_path,
        )


def test_frozen_input_comparison_ignores_volatile_inspect_message_ids() -> None:
    first = [ChatMessageUser(content="same question")]
    second = [ChatMessageUser(content="same question")]

    assert first[0].id != second[0].id
    assert report._frozen_input_value(first) == report._frozen_input_value(second)


def _review_record(*, reviewer: str, stage: str, verdict: str = "correct") -> dict[str, Any]:
    labels = [] if verdict == "correct" else [RootCause.SUT_ERROR.value]
    return {
        "blind_id": "R-blind",
        "reviewer_id": reviewer,
        "review_stage": stage,
        "grader_visible": False,
        "human_task_verdict": verdict,
        "environment_valid": True,
        "sut_error": verdict != "correct",
        "labels": labels,
        "primary_cause": None if not labels else RootCause.SUT_ERROR.value,
        "note": "transcript checked",
    }


def test_revealed_reviews_are_bound_to_primary_grade_and_agreement(tmp_path: Path) -> None:
    primary = _review_record(reviewer="primary", stage="primary")
    automatic = {"core_success": 1, "sample_error": None, "scores": {}}
    record = {
        "blind_id": "R-blind",
        "primary_review": primary,
        "automatic_grade": automatic,
        "grader_agrees": True,
    }
    path = tmp_path / "revealed.jsonl"
    _write_jsonl(path, [record])
    report._validate_revealed_reviews(path, {"R-blind": primary}, {"R-blind": automatic})

    record["grader_agrees"] = False
    _write_jsonl(path, [record])
    with pytest.raises(report.FinalReportError, match="differs from source"):
        report._validate_revealed_reviews(path, {"R-blind": primary}, {"R-blind": automatic})


def test_run_result_contains_item_level_automatic_and_human_evidence() -> None:
    primary = _review_record(reviewer="primary", stage="primary")
    automatic = {"core_success": 1, "sample_error": None, "scores": {}}
    record = report._run_result_record(
        blind_id="R-blind",
        identity={"sample_id": "item", "epoch": 2},
        family_id="family",
        grade=report._sample_grade(_scored_sample()),
        human_review=primary,
        automatic_grade=automatic,
    )

    assert record["item_id"] == "item"
    assert record["epoch"] == 2
    assert record["core_success"] == 1
    assert record["human_task_verdict"] == "correct"
    assert record["reviewer_id"] == "primary"
    assert record["review_note"] == "transcript checked"
    assert record["grader_agrees"] is True


def test_go_gate_rejects_repeated_grader_disagreements_in_either_direction() -> None:
    with pytest.raises(report.FinalReportError, match="repeated grader disagreements"):
        report._validate_decision_gate(
            decision="go_expand_eval",
            analysis_eligible_attempts=60,
            environment_invalid_runs=0,
            unresolved_human_runs=0,
            sut_error_family_count=2,
            grader_disagreement_runs=2,
            design_error_runs=2,
            genuine_sut_error_runs=2,
        )


@pytest.mark.parametrize(
    ("analysis_eligible_attempts", "environment_invalid_runs"),
    [(59, 0), (60, 1)],
)
def test_go_gate_requires_complete_valid_environment_evidence(
    analysis_eligible_attempts: int,
    environment_invalid_runs: int,
) -> None:
    with pytest.raises(report.FinalReportError, match="all 60 attempts"):
        report._validate_decision_gate(
            decision="go_expand_eval",
            analysis_eligible_attempts=analysis_eligible_attempts,
            environment_invalid_runs=environment_invalid_runs,
            unresolved_human_runs=0,
            sut_error_family_count=2,
            grader_disagreement_runs=0,
            design_error_runs=0,
            genuine_sut_error_runs=2,
        )


def test_go_gate_rejects_unresolved_human_verdict_or_attribution() -> None:
    with pytest.raises(report.FinalReportError, match="resolved human verdict"):
        report._validate_decision_gate(
            decision="go_expand_eval",
            analysis_eligible_attempts=60,
            environment_invalid_runs=0,
            unresolved_human_runs=1,
            sut_error_family_count=2,
            grader_disagreement_runs=0,
            design_error_runs=0,
            genuine_sut_error_runs=2,
        )


def test_disagreement_artifact_is_recomputed_exactly(tmp_path: Path) -> None:
    primary = _review_record(reviewer="primary", stage="primary")
    secondary = _review_record(reviewer="secondary", stage="secondary", verdict="incorrect")
    expected = report._expected_disagreements({"R-blind": primary}, [secondary])
    path = tmp_path / "disagreements.json"
    path.write_bytes(
        canonical_json_bytes(
            {
                "schema_version": "govagent-kr/review-disagreements-v0.1",
                "automatic_adjudication": False,
                "disagreements": expected,
            }
        )
    )
    assert report._validate_disagreement_artifact(path, {"R-blind": primary}, [secondary]) == {
        "R-blind"
    }

    path.write_bytes(
        canonical_json_bytes(
            {
                "schema_version": "govagent-kr/review-disagreements-v0.1",
                "automatic_adjudication": False,
                "disagreements": [],
            }
        )
    )
    with pytest.raises(report.FinalReportError, match="disagreement artifact"):
        report._validate_disagreement_artifact(path, {"R-blind": primary}, [secondary])


def _adjudication() -> dict[str, Any]:
    return {
        "blind_id": "R-blind",
        "adjudicator_id": "adjudicator",
        "review_stage": "adjudication",
        "grader_visible": False,
        "human_task_verdict": "incorrect",
        "environment_valid": True,
        "sut_error": True,
        "labels": [RootCause.SUT_ERROR.value],
        "primary_cause": RootCause.SUT_ERROR.value,
        "note": "resolved from transcript evidence",
    }


def test_genuine_sut_error_requires_unconfounded_primary_sut_cause() -> None:
    review_record = _adjudication()
    grade = report._sample_grade(_scored_sample(_completed_grade(core_success=0)))

    assert report._genuine_sut_error(review_record, grade) is True

    review_record["labels"].append(RootCause.TASK_AMBIGUITY.value)
    assert report._genuine_sut_error(review_record, grade) is False

    review_record["labels"] = [RootCause.SUT_ERROR.value, RootCause.GOLD_ERROR.value]
    review_record["primary_cause"] = RootCause.GOLD_ERROR.value
    assert report._genuine_sut_error(review_record, grade) is False


def test_adjudication_is_independent_and_fail_closed(tmp_path: Path) -> None:
    primary = _review_record(reviewer="primary", stage="primary")
    secondary = _review_record(reviewer="secondary", stage="secondary")
    path = tmp_path / "adjudications.jsonl"
    record = _adjudication()
    _write_jsonl(path, [record])
    result = report._validate_adjudications(
        path,
        {"R-blind"},
        {"R-blind": primary},
        {"R-blind": secondary},
    )
    assert result["R-blind"]["sut_error"] is True

    record["sut_error"] = 1
    _write_jsonl(path, [record])
    with pytest.raises(report.FinalReportError, match="invalid sut_error"):
        report._validate_adjudications(
            path,
            {"R-blind"},
            {"R-blind": primary},
            {"R-blind": secondary},
        )

    record = _adjudication()
    record["labels"] = [RootCause.TOOL_OR_ENV_ERROR.value]
    record["primary_cause"] = RootCause.TOOL_OR_ENV_ERROR.value
    record["sut_error"] = False
    _write_jsonl(path, [record])
    with pytest.raises(report.FinalReportError, match="environment error labels require"):
        report._validate_adjudications(
            path,
            {"R-blind"},
            {"R-blind": primary},
            {"R-blind": secondary},
        )

    record = _adjudication()
    record["adjudicator_id"] = " PRIMARY "
    _write_jsonl(path, [record])
    with pytest.raises(report.FinalReportError, match="independent"):
        report._validate_adjudications(
            path,
            {"R-blind"},
            {"R-blind": primary},
            {"R-blind": secondary},
        )


def test_secondary_selection_treats_indeterminate_as_mandatory_nonclean() -> None:
    clean = [
        {
            **_review_record(reviewer="primary", stage="primary"),
            "blind_id": f"R-{index:02d}",
        }
        for index in range(12)
    ]
    indeterminate = {
        **_review_record(reviewer="primary", stage="primary"),
        "blind_id": "R-99",
        "human_task_verdict": "indeterminate",
        "sut_error": "indeterminate",
    }

    selected = review._secondary_selection_ids([*clean, indeterminate])

    assert "R-99" in selected
    assert len(selected) == review.SECONDARY_CLEAN_SAMPLE_SIZE + 1


def test_analysis_freeze_requires_fixed_generate_controls() -> None:
    freeze = _analysis_freeze()
    normalized = task._validate_analysis_freeze(freeze)
    assert normalized["generate"]["max_retries"] == 0

    freeze["model_config"]["generate"]["max_retries"] = 2
    with pytest.raises(ValueError, match="conflicts"):
        task._validate_analysis_freeze(freeze)


def test_analysis_model_config_is_single_sut_and_provider_capability_closed() -> None:
    normalized = task.normalize_frozen_model_config({"temperature": 0})
    assert normalized == {
        "generate": {**task.REQUIRED_GENERATE_CONFIG, "temperature": 0.0},
        "model_args": {},
    }

    forbidden = [
        {"generate": {"fallback_models": ["other/model"]}, "model_args": {}},
        {"generate": {"internal_tools": True}, "model_args": {}},
        {
            "generate": {"extra_headers": {"Authorization": "Bearer secret"}},
            "model_args": {},
        },
        {"generate": {}, "model_args": {"base_url": "https://example.invalid"}},
        {
            "generate": {},
            "model_args": {"default_headers": {"Authorization": "Bearer secret"}},
        },
        {"generate": {}, "model_args": {"provider_option": "value"}},
    ]
    for value in forbidden:
        with pytest.raises(ValueError):
            task.normalize_frozen_model_config(value)


def test_system_prompt_requires_smallest_synthetic_locator_passages() -> None:
    assert "smallest locator passages" in task.SYSTEM_PROMPT
    assert "#본문" in task.SYSTEM_PROMPT
    assert "parent aggregate passage is\nnot a substitute" in task.SYSTEM_PROMPT


def test_analysis_task_binds_paths_trials_image_model_and_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    freeze = _analysis_freeze()
    manifest = tmp_path / "freeze.json"
    manifest.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(task, "verify_freeze_manifest", lambda *args, **kwargs: freeze)
    monkeypatch.setattr(
        task,
        "_samples",
        lambda **kwargs: [Sample(id="item", input="question", target="{}")],
    )
    monkeypatch.setenv("GOVAGENT_SUT_IMAGE", freeze["container"]["image_id"])

    eval_task = task.govagent_kr(
        freeze_manifest=str(manifest),
        compose_path=str(task.DEFAULT_FROZEN_COMPOSE_PATH),
        trials=3,
    )

    assert eval_task.epochs == 3
    assert eval_task.metadata["candidate_gold_allowed"] is False
    assert eval_task.metadata["frozen_model_config"] == task.normalize_frozen_model_config(
        freeze["model_config"]
    )
    assert eval_task.config.max_retries == 0
    assert eval_task.model is not None

    with pytest.raises(ValueError, match="exactly 3 trials"):
        task.govagent_kr(
            freeze_manifest=str(manifest),
            compose_path=str(task.DEFAULT_FROZEN_COMPOSE_PATH),
            trials=1,
        )


def test_sample_candidate_flag_tracks_adjudicated_gold(tmp_path: Path) -> None:
    case = load_cases()[0]
    gold = load_gold()[0].model_copy(
        update={
            "review": GoldReview(
                status=ReviewStatus.ADJUDICATED,
                reviewer_1="one",
                reviewer_2="two",
                reviewer_1_verdict="correct",
                reviewer_2_verdict="correct",
                resolution="accepted",
            )
        }
    )
    cases_path = tmp_path / "cases.jsonl"
    gold_path = tmp_path / "gold.jsonl"
    _write_jsonl(cases_path, [case.model_dump(mode="json")])
    _write_jsonl(gold_path, [gold.model_dump(mode="json")])

    sample = task._samples(
        cases_path=cases_path,
        gold_path=gold_path,
        allow_pending_gold=False,
    )[0]

    assert sample.metadata["candidate_gold"] is False
    assert sample.metadata["case_originated_as_candidate"] is True


def _receipt_without_docker_smoke() -> dict[str, Any]:
    gates = [
        {
            "command": list(command),
            "exit_code": 0,
            "stdout_sha256": "1" * 64,
            "stderr_sha256": "2" * 64,
            "stdout_tail": "ok",
            "stderr_tail": "",
        }
        for command in report.QUALITY_GATE_COMMANDS
    ]
    receipt: dict[str, Any] = {
        "schema_version": "govagent-kr/quality-receipt-v0.1",
        "profile_id": PROFILE_ID,
        "profile_version": PROFILE_VERSION,
        "docker_smoke_included": False,
        "docker_smoke_compose": "compose.yaml",
        "git": {"commit": "a" * 40, "gov_agent_worktree_status": ""},
        "artifact_set_sha256": "b" * 64,
        "container": None,
        "gates": gates,
        "ok": True,
        "analysis_quality_eligible": False,
    }
    receipt["receipt_sha256"] = content_sha256(canonical_json_bytes(receipt))
    return receipt


def test_readiness_rejects_bare_or_non_docker_quality_receipts(tmp_path: Path) -> None:
    valid, error = report._validate_current_quality_receipt({"ok": True}, root=tmp_path)
    assert valid is False
    assert error == "quality receipt hash is invalid"

    valid, error = report._validate_current_quality_receipt(
        _receipt_without_docker_smoke(), root=tmp_path
    )
    assert valid is False
    assert error == "quality receipt did not include a recognized Docker smoke"


def test_readiness_counts_only_verified_analysis_freezes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "forged-freeze.json").write_text('{"analysis_eligible":true}', encoding="utf-8")
    diagnostics = {
        "engineering_ready": True,
        "analysis_ready": True,
        "dataset_integrity": {"pending_human_review": 0},
    }
    monkeypatch.setattr(report, "run_diagnostics", lambda *args: diagnostics)
    monkeypatch.setattr(
        report,
        "_validate_current_quality_receipt",
        lambda receipt, root: (True, None),
    )
    monkeypatch.setattr(
        report,
        "verify_freeze_manifest",
        lambda *args, **kwargs: (_ for _ in ()).throw(FreezeError("forged")),
    )
    monkeypatch.setattr(report, "_docker_status", lambda: {"available": True, "image_ids": []})
    for name in report.PROVIDER_CREDENTIAL_NAMES:
        monkeypatch.delenv(name, raising=False)

    readiness = report.build_readiness_report(root=tmp_path)

    assert readiness["gates"]["analysis_freeze_receipts"] == []
    assert readiness["state"] == "engineering_environment_ready"


def test_eval_log_file_uri_is_hashed_as_path_independent_evidence(tmp_path: Path) -> None:
    first = tmp_path / "first.eval"
    second = tmp_path / "second.eval"
    first.write_bytes(b"same eval bytes")
    second.write_bytes(b"same eval bytes")

    assert report._local_eval_log_path(first.as_uri()) == first
    first_record = report._evidence_artifact_records({"inspect_eval_log": first})
    second_record = report._evidence_artifact_records({"inspect_eval_log": second})
    assert first_record == second_record
    assert first_record[0]["sha256"] == content_sha256(b"same eval bytes")


@pytest.mark.parametrize("exc", [FreezeError("bad freeze"), review.ReviewError("bad review")])
def test_final_cli_converts_freeze_and_review_errors_to_parser_errors(
    exc: Exception,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail(**kwargs: Any) -> dict[str, Any]:
        raise exc

    monkeypatch.setattr(report, "build_final_report", fail)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "govagent-final-report",
            "--freeze",
            "freeze.json",
            "--log-dir",
            "logs",
            "--review-bundle",
            "primary",
            "--primary-reviews",
            "primary.csv",
            "--secondary-bundle",
            "secondary",
            "--secondary-reviews",
            "secondary.csv",
            "--revealed-reviews",
            "revealed.jsonl",
            "--disagreements",
            "disagreements.json",
            "--decision",
            "stop",
            "--decision-note",
            "human decision",
            "--json-output",
            "result.json",
            "--markdown-output",
            "result.md",
        ],
    )

    with pytest.raises(SystemExit, match="2"):
        report.final_main()
    assert str(exc) in capsys.readouterr().err


def test_task_file_loads_through_the_documented_inspect_cli_spec() -> None:
    inspect_executable = Path(sys.executable).with_name("inspect")
    completed = subprocess.run(
        [str(inspect_executable), "list", "tasks", "src/govagent_kr/task.py"],
        cwd=PACKAGE_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "src/govagent_kr/task.py@govagent_kr" in completed.stdout
