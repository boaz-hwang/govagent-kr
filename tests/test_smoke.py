import csv
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from inspect_ai.log import EvalRevision
from inspect_ai.model import GenerateConfig
from pydantic import BaseModel

from govagent_kr import smoke
from govagent_kr.canonical import canonical_json_bytes
from govagent_kr.constants import PACKAGE_ROOT, PROFILE_ID, PROFILE_VERSION, SMOKE_ITEM_IDS
from govagent_kr.model_config import normalize_frozen_model_config
from govagent_kr.review import REVIEW_COLUMNS, export_blind_bundle
from govagent_kr.smoke import (
    SmokeReviewError,
    build_smoke_review_receipt,
    verify_smoke_review_receipt_sources,
)


class FakeScore(BaseModel):
    value: int
    metadata: dict[str, Any]


def _sample(item_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=item_id,
        epoch=1,
        uuid=f"uuid-{item_id}",
        invalidation=None,
        error_retries=None,
        error=None,
        model_fallbacks=None,
        limit=None,
        token_limit=None,
        token_limit_type=None,
        token_limit_usage=None,
        input={"question": "기준일의 조문을 찾으시오."},
        target={"secret_gold": True},
        messages=[{"role": "assistant", "content": "도구를 사용했습니다."}],
        output={"status": "answered", "selected_provisions": []},
        total_time=1.0,
        working_time=0.8,
        model_usage={"input_tokens": 50, "output_tokens": 10},
        turn_count=2,
        scores={
            "govagent_strict_scorer": FakeScore(
                value=1,
                metadata={"core_success": 1},
            )
        },
    )


def _freeze() -> dict[str, Any]:
    model_config = normalize_frozen_model_config({"temperature": 0, "reasoning_effort": "high"})
    return {
        "profile_id": PROFILE_ID,
        "profile_version": PROFILE_VERSION,
        "freeze_kind": "smoke_freeze",
        "analysis_eligible": False,
        "smoke_eligible": True,
        "freeze_id": "f" * 64,
        "requested_model": "mockllm/model",
        "model_config": model_config,
        "artifact_set_sha256": "1" * 64,
        "git": {"commit": "a" * 40, "gov_agent_worktree_status": ""},
        "runtime": {"inspect_ai": "0.3.260"},
        "container": {"image_id": "sha256:" + "c" * 64},
        "quality_receipt": {"receipt_sha256": "2" * 64},
        "gold_review_receipt": {"receipt_sha256": "3" * 64},
    }


def _log(freeze: dict[str, Any], samples: list[SimpleNamespace]) -> SimpleNamespace:
    model_config = freeze["model_config"]
    config = SimpleNamespace(
        limit=None,
        sample_id=list(SMOKE_ITEM_IDS),
        sample_shuffle=None,
        epochs=1,
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
    metadata = {
        "freeze_id": freeze["freeze_id"],
        "analysis_eligible": False,
        "smoke_eligible": True,
        "candidate_gold_allowed": False,
        "frozen_requested_model": freeze["requested_model"],
        "frozen_model_config": model_config,
        "frozen_model_config_sha256": smoke.frozen_model_config_sha256(model_config),
        "frozen_container_image_id": freeze["container"]["image_id"],
        "model_request_budget": 12,
        "tool_call_budget": 24,
        "network": "none",
        "provider_web_search": False,
        "provider_code_execution": False,
        "remote_mcp": False,
    }
    eval_spec = SimpleNamespace(
        run_id="smoke-run",
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
            sample_ids=list(SMOKE_ITEM_IDS),
            shuffled=False,
        ),
        config=config,
        sandbox=SimpleNamespace(
            type="docker",
            config=str((PACKAGE_ROOT / "compose.frozen.yaml").resolve()),
        ),
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
        model_args={},
        model_base_url=None,
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
        config=GenerateConfig.model_validate(model_config["generate"]),
    )
    return SimpleNamespace(
        status="success",
        invalidated=False,
        error=None,
        eval=eval_spec,
        plan=plan,
        samples=samples,
    )


def _patch_inspect(
    monkeypatch: pytest.MonkeyPatch,
    log_path: Path,
    log: SimpleNamespace,
) -> None:
    import inspect_ai.log

    info = SimpleNamespace(name=str(log_path))
    monkeypatch.setattr(inspect_ai.log, "list_eval_logs", lambda *args, **kwargs: [info])
    monkeypatch.setattr(
        inspect_ai.log,
        "read_eval_log",
        lambda value, header_only=False: log,
    )
    monkeypatch.setattr(inspect_ai.log, "read_eval_log_samples", lambda value: iter(log.samples))


def _complete_review(path: Path) -> None:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row.update(
            reviewer_id="reviewer-a",
            human_task_verdict="correct",
            environment_valid="true",
            sut_error="false",
            labels="",
            primary_cause="",
            note="grader-blind smoke review",
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _workflow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, Path, Path, dict[str, Any]]:
    root = tmp_path / "project"
    (root / "data").mkdir(parents=True)
    (root / "data/cases.jsonl").write_text("{}\n", encoding="utf-8")
    (root / "data/gold.jsonl").write_text("{}\n", encoding="utf-8")
    smoke_freeze = _freeze()
    smoke_freeze_path = tmp_path / "smoke-freeze.json"
    smoke_freeze_path.write_bytes(canonical_json_bytes(smoke_freeze))
    log_path = tmp_path / "smoke.eval"
    log_path.write_bytes(b"synthetic Inspect log")
    samples = [_sample(item_id) for item_id in SMOKE_ITEM_IDS]
    log = _log(smoke_freeze, samples)
    _patch_inspect(monkeypatch, log_path, log)
    monkeypatch.setattr(smoke, "verify_freeze_manifest", lambda *args, **kwargs: smoke_freeze)
    monkeypatch.setattr(smoke, "_validate_samples_and_replay", lambda samples: None)
    bundle = tmp_path / "bundle"
    key = tmp_path / "blind.key"
    key.write_bytes(b"k" * 32)
    export_blind_bundle(tmp_path, bundle, key)
    reviews = bundle / "primary-review.csv"
    _complete_review(reviews)
    return root, smoke_freeze_path, log_path, bundle, smoke_freeze


def test_smoke_receipt_binds_fixed_log_freeze_and_blind_reviews(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, smoke_freeze_path, _log_path, bundle, freeze = _workflow(tmp_path, monkeypatch)
    reviews = bundle / "primary-review.csv"

    receipt = build_smoke_review_receipt(
        smoke_freeze_path=smoke_freeze_path,
        log_dir=tmp_path,
        review_bundle_dir=bundle,
        reviews_path=reviews,
        root=root,
    )

    assert receipt["item_ids"] == list(SMOKE_ITEM_IDS)
    assert receipt["item_count"] == 5
    assert receipt["smoke_freeze_id"] == freeze["freeze_id"]
    assert receipt["human_review_complete"] is True
    assert receipt["grader_visible"] is False
    assert receipt["grader_agreement_complete"] is True
    assert receipt["grader_agreement_count"] == 5
    assert receipt["all_environment_valid"] is True
    assert receipt["unresolved_defect_count"] == 0
    receipt_path = tmp_path / "smoke-receipt.json"
    receipt_path.write_bytes(canonical_json_bytes(receipt))
    assert (
        verify_smoke_review_receipt_sources(
            receipt_path,
            smoke_freeze_path=smoke_freeze_path,
            log_dir=tmp_path,
            review_bundle_dir=bundle,
            reviews_path=reviews,
            root=root,
        )
        == receipt
    )


def test_smoke_receipt_rejects_unresolved_defect_and_source_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, smoke_freeze_path, log_path, bundle, _ = _workflow(tmp_path, monkeypatch)
    reviews = bundle / "primary-review.csv"
    with reviews.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0].update(
        human_task_verdict="incorrect",
        environment_valid="true",
        sut_error="false",
        labels="TASK_AMBIGUITY",
        primary_cause="TASK_AMBIGUITY",
    )
    with reviews.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(SmokeReviewError, match="unresolved"):
        build_smoke_review_receipt(
            smoke_freeze_path=smoke_freeze_path,
            log_dir=tmp_path,
            review_bundle_dir=bundle,
            reviews_path=reviews,
            root=root,
        )

    _complete_review(reviews)
    receipt = build_smoke_review_receipt(
        smoke_freeze_path=smoke_freeze_path,
        log_dir=tmp_path,
        review_bundle_dir=bundle,
        reviews_path=reviews,
        root=root,
    )
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_bytes(canonical_json_bytes(receipt))
    log_path.write_bytes(b"tampered log")
    with pytest.raises(SmokeReviewError, match="differs from its current source evidence"):
        verify_smoke_review_receipt_sources(
            receipt_path,
            smoke_freeze_path=smoke_freeze_path,
            log_dir=tmp_path,
            review_bundle_dir=bundle,
            reviews_path=reviews,
            root=root,
        )


def test_smoke_receipt_rejects_wrong_sample_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, smoke_freeze_path, _log_path, bundle, _ = _workflow(tmp_path, monkeypatch)
    import inspect_ai.log

    original = inspect_ai.log.read_eval_log(SimpleNamespace(name="unused"))
    original.samples[-1].id = "unexpected-item"
    with pytest.raises(SmokeReviewError, match="fixed five items"):
        build_smoke_review_receipt(
            smoke_freeze_path=smoke_freeze_path,
            log_dir=tmp_path,
            review_bundle_dir=bundle,
            reviews_path=bundle / "primary-review.csv",
            root=root,
        )


def test_smoke_receipt_rejects_human_and_grader_disagreement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, smoke_freeze_path, _log_path, bundle, _ = _workflow(tmp_path, monkeypatch)
    reviews = bundle / "primary-review.csv"
    with reviews.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0].update(
        human_task_verdict="incorrect",
        environment_valid="true",
        sut_error="true",
        labels="SUT_ERROR",
        primary_cause="SUT_ERROR",
    )
    with reviews.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(SmokeReviewError, match="frozen grader disagree"):
        build_smoke_review_receipt(
            smoke_freeze_path=smoke_freeze_path,
            log_dir=tmp_path,
            review_bundle_dir=bundle,
            reviews_path=reviews,
            root=root,
        )
