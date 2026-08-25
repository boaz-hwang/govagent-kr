from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from .canonical import canonical_json_bytes, content_sha256
from .constants import (
    PACKAGE_ROOT,
    PROFILE_ID,
    PROFILE_VERSION,
    SMOKE_ITEM_IDS,
)
from .freeze import FreezeError, verify_freeze_manifest
from .model_config import (
    REQUIRED_GENERATE_CONFIG,
    frozen_model_config_sha256,
    normalize_frozen_model_config,
)
from .models import HumanTaskVerdict, RootCause
from .report import (
    FinalReportError,
    _load_identity_map,
    _validate_review_source_binding,
    _validate_samples_and_replay,
)
from .review import (
    ReviewError,
    _normalized_reviews_sha256,
    _verify_bundle_integrity,
    validate_primary_reviews,
)
from .strict_json import StrictJSONError, parse_strict_json_object

SMOKE_RECEIPT_SCHEMA_VERSION = "govagent-kr/smoke-review-receipt-v0.1"


class SmokeReviewError(RuntimeError):
    """Raised when the five-item pre-analysis smoke is incomplete or unbound."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _local_eval_log_path(value: str) -> Path:
    parsed = urlparse(value)
    if parsed.scheme == "file" and parsed.netloc in {"", "localhost"}:
        return Path(unquote(parsed.path))
    if parsed.scheme:
        raise SmokeReviewError("smoke receipt requires a local file-backed Inspect log")
    return Path(value)


def _resolved_task_file(value: Any) -> Path | None:
    if not isinstance(value, str):
        return None
    path = Path(value)
    if not path.is_absolute():
        path = PACKAGE_ROOT / path
    return path.resolve()


def _validate_smoke_log(log: Any, freeze: dict[str, Any], samples: list[Any]) -> None:
    from .task import (
        DEFAULT_FROZEN_COMPOSE_PATH,
        MAX_MODEL_REQUESTS,
        MAX_TOOL_CALLS,
        WALL_CLOCK_SECONDS,
    )

    expected_ids = list(SMOKE_ITEM_IDS)
    expected_pairs = {(item_id, 1) for item_id in expected_ids}
    actual_pairs = {(str(sample.id), sample.epoch) for sample in samples}
    if log.status != "success" or log.invalidated or log.error is not None:
        raise SmokeReviewError("Inspect smoke log is not a valid successful run")
    if len(samples) != len(expected_pairs) or actual_pairs != expected_pairs:
        raise SmokeReviewError("Inspect smoke log is not the fixed five items at epoch 1")
    uuids = [sample.uuid for sample in samples]
    if (
        any(not isinstance(value, str) or not value for value in uuids)
        or len(set(uuids)) != len(uuids)
        or any(sample.invalidation is not None for sample in samples)
        or any(sample.error_retries for sample in samples)
    ):
        raise SmokeReviewError("Inspect smoke samples have invalid identity or retry history")
    if any(sample.error is not None for sample in samples):
        raise SmokeReviewError("Inspect smoke contains an errored sample")
    if any(set(sample.scores or {}) != {"govagent_strict_scorer"} for sample in samples):
        raise SmokeReviewError("Inspect smoke does not contain one frozen score per sample")
    if any(sample.model_fallbacks for sample in samples):
        raise SmokeReviewError("Inspect smoke used a model fallback")
    if any(
        sample.limit is not None
        or sample.token_limit is not None
        or sample.token_limit_type is not None
        or sample.token_limit_usage is not None
        for sample in samples
    ):
        raise SmokeReviewError("Inspect smoke used a sample-specific limit")

    if log.eval.model != freeze.get("requested_model"):
        raise SmokeReviewError("Inspect smoke model differs from the smoke freeze")
    if log.eval.task != "govagent_kr" or log.eval.task_version != PROFILE_VERSION:
        raise SmokeReviewError("Inspect smoke task identity differs")
    if (
        _resolved_task_file(log.eval.task_file)
        != (PACKAGE_ROOT / "src/govagent_kr/task.py").resolve()
        or log.eval.task_registry_name != "govagent_kr"
        or log.eval.solver is not None
        or log.eval.solver_args is not None
        or log.eval.solver_args_passed != {}
    ):
        raise SmokeReviewError("Inspect smoke task file or solver override differs")

    dataset = log.eval.dataset
    if (
        dataset.name != "govagent-kr-v0.1"
        or dataset.location is not None
        or dataset.samples != 20
        or dataset.sample_ids != expected_ids
        or dataset.shuffled is not False
    ):
        raise SmokeReviewError("Inspect smoke dataset identity differs")
    config = log.eval.config
    if (
        config.limit is not None
        or config.sample_id != expected_ids
        or config.sample_shuffle is not None
        or config.epochs != 1
        or config.epochs_reducer != []
        or config.fail_on_error is not False
        or config.continue_on_fail is not False
        or config.score_on_error is not False
        or config.retry_on_error not in {None, 0}
        or config.approval is not None
        or config.notification is not None
        or config.message_limit != 80
        or config.token_limit is not None
        or config.token_limit_type is not None
        or config.turn_limit is not None
        or config.time_limit != WALL_CLOCK_SECONDS
        or config.working_limit is not None
        or config.cost_limit is not None
        or config.max_samples != 1
        or config.max_dataset_memory is not None
        or config.max_tasks is not None
        or config.max_subprocesses is not None
        or config.max_sandboxes is not None
        or config.log_samples is not True
        or config.log_realtime is not True
        or config.log_images is not True
        or config.log_model_api is not True
        or config.log_buffer != 1
        or config.log_shared not in {None, 0}
        or config.score_display is not True
        or config.sandbox_cleanup is not True
        or config.acp_server is not None
    ):
        raise SmokeReviewError("Inspect smoke run controls differ from the fixed protocol")
    if (
        log.plan.name != "plan"
        or log.plan.finish is not None
        or len(log.plan.steps) != 1
        or log.plan.steps[0].solver != "govagent_sandbox_agent"
        or log.plan.steps[0].params
        != {
            "max_model_requests": MAX_MODEL_REQUESTS,
            "max_tool_calls": MAX_TOOL_CALLS,
            "timeout_seconds": WALL_CLOCK_SECONDS - 5,
        }
    ):
        raise SmokeReviewError("Inspect smoke solver plan differs")
    sandbox = log.eval.sandbox
    if (
        sandbox is None
        or sandbox.type != "docker"
        or not isinstance(sandbox.config, str)
        or Path(sandbox.config).resolve() != DEFAULT_FROZEN_COMPOSE_PATH.resolve()
    ):
        raise SmokeReviewError("Inspect smoke did not use the frozen Docker sandbox")
    scorers = log.eval.scorers or []
    if (
        len(scorers) != 1
        or scorers[0].name != "govagent_strict_scorer"
        or scorers[0].options
        != {"fixture_root": str((PACKAGE_ROOT / "fixtures/local-contract-decree").resolve())}
    ):
        raise SmokeReviewError("Inspect smoke did not use the frozen scorer")

    git_commit = freeze.get("git", {}).get("commit")
    revision = log.eval.revision
    if (
        revision is None
        or revision.type != "git"
        or type(revision.dirty) is not bool
        or not isinstance(git_commit, str)
        or not isinstance(revision.commit, str)
        or not 7 <= len(revision.commit) <= 40
        or any(character not in "0123456789abcdef" for character in revision.commit)
        or not git_commit.startswith(revision.commit)
    ):
        raise SmokeReviewError("Inspect smoke source revision differs from the smoke freeze")
    if log.eval.packages.get("inspect_ai") != freeze.get("runtime", {}).get("inspect_ai"):
        raise SmokeReviewError("Inspect smoke package version differs from the smoke freeze")

    expected_model_config = normalize_frozen_model_config(freeze.get("model_config"))
    if any(
        expected_model_config["generate"].get(key) != value
        for key, value in REQUIRED_GENERATE_CONFIG.items()
    ):
        raise SmokeReviewError("smoke freeze does not fix required model controls")
    effective_generate = log.plan.config.model_dump(mode="json", exclude_none=True)
    model_generate = log.eval.model_generate_config.model_dump(mode="json", exclude_none=True)
    if effective_generate != expected_model_config["generate"] or any(
        expected_model_config["generate"].get(key) != value for key, value in model_generate.items()
    ):
        raise SmokeReviewError("Inspect smoke generation settings differ from the smoke freeze")
    if (
        log.eval.model_args != expected_model_config["model_args"]
        or log.eval.model_base_url is not None
        or log.eval.model_roles
    ):
        raise SmokeReviewError("Inspect smoke model arguments or roles differ")

    container = freeze.get("container")
    image_id = container.get("image_id") if isinstance(container, dict) else None
    expected_metadata = {
        "freeze_id": freeze.get("freeze_id"),
        "analysis_eligible": False,
        "smoke_eligible": True,
        "candidate_gold_allowed": False,
        "frozen_requested_model": freeze.get("requested_model"),
        "frozen_model_config": expected_model_config,
        "frozen_model_config_sha256": frozen_model_config_sha256(freeze.get("model_config")),
        "frozen_container_image_id": image_id,
        "model_request_budget": MAX_MODEL_REQUESTS,
        "tool_call_budget": MAX_TOOL_CALLS,
        "network": "none",
        "provider_web_search": False,
        "provider_code_execution": False,
        "remote_mcp": False,
    }
    metadata = log.eval.metadata or {}
    if any(metadata.get(key) != value for key, value in expected_metadata.items()):
        raise SmokeReviewError("Inspect smoke metadata differs from the smoke freeze")
    _validate_samples_and_replay(samples)


def _receipt_hashes(
    *,
    smoke_freeze_path: Path,
    log_path: Path,
    review_bundle_dir: Path,
    reviews_path: Path,
) -> dict[str, str]:
    paths = {
        "smoke_freeze_sha256": smoke_freeze_path,
        "source_log_sha256": log_path,
        "review_bundle_manifest_sha256": review_bundle_dir / "manifest.json",
        "blind_items_sha256": review_bundle_dir / "blind-items.jsonl",
        "identity_map_sha256": review_bundle_dir / "private" / "identity-map.jsonl",
    }
    hashes = {name: _file_sha256(path) for name, path in paths.items()}
    hashes["primary_reviews_sha256"] = _normalized_reviews_sha256(
        validate_primary_reviews(reviews_path, paths["identity_map_sha256"])
    )
    return hashes


def build_smoke_review_receipt(
    *,
    smoke_freeze_path: Path,
    log_dir: Path,
    review_bundle_dir: Path,
    reviews_path: Path,
    root: Path = PACKAGE_ROOT,
) -> dict[str, Any]:
    from inspect_ai.log import list_eval_logs, read_eval_log

    root = root.resolve()
    freeze = verify_freeze_manifest(
        smoke_freeze_path,
        root=root,
        require_smoke=True,
    )
    log_infos = list_eval_logs(str(log_dir), formats=["eval"])
    if len(log_infos) != 1:
        raise SmokeReviewError("smoke receipt requires exactly one Inspect .eval log")
    log_info = log_infos[0]
    log = read_eval_log(log_info)
    samples = log.samples or []
    _validate_smoke_log(log, freeze, samples)

    public_records = _verify_bundle_integrity(review_bundle_dir)
    identity_path = review_bundle_dir / "private" / "identity-map.jsonl"
    identities = _load_identity_map(identity_path)
    reviews = validate_primary_reviews(reviews_path, identity_path)
    expected_pairs = {(item_id, 1) for item_id in SMOKE_ITEM_IDS}
    identity_pairs = {
        (str(identity["sample_id"]), identity["epoch"]) for identity in identities.values()
    }
    if (
        set(public_records) != set(identities)
        or len(identities) != len(SMOKE_ITEM_IDS)
        or identity_pairs != expected_pairs
    ):
        raise SmokeReviewError("blind review bundle does not cover the fixed five smoke items")
    samples_by_pair = {(str(sample.id), sample.epoch): sample for sample in samples}
    automatic_by_id = _validate_review_source_binding(
        log_file=log_info.name,
        run_id=log.eval.run_id,
        run_model=log.eval.model,
        samples_by_pair=samples_by_pair,
        identities=identities,
        review_bundle_dir=review_bundle_dir,
    )

    allowed_labels = {RootCause.SUT_ERROR.value}
    for review in reviews:
        if (
            review["human_task_verdict"] == HumanTaskVerdict.INDETERMINATE.value
            or review["sut_error"] == "indeterminate"
            or review["environment_valid"] is not True
            or not set(review["labels"]).issubset(allowed_labels)
        ):
            raise SmokeReviewError(
                "smoke review has an unresolved task, gold, grader, tool, environment, "
                "or infra defect"
            )
        automatic = automatic_by_id[review["blind_id"]]
        human_correct = review["human_task_verdict"] == HumanTaskVerdict.CORRECT.value
        if automatic["core_success"] is None or human_correct != (automatic["core_success"] == 1):
            raise SmokeReviewError(
                "smoke review and frozen grader disagree; resolve the grader or task before freeze"
            )

    log_path = _local_eval_log_path(log_info.name)
    hashes = _receipt_hashes(
        smoke_freeze_path=smoke_freeze_path,
        log_path=log_path,
        review_bundle_dir=review_bundle_dir,
        reviews_path=reviews_path,
    )
    evidence_hashes = {
        key: hashes[key]
        for key in (
            "smoke_freeze_sha256",
            "source_log_sha256",
            "review_bundle_manifest_sha256",
            "blind_items_sha256",
            "identity_map_sha256",
            "primary_reviews_sha256",
        )
    }
    receipt: dict[str, Any] = {
        "schema_version": SMOKE_RECEIPT_SCHEMA_VERSION,
        "profile_id": PROFILE_ID,
        "profile_version": PROFILE_VERSION,
        "smoke_freeze_id": freeze["freeze_id"],
        **hashes,
        "evidence_artifact_set_sha256": content_sha256(canonical_json_bytes(evidence_hashes)),
        "item_ids": list(SMOKE_ITEM_IDS),
        "item_count": len(SMOKE_ITEM_IDS),
        "epoch": 1,
        "eval_run_id": log.eval.run_id,
        "requested_model": freeze["requested_model"],
        "model_config_sha256": frozen_model_config_sha256(freeze["model_config"]),
        "container_image_id": freeze["container"]["image_id"],
        "artifact_set_sha256": freeze["artifact_set_sha256"],
        "git_commit": freeze["git"]["commit"],
        "quality_receipt_sha256": freeze["quality_receipt"]["receipt_sha256"],
        "gold_review_receipt_sha256": freeze["gold_review_receipt"]["receipt_sha256"],
        "source_cases_sha256": content_sha256((root / "data" / "cases.jsonl").read_bytes()),
        "source_gold_sha256": content_sha256((root / "data" / "gold.jsonl").read_bytes()),
        "reviewer_ids": sorted({review["reviewer_id"] for review in reviews}),
        "reviewer_identity_authenticated": False,
        "human_review_complete": True,
        "grader_visible": False,
        "grader_agreement_complete": True,
        "grader_agreement_count": len(SMOKE_ITEM_IDS),
        "all_environment_valid": True,
        "unresolved_defect_count": 0,
        "resolved_sut_error_count": sum(
            RootCause.SUT_ERROR.value in review["labels"] for review in reviews
        ),
    }
    receipt["receipt_sha256"] = content_sha256(canonical_json_bytes(receipt))
    return receipt


def verify_smoke_review_receipt_sources(
    receipt_path: Path,
    *,
    smoke_freeze_path: Path,
    log_dir: Path,
    review_bundle_dir: Path,
    reviews_path: Path,
    root: Path = PACKAGE_ROOT,
) -> dict[str, Any]:
    try:
        receipt = parse_strict_json_object(receipt_path.read_bytes(), max_bytes=8 * 1024 * 1024)
    except (OSError, StrictJSONError) as exc:
        raise SmokeReviewError("invalid smoke review receipt") from exc
    expected = build_smoke_review_receipt(
        smoke_freeze_path=smoke_freeze_path,
        log_dir=log_dir,
        review_bundle_dir=review_bundle_dir,
        reviews_path=reviews_path,
        root=root,
    )
    if receipt != expected:
        raise SmokeReviewError("smoke review receipt differs from its current source evidence")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description="Bind the five-item human smoke review")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("create", "verify"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--smoke-freeze", type=Path, required=True)
        subparser.add_argument("--log-dir", type=Path, required=True)
        subparser.add_argument("--review-bundle", type=Path, required=True)
        subparser.add_argument("--reviews", type=Path, required=True)
        if command == "create":
            subparser.add_argument("--output", type=Path, required=True)
        else:
            subparser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "create":
            receipt = build_smoke_review_receipt(
                smoke_freeze_path=args.smoke_freeze,
                log_dir=args.log_dir,
                review_bundle_dir=args.review_bundle,
                reviews_path=args.reviews,
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_bytes(canonical_json_bytes(receipt))
        else:
            receipt = verify_smoke_review_receipt_sources(
                args.receipt,
                smoke_freeze_path=args.smoke_freeze,
                log_dir=args.log_dir,
                review_bundle_dir=args.review_bundle,
                reviews_path=args.reviews,
            )
    except (
        SmokeReviewError,
        FreezeError,
        ReviewError,
        FinalReportError,
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        parser.error(str(exc))
    sys.stdout.buffer.write(canonical_json_bytes(receipt))


if __name__ == "__main__":
    main()
