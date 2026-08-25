from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from pydantic import ValidationError

from .canonical import canonical_json_bytes, content_sha256
from .constants import PACKAGE_ROOT, PROFILE_ID, PROFILE_VERSION
from .dataset import load_cases
from .diagnostics import run_diagnostics
from .freeze import FreezeError, verify_freeze_manifest
from .models import (
    SUT_TERMINAL_ERROR_GRADES,
    EpisodeOutcome,
    GradeResult,
    HumanTaskVerdict,
    RootCause,
)
from .review import (
    ReviewError,
    _automatic_grade,
    _safe_sample_payload,
    validate_adjudication_records,
    validate_primary_reviews,
    validate_secondary_reviews,
)
from .scorer import FAILURE_CODE_ORDER
from .strict_json import StrictJSONError, parse_strict_json_object

PROVIDER_CREDENTIAL_NAMES = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "AWS_ACCESS_KEY_ID",
)
QUALITY_GATE_COMMANDS = (
    ("uv", "run", "govagent-validate"),
    ("uv", "run", "govagent-diagnostics"),
    ("uv", "run", "ruff", "check", "."),
    ("uv", "run", "ruff", "format", "--check", "."),
    ("uv", "run", "pytest", "-q"),
)
ENVIRONMENT_ERROR_LABELS = {
    RootCause.TOOL_OR_ENV_ERROR.value,
    RootCause.INFRA_ERROR.value,
}
SUT_CONFOUND_LABELS = {
    RootCause.TASK_AMBIGUITY.value,
    RootCause.GOLD_ERROR.value,
    RootCause.GRADER_ERROR.value,
    *ENVIRONMENT_ERROR_LABELS,
}


def _load_json_object(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return parse_strict_json_object(path.read_bytes())
    except StrictJSONError as exc:
        raise ValueError(f"expected one strict JSON object: {path}") from exc


def _docker_status() -> dict[str, Any]:
    try:
        version = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            cwd=PACKAGE_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        images_raw = subprocess.run(
            ["docker", "compose", "images", "--format", "json"],
            cwd=PACKAGE_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        parsed_images = json.loads(images_raw)
        image_records = parsed_images if isinstance(parsed_images, list) else [parsed_images]
        images = sorted(
            {str(record.get("ID", "")) for record in image_records if isinstance(record, dict)}
            - {""}
        )
        return {"available": True, "server_version": version, "image_ids": images}
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        return {"available": False, "error_type": type(exc).__name__, "image_ids": []}


def _validate_current_quality_receipt(
    receipt: dict[str, Any] | None,
    *,
    root: Path,
) -> tuple[bool, str | None]:
    from .freeze import _container_record, _critical_files, _file_record, _git_value

    if not isinstance(receipt, dict):
        return False, "missing quality receipt"
    unsigned = dict(receipt)
    recorded_hash = unsigned.pop("receipt_sha256", None)
    if recorded_hash != content_sha256(canonical_json_bytes(unsigned)):
        return False, "quality receipt hash is invalid"
    if (
        receipt.get("schema_version") != "govagent-kr/quality-receipt-v0.1"
        or receipt.get("profile_id") != PROFILE_ID
        or receipt.get("profile_version") != PROFILE_VERSION
    ):
        return False, "quality receipt schema or profile differs"
    gates = receipt.get("gates")
    if not isinstance(gates, list) or len(gates) != len(QUALITY_GATE_COMMANDS):
        return False, "quality receipt gate set differs"
    for gate, command in zip(gates, QUALITY_GATE_COMMANDS, strict=True):
        if not isinstance(gate, dict) or gate.get("command") != list(command):
            return False, "quality receipt gate command differs"
        if type(gate.get("exit_code")) is not int or gate["exit_code"] != 0:
            return False, "quality receipt contains a failed gate"
        if not _sha256(gate.get("stdout_sha256")) or not _sha256(gate.get("stderr_sha256")):
            return False, "quality receipt gate output hash is invalid"
        if not isinstance(gate.get("stdout_tail"), str) or not isinstance(
            gate.get("stderr_tail"), str
        ):
            return False, "quality receipt gate output excerpt is invalid"
    if receipt.get("ok") is not True:
        return False, "quality receipt does not derive ok=true"
    if receipt.get("docker_smoke_included") is not True or receipt.get(
        "docker_smoke_compose"
    ) not in {"compose.yaml", "compose.frozen.yaml"}:
        return False, "quality receipt did not include a recognized Docker smoke"

    artifacts = [_file_record(path, root) for path in _critical_files(root)]
    artifact_hash = content_sha256(canonical_json_bytes(artifacts))
    if receipt.get("artifact_set_sha256") != artifact_hash:
        return False, "quality receipt is stale for the current artifacts"
    repository_root = Path(_git_value(root, "rev-parse", "--show-toplevel"))
    scoped_path = root.relative_to(repository_root).as_posix()
    current_commit = _git_value(repository_root, "rev-parse", "HEAD")
    current_status = _git_value(repository_root, "status", "--short", "--", scoped_path)
    git = receipt.get("git")
    if not isinstance(git, dict) or git != {
        "commit": current_commit,
        "gov_agent_worktree_status": current_status,
    }:
        return False, "quality receipt is stale for the current git state"

    analysis_claimed = receipt.get("analysis_quality_eligible") is True
    analysis_actual = False
    if (
        receipt.get("docker_smoke_included") is True
        and receipt.get("docker_smoke_compose") == "compose.frozen.yaml"
        and not current_status
    ):
        container = receipt.get("container")
        requested_image = container.get("requested_image") if isinstance(container, dict) else None
        if isinstance(requested_image, str):
            current_container = _container_record(requested_image)
            analysis_actual = bool(
                current_container is not None
                and requested_image == current_container.get("image_id")
                and container == current_container
            )
    if analysis_claimed != analysis_actual:
        return False, "quality receipt analysis eligibility is not derived from current evidence"
    return True, None


def build_readiness_report(
    *,
    root: Path = PACKAGE_ROOT,
    quality_receipt_path: Path | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    quality_path = quality_receipt_path or root / "reports" / "quality-receipt.json"
    quality = _load_json_object(quality_path)
    diagnostics = run_diagnostics(
        root / "fixtures" / "local-contract-decree",
        root / "data" / "cases.jsonl",
        root / "data" / "gold.jsonl",
    )
    try:
        quality_valid, quality_error = _validate_current_quality_receipt(quality, root=root)
    except (OSError, ValueError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        quality_valid = False
        quality_error = f"quality receipt verification failed: {type(exc).__name__}"
    credential_names = sorted(
        name for name in PROVIDER_CREDENTIAL_NAMES if bool(os.environ.get(name))
    )
    analysis_freezes: list[str] = []
    for path in sorted((root / "reports").glob("*freeze*.json")):
        try:
            verify_freeze_manifest(path, root=root, require_analysis=True)
        except (
            FreezeError,
            OSError,
            ValueError,
            subprocess.CalledProcessError,
            json.JSONDecodeError,
        ):
            continue
        else:
            analysis_freezes.append(path.relative_to(root).as_posix())
    frontier_logs = (
        sorted(
            path.relative_to(root).as_posix()
            for path in (root / "reports" / "runs").rglob("*.eval")
        )
        if (root / "reports" / "runs").exists()
        else []
    )
    review_outputs = sorted(
        path.relative_to(root).as_posix() for path in (root / "reports").rglob("revealed*.jsonl")
    )

    gates = {
        "engineering_diagnostics": diagnostics["engineering_ready"],
        "quality_receipt": quality_valid,
        "quality_receipt_error": quality_error,
        "docker": _docker_status(),
        "gold_two_person_adjudication": diagnostics["analysis_ready"],
        "provider_credential_names_present": credential_names,
        "analysis_freeze_receipts": analysis_freezes,
        "frontier_eval_log_count": len(frontier_logs),
        "revealed_review_file_count": len(review_outputs),
    }
    pending_actions: list[str] = []
    if not diagnostics["analysis_ready"]:
        pending_reviews = diagnostics["dataset_integrity"]["pending_human_review"]
        pending_actions.append(
            f"법률 전문가 2인이 후보 gold {pending_reviews}건을 독립 검토하고 adjudication한다."
        )
    if not credential_names:
        pending_actions.append(
            "고정할 frontier model의 provider credential을 실행 환경에 제공한다."
        )
    if not analysis_freezes:
        pending_actions.append("SME 검수와 5-item smoke 후 analysis freeze receipt를 만든다.")
    if len(frontier_logs) == 0:
        pending_actions.append("동결된 SUT로 20 items x 3 epochs의 frontier run을 실행한다.")
    if not review_outputs:
        pending_actions.append("60개 transcript를 blind 1차·2차 검토하고 grader 결과를 공개한다.")

    if diagnostics["engineering_ready"] and quality_valid:
        state = "engineering_environment_ready"
    else:
        state = "engineering_environment_not_ready"
    if not pending_actions:
        state = "evaluation_evidence_present_requires_final_validation"
    report: dict[str, Any] = {
        "schema_version": "govagent-kr/readiness-v0.1",
        "profile_id": PROFILE_ID,
        "profile_version": PROFILE_VERSION,
        "state": state,
        "not_a_certification": True,
        "diagnostics": diagnostics,
        "quality_receipt": quality,
        "gates": gates,
        "pending_actions": pending_actions,
    }
    report["report_sha256"] = content_sha256(canonical_json_bytes(report))
    return report


def readiness_markdown(report: dict[str, Any]) -> str:
    diagnostics = report["diagnostics"]
    gates = report["gates"]
    actions = "\n".join(f"- {item}" for item in report["pending_actions"]) or "- 없음"
    credential_state = (
        ", ".join(gates["provider_credential_names_present"])
        if gates["provider_credential_names_present"]
        else "없음"
    )
    oracle = diagnostics["fixture_oracle"]
    ablation = diagnostics["current_version_ablation"]
    return f"""# GovAgent-KR MVP Readiness

현재 상태: `{report["state"]}`

이 문서는 개발용 평가 환경의 준비 상태만 보고한다. 정부 인증, 법률 정확성 보증 또는
공공행정 투입 판정이 아니다.

## 자동 검증 결과

- fixture oracle: {oracle["successes"]}/{oracle["total"]}
- current-version ablation 실패: {ablation["failures"]}/{ablation["total"]}
- frozen replay byte-identical: {str(diagnostics["replay_byte_identical"]).lower()}
- engineering diagnostics: {str(diagnostics["engineering_ready"]).lower()}
- 현재 checkout에 결합된 quality receipt: {str(gates["quality_receipt"]).lower()}
- Docker 사용 가능: {str(gates["docker"]["available"]).lower()}
- gold 전문가 검수 완료: {str(diagnostics["analysis_ready"]).lower()}
- 감지된 provider credential 이름: {credential_state}
- frontier `.eval` 로그 수: {gates["frontier_eval_log_count"]}
- grader 공개까지 마친 review 파일 수: {gates["revealed_review_file_count"]}

## 남은 외부 작업

{actions}

단일 Overall/PASS는 계산하지 않는다. 실제 결과는 freeze 뒤 60회 실행과 blind transcript
검토가 모두 끝난 뒤 family별로 보고한다.
"""


class FinalReportError(RuntimeError):
    """Raised when evidence is incomplete or combines incompatible runs."""


EXPECTED_ITEM_COUNT = 20
EXPECTED_EPOCHS = (1, 2, 3)
REVIEW_COMPARISON_FIELDS = (
    "human_task_verdict",
    "environment_valid",
    "sut_error",
    "labels",
    "primary_cause",
)
IDENTITY_FIELDS = {
    "blind_id",
    "log_file",
    "eval_run_id",
    "sample_id",
    "epoch",
    "sample_uuid",
    "identity_hmac_sha256",
    "public_sample_sha256",
    "automatic_grade_sha256",
}
ADJUDICATION_FIELDS = {
    "blind_id",
    "adjudicator_id",
    "review_stage",
    "grader_visible",
    "human_task_verdict",
    "environment_valid",
    "sut_error",
    "labels",
    "primary_cause",
    "note",
}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            record = parse_strict_json_object(line)
        except StrictJSONError as exc:
            raise FinalReportError(f"invalid JSONL at {path}:{line_number}") from exc
        blind_id = record.get("blind_id")
        if isinstance(blind_id, str):
            if blind_id in seen_ids:
                raise FinalReportError(f"duplicate blind_id in {path}: {blind_id}")
            seen_ids.add(blind_id)
        records.append(record)
    return records


def _sample_grade(sample: Any) -> dict[str, Any]:
    if getattr(sample, "limit", None) is not None:
        raise FinalReportError(
            f"sample {sample.id}/{sample.epoch} was halted by an Inspect sample limit"
        )
    if sample.error is not None:
        error = sample.error.model_dump(mode="json")
        terminal = SUT_TERMINAL_ERROR_GRADES.get(sample.error.message)
        if terminal is not None:
            episode_outcome, failure_code = terminal
            return {
                "episode_outcome": episode_outcome.value,
                "attempt_state": "sut_terminal_failure",
                "error_classification": failure_code,
                "core_success": 0,
                "attempt_success": 0,
                "analysis_eligible": True,
                "failure_codes": [failure_code],
                "sample_error": error,
            }
        return {
            "episode_outcome": None,
            "attempt_state": "excluded_error",
            "error_classification": "unclassified",
            "core_success": None,
            "attempt_success": 0,
            "analysis_eligible": False,
            "failure_codes": [],
            "sample_error": error,
        }
    candidates: list[dict[str, Any]] = []
    for score in (sample.scores or {}).values():
        metadata = score.metadata or {}
        if "core_success" in metadata:
            candidates.append(metadata)
    if len(candidates) != 1:
        raise FinalReportError(f"sample {sample.id}/{sample.epoch} has no unique core score")
    metadata = candidates[0]
    try:
        grade = GradeResult.model_validate(metadata, strict=False)
    except ValidationError as exc:
        raise FinalReportError(
            f"sample {sample.id}/{sample.epoch} has invalid grade metadata"
        ) from exc
    normalized = grade.model_dump(mode="json")
    if canonical_json_bytes(normalized) != canonical_json_bytes(metadata):
        raise FinalReportError(
            f"sample {sample.id}/{sample.epoch} grade metadata required coercion"
        )
    if (
        grade.episode_outcome != EpisodeOutcome.COMPLETED
        or grade.analysis_eligible is not True
        or grade.environment_valid is not True
        or grade.core_success not in {0, 1}
        or grade.attempt_success != grade.core_success
    ):
        raise FinalReportError(
            f"sample {sample.id}/{sample.epoch} has inconsistent completed grade metadata"
        )
    expected_codes = [code for code in FAILURE_CODE_ORDER if code in grade.failure_codes]
    if expected_codes != grade.failure_codes or bool(grade.failure_codes) == bool(
        grade.core_success
    ):
        raise FinalReportError(f"sample {sample.id}/{sample.epoch} has inconsistent failure codes")
    return {
        "episode_outcome": grade.episode_outcome.value,
        "attempt_state": "completed",
        "error_classification": None,
        "core_success": grade.core_success,
        "attempt_success": grade.attempt_success,
        "analysis_eligible": grade.analysis_eligible,
        "failure_codes": grade.failure_codes,
        "sample_error": None,
    }


def _sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _evidence_artifact_records(entries: dict[str, Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for role, path in sorted(entries.items()):
        if not path.is_file():
            raise FinalReportError(f"missing final-report evidence artifact: {role}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        records.append(
            {
                "role": role,
                "sha256": digest.hexdigest(),
                "size_bytes": path.stat().st_size,
            }
        )
    return records


def _local_eval_log_path(name: str) -> Path:
    parsed = urlparse(name)
    if parsed.scheme == "file" and parsed.netloc in {"", "localhost"}:
        return Path(unquote(parsed.path))
    if parsed.scheme:
        raise FinalReportError("final report requires a local file-backed Inspect log")
    return Path(name)


def _load_identity_map(path: Path) -> dict[str, dict[str, Any]]:
    records = _load_jsonl(path)
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        if set(record) != IDENTITY_FIELDS:
            raise FinalReportError("identity map fields differ from the frozen review schema")
        blind_id = record.get("blind_id")
        sample_id = record.get("sample_id")
        epoch = record.get("epoch")
        if not isinstance(blind_id, str) or type(sample_id) not in {str, int}:
            raise FinalReportError("identity map has invalid blind_id or sample_id")
        if type(epoch) is not int or epoch not in EXPECTED_EPOCHS:
            raise FinalReportError("identity map has invalid epoch")
        required_strings = ("log_file", "eval_run_id", "sample_uuid")
        if any(
            not isinstance(record.get(field), str) or not record[field]
            for field in required_strings
        ):
            raise FinalReportError("identity map has an invalid source-log identity")
        hash_fields = (
            "identity_hmac_sha256",
            "public_sample_sha256",
            "automatic_grade_sha256",
        )
        if any(not _sha256(record.get(field)) for field in hash_fields):
            raise FinalReportError("identity map has an invalid evidence hash")
        result[blind_id] = record
    return result


def _validate_review_record(record: dict[str, Any], *, context: str) -> None:
    try:
        verdict = HumanTaskVerdict(record.get("human_task_verdict"))
    except ValueError as exc:
        raise FinalReportError(f"{context}: invalid human_task_verdict") from exc
    environment_valid = record.get("environment_valid")
    if type(environment_valid) is not bool:
        raise FinalReportError(f"{context}: invalid environment_valid")
    sut_error = record.get("sut_error")
    if type(sut_error) is not bool and sut_error != "indeterminate":
        raise FinalReportError(f"{context}: invalid sut_error")
    labels = record.get("labels")
    allowed_labels = {label.value for label in RootCause}
    if (
        not isinstance(labels, list)
        or any(type(label) is not str or label not in allowed_labels for label in labels)
        or len(labels) != len(set(labels))
    ):
        raise FinalReportError(f"{context}: invalid labels")
    if RootCause.GRADER_ERROR.value in labels:
        raise FinalReportError(f"{context}: grader-blind review cannot assign GRADER_ERROR")
    if verdict == HumanTaskVerdict.INCORRECT and not labels:
        raise FinalReportError(f"{context}: incorrect verdict requires a cause label")
    environment_labels = ENVIRONMENT_ERROR_LABELS.intersection(labels)
    if environment_valid and environment_labels:
        raise FinalReportError(
            f"{context}: environment error labels require environment_valid=false"
        )
    if not environment_valid and not environment_labels:
        raise FinalReportError(
            f"{context}: environment_valid=false requires an environment error label"
        )
    if sut_error is True and RootCause.SUT_ERROR.value not in labels:
        raise FinalReportError(f"{context}: sut_error=true requires SUT_ERROR label")
    if sut_error is not True and RootCause.SUT_ERROR.value in labels:
        raise FinalReportError(f"{context}: SUT_ERROR label requires sut_error=true")
    if sut_error is True and (
        verdict != HumanTaskVerdict.INCORRECT or environment_valid is not True
    ):
        raise FinalReportError(
            f"{context}: sut_error=true requires incorrect verdict and valid environment"
        )
    primary = record.get("primary_cause")
    if primary is not None and primary not in labels:
        raise FinalReportError(f"{context}: primary_cause is not in labels")
    if labels and primary is None:
        raise FinalReportError(f"{context}: nonempty labels require primary_cause")
    if not labels and primary is not None:
        raise FinalReportError(f"{context}: empty labels require blank primary_cause")


def _validate_adjudications(
    path: Path | None,
    disagreement_ids: set[str],
    primary_by_id: dict[str, dict[str, Any]],
    secondary_by_id: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    if not disagreement_ids:
        if path is not None and _load_jsonl(path):
            raise FinalReportError("adjudications were supplied without any disagreement")
        return {}
    if path is None:
        raise FinalReportError("secondary-review disagreements require human adjudication")
    records = _load_jsonl(path)
    try:
        validated = validate_adjudication_records(
            records,
            disagreement_ids,
            primary_by_id,
            secondary_by_id,
        )
    except ReviewError as exc:
        raise FinalReportError(str(exc)) from exc
    return {record["blind_id"]: record for record in validated}


def _expected_disagreements(
    primary_by_id: dict[str, dict[str, Any]],
    secondary: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    disagreements: list[dict[str, Any]] = []
    for secondary_review in secondary:
        blind_id = secondary_review["blind_id"]
        primary_review = primary_by_id[blind_id]
        differing_fields: list[str] = []
        for field in REVIEW_COMPARISON_FIELDS:
            primary_value = (
                sorted(primary_review[field]) if field == "labels" else primary_review[field]
            )
            secondary_value = (
                sorted(secondary_review[field]) if field == "labels" else secondary_review[field]
            )
            if primary_value != secondary_value:
                differing_fields.append(field)
        if differing_fields:
            disagreements.append(
                {
                    "blind_id": blind_id,
                    "differing_fields": differing_fields,
                    "primary_review": primary_review,
                    "secondary_review": secondary_review,
                }
            )
    return disagreements


def _validate_disagreement_artifact(
    path: Path,
    primary_by_id: dict[str, dict[str, Any]],
    secondary: list[dict[str, Any]],
) -> set[str]:
    expected_disagreements = _expected_disagreements(primary_by_id, secondary)
    expected_document = {
        "schema_version": "govagent-kr/review-disagreements-v0.1",
        "automatic_adjudication": False,
        "disagreements": expected_disagreements,
    }
    if _load_json_object(path) != expected_document:
        raise FinalReportError("missing or invalid secondary disagreement artifact")
    return {record["blind_id"] for record in expected_disagreements}


def _validate_frozen_run(
    log: Any,
    freeze: dict[str, Any],
    samples: list[Any],
    case_by_id: dict[str, Any],
) -> set[tuple[str, int]]:
    from .task import (
        DEFAULT_FROZEN_COMPOSE_PATH,
        MAX_MODEL_REQUESTS,
        MAX_TOOL_CALLS,
        REQUIRED_GENERATE_CONFIG,
        frozen_model_config_sha256,
        normalize_frozen_model_config,
    )

    if log.status != "success" or log.invalidated or log.error is not None:
        raise FinalReportError("Inspect log is not a valid successful run")
    if len(case_by_id) != EXPECTED_ITEM_COUNT:
        raise FinalReportError("the frozen dataset is not exactly 20 unique items")
    expected_pairs = {(item_id, epoch) for item_id in case_by_id for epoch in EXPECTED_EPOCHS}
    actual_pairs = {(str(sample.id), sample.epoch) for sample in samples}
    if len(samples) != len(expected_pairs) or actual_pairs != expected_pairs:
        raise FinalReportError("Inspect log is not exactly 20 items x 3 epochs")
    sample_uuids = [sample.uuid for sample in samples]
    if (
        any(not isinstance(value, str) or not value for value in sample_uuids)
        or len(set(sample_uuids)) != len(sample_uuids)
        or any(sample.invalidation is not None for sample in samples)
    ):
        raise FinalReportError("Inspect samples have missing, duplicate, or invalidated UUIDs")
    if any(sample.error_retries for sample in samples):
        raise FinalReportError("sample retry history is not allowed in the frozen run")
    if freeze.get("profile_id") != PROFILE_ID or freeze.get("profile_version") != PROFILE_VERSION:
        raise FinalReportError("freeze profile differs from the final-report profile")
    if (
        freeze.get("freeze_kind") != "analysis_freeze"
        or freeze.get("analysis_eligible") is not True
    ):
        raise FinalReportError("freeze is not a valid analysis freeze")
    if log.eval.model != freeze.get("requested_model"):
        raise FinalReportError("Inspect requested model differs from the freeze")
    if log.eval.task != "govagent_kr" or log.eval.task_version != PROFILE_VERSION:
        raise FinalReportError("Inspect task identity or version differs from the freeze")
    task_file = log.eval.task_file
    if isinstance(task_file, str):
        resolved_task_file = Path(task_file)
        if not resolved_task_file.is_absolute():
            resolved_task_file = PACKAGE_ROOT / resolved_task_file
        resolved_task_file = resolved_task_file.resolve()
    else:
        resolved_task_file = None
    if (
        resolved_task_file != (PACKAGE_ROOT / "src/govagent_kr/task.py").resolve()
        or log.eval.task_registry_name != "govagent_kr"
        or log.eval.solver is not None
        or log.eval.solver_args is not None
        or log.eval.solver_args_passed != {}
    ):
        raise FinalReportError("Inspect task file, registry, or solver override differs")
    dataset = log.eval.dataset
    expected_item_ids = list(case_by_id)
    if (
        dataset.name != "govagent-kr-v0.1"
        or dataset.location is not None
        or dataset.samples != EXPECTED_ITEM_COUNT
        or dataset.sample_ids != expected_item_ids
        or dataset.shuffled is not False
    ):
        raise FinalReportError("Inspect dataset identity differs from the frozen dataset")
    config = log.eval.config
    if (
        config.limit is not None
        or config.sample_id is not None
        or config.sample_shuffle is not None
        or config.epochs != len(EXPECTED_EPOCHS)
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
        or config.time_limit != 180
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
        or config.log_shared is not None
        or config.score_display is not True
        or config.sandbox_cleanup is not True
        or config.acp_server is not None
    ):
        raise FinalReportError("Inspect run controls differ from the frozen protocol")
    if (
        log.plan.name != "plan"
        or log.plan.finish is not None
        or len(log.plan.steps) != 1
        or log.plan.steps[0].solver != "govagent_sandbox_agent"
        or log.plan.steps[0].params
        != {
            "max_model_requests": 12,
            "max_tool_calls": 24,
            "timeout_seconds": 175,
        }
    ):
        raise FinalReportError("Inspect solver plan differs from the frozen agent")
    sandbox_spec = log.eval.sandbox
    if (
        sandbox_spec is None
        or sandbox_spec.type != "docker"
        or not isinstance(sandbox_spec.config, str)
        or Path(sandbox_spec.config).resolve() != DEFAULT_FROZEN_COMPOSE_PATH.resolve()
    ):
        raise FinalReportError("Inspect run did not use the frozen Docker sandbox")
    scorers = log.eval.scorers or []
    if (
        len(scorers) != 1
        or scorers[0].name != "govagent_strict_scorer"
        or scorers[0].options
        != {"fixture_root": str((PACKAGE_ROOT / "fixtures/local-contract-decree").resolve())}
    ):
        raise FinalReportError("Inspect run did not use the frozen scorer")
    revision = log.eval.revision
    frozen_commit = freeze.get("git", {}).get("commit")
    if (
        revision is None
        or revision.type != "git"
        or type(revision.dirty) is not bool
        or not isinstance(frozen_commit, str)
        or not isinstance(revision.commit, str)
        or not 7 <= len(revision.commit) <= 40
        or any(character not in "0123456789abcdef" for character in revision.commit)
        or not frozen_commit.startswith(revision.commit)
    ):
        raise FinalReportError("Inspect source revision differs from the freeze")
    if log.eval.packages.get("inspect_ai") != freeze.get("runtime", {}).get("inspect_ai"):
        raise FinalReportError("Inspect package version differs from the freeze")

    try:
        expected_model_config = normalize_frozen_model_config(freeze.get("model_config"))
        expected_model_config_hash = frozen_model_config_sha256(freeze.get("model_config"))
    except ValueError as exc:
        raise FinalReportError("freeze has an invalid model_config") from exc
    if any(
        expected_model_config["generate"].get(key) != value
        for key, value in REQUIRED_GENERATE_CONFIG.items()
    ):
        raise FinalReportError("freeze does not fix the required Inspect model controls")
    effective_generate = log.plan.config.model_dump(mode="json", exclude_none=True)
    model_generate = log.eval.model_generate_config.model_dump(mode="json", exclude_none=True)
    if effective_generate != expected_model_config["generate"] or any(
        expected_model_config["generate"].get(key) != value for key, value in model_generate.items()
    ):
        raise FinalReportError("Inspect generation settings differ from the freeze")
    if log.eval.model_args != expected_model_config["model_args"]:
        raise FinalReportError("Inspect model arguments differ from the freeze")
    if log.eval.model_base_url is not None or log.eval.model_roles:
        raise FinalReportError("Inspect model base URL or model roles are not permitted")
    if any(sample.model_fallbacks for sample in samples):
        raise FinalReportError("Inspect model fallback use is not permitted")
    if any(
        sample.token_limit is not None
        or sample.token_limit_type is not None
        or sample.token_limit_usage is not None
        for sample in samples
    ):
        raise FinalReportError("sample-specific token limits are not permitted")

    eval_metadata = log.eval.metadata or {}
    container = freeze.get("container")
    frozen_image_id = container.get("image_id") if isinstance(container, dict) else None
    expected_metadata = {
        "freeze_id": freeze.get("freeze_id"),
        "analysis_eligible": True,
        "candidate_gold_allowed": False,
        "frozen_requested_model": freeze.get("requested_model"),
        "frozen_model_config": expected_model_config,
        "frozen_model_config_sha256": expected_model_config_hash,
        "frozen_container_image_id": frozen_image_id,
        "model_request_budget": MAX_MODEL_REQUESTS,
        "tool_call_budget": MAX_TOOL_CALLS,
        "network": "none",
        "provider_web_search": False,
        "provider_code_execution": False,
        "remote_mcp": False,
    }
    if any(eval_metadata.get(key) != value for key, value in expected_metadata.items()):
        raise FinalReportError("Inspect task metadata differs from the analysis freeze")
    return expected_pairs


def _frozen_input_value(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(
            mode="json",
            exclude={"id", "source"},
            exclude_none=True,
        )
    if isinstance(value, list):
        return [_frozen_input_value(item) for item in value]
    return value


def _validate_samples_and_replay(samples: list[Any]) -> None:
    from .constants import DEFAULT_CASES_PATH, DEFAULT_FIXTURE_ROOT, DEFAULT_GOLD_PATH
    from .dataset import load_gold
    from .fixtures import FixtureStore
    from .scorer import grade_submission
    from .task import _samples

    expected_samples = _samples(
        cases_path=DEFAULT_CASES_PATH.resolve(),
        gold_path=DEFAULT_GOLD_PATH.resolve(),
        allow_pending_gold=False,
    )
    expected_by_id = {str(sample.id): sample for sample in expected_samples}
    gold_by_id = {record.item_id: record for record in load_gold()}
    store = FixtureStore(DEFAULT_FIXTURE_ROOT)
    for sample in samples:
        item_id = str(sample.id)
        expected = expected_by_id[item_id]
        if (
            _frozen_input_value(sample.input) != _frozen_input_value(expected.input)
            or sample.target != expected.target
            or sample.metadata != expected.metadata
        ):
            raise FinalReportError(f"Inspect sample differs from the frozen dataset: {item_id}")
        if sample.error is not None:
            if sample.scores:
                raise FinalReportError(
                    f"errored sample unexpectedly contains an automatic score: {item_id}"
                )
            continue
        scores = sample.scores or {}
        if set(scores) != {"govagent_strict_scorer"}:
            raise FinalReportError(f"sample does not have exactly one frozen score: {item_id}")
        score = scores["govagent_strict_scorer"]
        replayed = grade_submission(sample.output.completion, gold_by_id[item_id], store)
        expected_metadata = replayed.model_dump(mode="json")
        expected_explanation = (
            "pass" if replayed.core_success == 1 else ",".join(replayed.failure_codes)
        )
        if (
            canonical_json_bytes(score.metadata) != canonical_json_bytes(expected_metadata)
            or score.value != replayed.core_success
            or score.answer != sample.output.completion
            or score.explanation != expected_explanation
            or score.history
        ):
            raise FinalReportError(f"frozen scorer replay differs from the log: {item_id}")


def _identity_pair(identity: dict[str, Any]) -> tuple[str, int]:
    return (str(identity["sample_id"]), identity["epoch"])


def _validate_review_source_binding(
    *,
    log_file: str,
    run_id: str,
    run_model: str,
    samples_by_pair: dict[tuple[str, int], Any],
    identities: dict[str, dict[str, Any]],
    review_bundle_dir: Path,
) -> dict[str, dict[str, Any]]:
    public_records: dict[str, dict[str, Any]] = {}
    for record in _load_jsonl(review_bundle_dir / "blind-items.jsonl"):
        blind_id = record.get("blind_id")
        if not isinstance(blind_id, str):
            raise FinalReportError("public review record has no valid blind_id")
        public_records[blind_id] = record
    if set(public_records) != set(identities):
        raise FinalReportError("public review bundle and identity map differ")
    automatic_by_id: dict[str, dict[str, Any]] = {}
    for blind_id, identity in identities.items():
        pair = _identity_pair(identity)
        sample = samples_by_pair[pair]
        if (
            identity["log_file"] != log_file
            or identity["eval_run_id"] != run_id
            or identity["sample_uuid"] != sample.uuid
        ):
            raise FinalReportError(f"review source identity differs from Inspect log: {blind_id}")
        public = _safe_sample_payload(sample, blind_id, run_model)
        if public_records[blind_id] != public or identity["public_sample_sha256"] != content_sha256(
            canonical_json_bytes(public)
        ):
            raise FinalReportError(f"review transcript differs from Inspect log: {blind_id}")
        automatic = _automatic_grade(sample)
        if identity["automatic_grade_sha256"] != content_sha256(canonical_json_bytes(automatic)):
            raise FinalReportError(f"review automatic grade differs from Inspect log: {blind_id}")
        automatic_by_id[blind_id] = automatic
    return automatic_by_id


def _grader_agreement(review: dict[str, Any], automatic: dict[str, Any]) -> bool | str:
    if (
        review["human_task_verdict"] == HumanTaskVerdict.INDETERMINATE.value
        or not review["environment_valid"]
        or automatic["core_success"] is None
    ):
        return "not_applicable"
    human_correct = review["human_task_verdict"] == HumanTaskVerdict.CORRECT.value
    return human_correct == (automatic["core_success"] == 1)


def _validate_revealed_reviews(
    path: Path,
    primary_by_id: dict[str, dict[str, Any]],
    automatic_by_id: dict[str, dict[str, Any]],
) -> None:
    revealed = _load_jsonl(path)
    revealed_by_id = {record.get("blind_id"): record for record in revealed}
    if set(revealed_by_id) != set(primary_by_id):
        raise FinalReportError("revealed grader records do not cover all 60 runs")
    for blind_id, primary_review in primary_by_id.items():
        automatic = automatic_by_id[blind_id]
        expected = {
            "blind_id": blind_id,
            "primary_review": primary_review,
            "automatic_grade": automatic,
            "grader_agrees": _grader_agreement(primary_review, automatic),
        }
        if revealed_by_id[blind_id] != expected:
            raise FinalReportError(f"revealed review evidence differs from source: {blind_id}")


def _genuine_sut_error(review: dict[str, Any], grade: dict[str, Any]) -> bool:
    return (
        review["human_task_verdict"] == HumanTaskVerdict.INCORRECT.value
        and review["environment_valid"] is True
        and review["sut_error"] is True
        and RootCause.SUT_ERROR.value in review["labels"]
        and review["primary_cause"] == RootCause.SUT_ERROR.value
        and not SUT_CONFOUND_LABELS.intersection(review["labels"])
        and grade["analysis_eligible"] is True
    )


def _run_result_record(
    *,
    blind_id: str,
    identity: dict[str, Any],
    family_id: str,
    grade: dict[str, Any],
    human_review: dict[str, Any],
    automatic_grade: dict[str, Any],
) -> dict[str, Any]:
    reviewer_id = human_review.get("reviewer_id", human_review.get("adjudicator_id"))
    return {
        "blind_id": blind_id,
        "item_id": str(identity["sample_id"]),
        "epoch": identity["epoch"],
        "family_id": family_id,
        "episode_outcome": grade["episode_outcome"],
        "attempt_state": grade["attempt_state"],
        "error_classification": grade["error_classification"],
        "analysis_eligible": grade["analysis_eligible"],
        "attempt_success": grade["attempt_success"],
        "core_success": grade["core_success"],
        "failure_codes": list(grade["failure_codes"]),
        "human_task_verdict": human_review["human_task_verdict"],
        "environment_valid": human_review["environment_valid"],
        "sut_error": human_review["sut_error"],
        "labels": list(human_review["labels"]),
        "primary_cause": human_review["primary_cause"],
        "review_stage": human_review.get("review_stage", "adjudication"),
        "reviewer_id": reviewer_id,
        "review_note": human_review["note"],
        "grader_agrees": _grader_agreement(human_review, automatic_grade),
    }


def _validate_decision_gate(
    *,
    decision: str,
    analysis_eligible_attempts: int,
    environment_invalid_runs: int,
    unresolved_human_runs: int,
    sut_error_family_count: int,
    grader_disagreement_runs: int,
    design_error_runs: int,
    genuine_sut_error_runs: int,
) -> None:
    if decision != "go_expand_eval":
        return
    if analysis_eligible_attempts != 60 or environment_invalid_runs != 0:
        raise FinalReportError(
            "go_expand_eval requires all 60 attempts to be analysis-eligible and "
            "human-confirmed environment-valid"
        )
    if unresolved_human_runs != 0:
        raise FinalReportError(
            "go_expand_eval requires resolved human verdict and SUT attribution for all runs"
        )
    if sut_error_family_count < 2:
        raise FinalReportError("go_expand_eval requires genuine SUT_ERROR in at least two families")
    if grader_disagreement_runs >= 2:
        raise FinalReportError("go_expand_eval is blocked by repeated grader disagreements")
    if design_error_runs > genuine_sut_error_runs:
        raise FinalReportError(
            "go_expand_eval is blocked because task, gold, or grader errors exceed SUT errors"
        )


def build_final_report(
    *,
    freeze_path: Path,
    log_dir: Path,
    review_bundle_dir: Path,
    primary_review_csv: Path,
    secondary_bundle_dir: Path,
    secondary_review_csv: Path,
    revealed_reviews_path: Path,
    disagreements_path: Path,
    adjudications_path: Path | None,
    decision: str,
    decision_note: str,
) -> dict[str, Any]:
    from inspect_ai.log import list_eval_logs, read_eval_log

    if decision not in {"go_expand_eval", "redesign", "stop"}:
        raise FinalReportError(f"unsupported decision: {decision}")
    if not decision_note.strip():
        raise FinalReportError("a human decision note is required")
    freeze = verify_freeze_manifest(freeze_path, require_analysis=True)
    log_infos = list_eval_logs(str(log_dir), formats=["eval"])
    if len(log_infos) != 1:
        raise FinalReportError("final report requires exactly one frozen Inspect log")
    log_info = log_infos[0]
    log = read_eval_log(log_info)
    samples = log.samples or []
    cases = load_cases()
    case_by_id = {case.item_id: case for case in cases}
    if len(case_by_id) != len(cases):
        raise FinalReportError("frozen cases contain duplicate item IDs")
    expected_pairs = _validate_frozen_run(log, freeze, samples, case_by_id)
    _validate_samples_and_replay(samples)

    grades = {(str(sample.id), sample.epoch): _sample_grade(sample) for sample in samples}
    samples_by_pair = {(str(sample.id), sample.epoch): sample for sample in samples}
    identity_path = review_bundle_dir / "private" / "identity-map.jsonl"
    identities = _load_identity_map(identity_path)
    identity_pairs = {_identity_pair(identity) for identity in identities.values()}
    if identity_pairs != expected_pairs or len(identities) != len(expected_pairs):
        raise FinalReportError("review identity map does not cover the frozen 60-run set")
    primary = validate_primary_reviews(primary_review_csv, identity_path)
    secondary = validate_secondary_reviews(
        secondary_review_csv,
        primary_review_csv,
        review_bundle_dir,
        secondary_bundle_dir,
    )
    primary_by_id = {record["blind_id"]: record for record in primary}
    secondary_by_id = {record["blind_id"]: record for record in secondary}
    for record in [*primary, *secondary]:
        blind_id = record["blind_id"]
        _validate_review_record(record, context=f"{blind_id}: {record['review_stage']}")
        if not isinstance(record.get("note"), str) or not record["note"].strip():
            raise FinalReportError(f"{blind_id}: every transcript review requires a note")

    automatic_by_id = _validate_review_source_binding(
        log_file=log_info.name,
        run_id=log.eval.run_id,
        run_model=log.eval.model,
        samples_by_pair=samples_by_pair,
        identities=identities,
        review_bundle_dir=review_bundle_dir,
    )

    disagreement_ids = _validate_disagreement_artifact(disagreements_path, primary_by_id, secondary)
    adjudications = _validate_adjudications(
        adjudications_path,
        disagreement_ids,
        primary_by_id,
        secondary_by_id,
    )

    _validate_revealed_reviews(revealed_reviews_path, primary_by_id, automatic_by_id)

    final_reviews: dict[str, dict[str, Any]] = dict(primary_by_id)
    for blind_id, record in secondary_by_id.items():
        if blind_id not in disagreement_ids:
            final_reviews[blind_id] = record
    final_reviews.update(adjudications)

    root_causes: Counter[str] = Counter()
    family_summaries: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "attempts": 0,
            "attempt_successes": 0,
            "core_successes": 0,
            "analysis_eligible_attempts": 0,
            "excluded_errors": 0,
            "sut_error_runs": 0,
            "human_correct": 0,
            "human_incorrect": 0,
            "human_indeterminate": 0,
            "grader_agreement_applicable": 0,
            "grader_agreements": 0,
            "grader_disagreements": 0,
        }
    )
    sut_error_families: set[str] = set()
    episode_outcomes: Counter[str] = Counter()
    error_classifications: Counter[str] = Counter()
    human_verdicts: Counter[str] = Counter()
    grader_agreements: Counter[str] = Counter()
    genuine_sut_error_runs = 0
    grader_false_positive_runs = 0
    grader_false_negative_runs = 0
    grader_disagreement_runs = 0
    design_error_runs = 0
    design_error_labels = {
        RootCause.TASK_AMBIGUITY.value,
        RootCause.GOLD_ERROR.value,
        RootCause.GRADER_ERROR.value,
    }
    for blind_id, identity in identities.items():
        pair = _identity_pair(identity)
        item_id, _epoch = pair
        family_id = case_by_id[item_id].family_id
        grade = grades[pair]
        summary = family_summaries[family_id]
        summary["attempts"] += 1
        summary["attempt_successes"] += grade["attempt_success"]
        summary["core_successes"] += int(grade["core_success"] == 1)
        summary["analysis_eligible_attempts"] += int(grade["analysis_eligible"] is True)
        summary["excluded_errors"] += int(grade["analysis_eligible"] is False)
        if grade["episode_outcome"] is not None:
            episode_outcomes[grade["episode_outcome"]] += 1
        if grade["error_classification"] is not None:
            error_classifications[grade["error_classification"]] += 1
        review = final_reviews[blind_id]
        human_verdicts[review["human_task_verdict"]] += 1
        agreement = _grader_agreement(review, automatic_by_id[blind_id])
        agreement_key = (
            "not_applicable"
            if agreement == "not_applicable"
            else "agree"
            if agreement is True
            else "disagree"
        )
        grader_agreements[agreement_key] += 1
        summary[f"human_{review['human_task_verdict']}"] += 1
        if agreement != "not_applicable":
            summary["grader_agreement_applicable"] += 1
            summary["grader_agreements"] += int(agreement is True)
            summary["grader_disagreements"] += int(agreement is False)
        for label in review["labels"]:
            root_causes[label] += 1
        if design_error_labels.intersection(review["labels"]) or agreement is False:
            design_error_runs += 1
        if _genuine_sut_error(review, grade):
            summary["sut_error_runs"] += 1
            sut_error_families.add(family_id)
            genuine_sut_error_runs += 1
        if (
            review["human_task_verdict"] == HumanTaskVerdict.INCORRECT.value
            and review["environment_valid"] is True
            and grade["core_success"] == 1
        ):
            grader_false_positive_runs += 1
        if (
            review["human_task_verdict"] == HumanTaskVerdict.CORRECT.value
            and review["environment_valid"] is True
            and grade["core_success"] == 0
        ):
            grader_false_negative_runs += 1
        grader_disagreement_runs += int(agreement is False)

    _validate_decision_gate(
        decision=decision,
        analysis_eligible_attempts=sum(
            grade["analysis_eligible"] is True for grade in grades.values()
        ),
        environment_invalid_runs=sum(
            review["environment_valid"] is not True for review in final_reviews.values()
        ),
        unresolved_human_runs=sum(
            review["human_task_verdict"] == HumanTaskVerdict.INDETERMINATE.value
            or review["sut_error"] == "indeterminate"
            for review in final_reviews.values()
        ),
        sut_error_family_count=len(sut_error_families),
        grader_disagreement_runs=grader_disagreement_runs,
        design_error_runs=design_error_runs,
        genuine_sut_error_runs=genuine_sut_error_runs,
    )
    run_results = [
        _run_result_record(
            blind_id=blind_id,
            identity=identity,
            family_id=case_by_id[str(identity["sample_id"])].family_id,
            grade=grades[_identity_pair(identity)],
            human_review=final_reviews[blind_id],
            automatic_grade=automatic_by_id[blind_id],
        )
        for blind_id, identity in sorted(
            identities.items(), key=lambda item: _identity_pair(item[1])
        )
    ]
    evidence_paths = {
        "analysis_freeze": freeze_path,
        "blind_items": review_bundle_dir / "blind-items.jsonl",
        "disagreements": disagreements_path,
        "identity_map": identity_path,
        "inspect_eval_log": _local_eval_log_path(log_info.name),
        "primary_review_csv": primary_review_csv,
        "primary_review_manifest": review_bundle_dir / "manifest.json",
        "revealed_reviews": revealed_reviews_path,
        "secondary_items": secondary_bundle_dir / "secondary-items.jsonl",
        "secondary_review_csv": secondary_review_csv,
        "secondary_review_manifest": secondary_bundle_dir / "manifest.json",
    }
    if adjudications_path is not None:
        evidence_paths["adjudications"] = adjudications_path
    report: dict[str, Any] = {
        "schema_version": "govagent-kr/final-result-v0.1",
        "profile_id": PROFILE_ID,
        "profile_version": PROFILE_VERSION,
        "not_a_certification": True,
        "freeze_id": freeze["freeze_id"],
        "requested_model": freeze["requested_model"],
        "inspect_returned_model_ids": sorted(
            {
                event.output.model
                for sample in samples
                for event in sample.events
                if getattr(event, "event", None) == "model"
                and getattr(event, "output", None) is not None
                and getattr(event.output, "model", None)
            }
        ),
        "attempts": 60,
        "attempt_successes": sum(grade["attempt_success"] for grade in grades.values()),
        "core_successes": sum(grade["core_success"] == 1 for grade in grades.values()),
        "analysis_eligible_attempts": sum(
            grade["analysis_eligible"] is True for grade in grades.values()
        ),
        "excluded_error_attempts": sum(
            grade["analysis_eligible"] is False for grade in grades.values()
        ),
        "episode_outcome_counts": dict(sorted(episode_outcomes.items())),
        "error_classification_counts": dict(sorted(error_classifications.items())),
        "family_results": dict(sorted(family_summaries.items())),
        "root_cause_counts": dict(sorted(root_causes.items())),
        "human_verdict_counts": dict(sorted(human_verdicts.items())),
        "grader_agreement_counts": dict(sorted(grader_agreements.items())),
        "grader_false_positive_runs": grader_false_positive_runs,
        "grader_false_negative_runs": grader_false_negative_runs,
        "grader_disagreement_runs": grader_disagreement_runs,
        "design_error_runs": design_error_runs,
        "genuine_sut_error_runs": genuine_sut_error_runs,
        "sut_error_families": sorted(sut_error_families),
        "primary_reviewed_runs": len(primary),
        "secondary_reviewed_runs": len(secondary),
        "human_adjudicated_disagreements": len(adjudications),
        "run_results": run_results,
        "evidence_artifacts": _evidence_artifact_records(evidence_paths),
        "decision": decision,
        "decision_note": decision_note,
        "decision_scope": "expand this eval only; never an RL or deployment approval",
    }
    report["report_sha256"] = content_sha256(canonical_json_bytes(report))
    return report


def final_markdown(report: dict[str, Any]) -> str:
    rows = "\n".join(
        "| {family} | {attempts} | {successes} | {infra} | {sut} |".format(
            family=family,
            attempts=value["attempts"],
            successes=value["core_successes"],
            infra=value["excluded_errors"],
            sut=value["sut_error_runs"],
        )
        for family, value in report["family_results"].items()
    )
    causes = (
        "\n".join(f"- {label}: {count}" for label, count in report["root_cause_counts"].items())
        or "- 없음"
    )
    human = report["human_verdict_counts"]
    agreement = report["grader_agreement_counts"]
    human_summary = " / ".join(
        str(human.get(value, 0)) for value in ("correct", "incorrect", "indeterminate")
    )
    agreement_summary = " / ".join(
        str(agreement.get(value, 0)) for value in ("agree", "disagree", "not_applicable")
    )
    return f"""# GovAgent-KR MVP Result

결정: `{report["decision"]}`

{report["decision_note"]}

이 결과는 동결된 한 법령 corpus와 20개 공개 item에 대한 개발 실험이다. 정부 인증,
법률 정확성 보증, 공공행정 배치 승인 또는 RL 학습 승인으로 해석할 수 없다.

## 고정 분모 결과

- 전체 attempts: {report["attempts"]}
- attempt successes: {report["attempt_successes"]}
- core successes: {report["core_successes"]}
- analysis-eligible attempts: {report["analysis_eligible_attempts"]}
- excluded unclassified errors: {report["excluded_error_attempts"]}
- SUT_ERROR family 수: {len(report["sut_error_families"])}
- human verdicts (correct / incorrect / indeterminate): {human_summary}
- grader agreement (agree / disagree / N/A): {agreement_summary}
- grader false-positive runs: {report["grader_false_positive_runs"]}
- grader false-negative runs: {report["grader_false_negative_runs"]}
- grader disagreement runs: {report["grader_disagreement_runs"]}
- task/gold/grader design-error runs: {report["design_error_runs"]}

| Family | Attempts | Core successes | Excluded errors | SUT_ERROR runs |
| --- | ---: | ---: | ---: | ---: |
{rows}

## 사람 검토 root cause

{causes}

단일 Overall/PASS는 계산하지 않았다. 다음 행동의 범위는 `{report["decision_scope"]}`이다.
"""


def final_main() -> None:
    parser = argparse.ArgumentParser(description="Generate the fully reviewed 60-run MVP result")
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--review-bundle", type=Path, required=True)
    parser.add_argument("--primary-reviews", type=Path, required=True)
    parser.add_argument("--secondary-bundle", type=Path, required=True)
    parser.add_argument("--secondary-reviews", type=Path, required=True)
    parser.add_argument("--revealed-reviews", type=Path, required=True)
    parser.add_argument("--disagreements", type=Path, required=True)
    parser.add_argument("--adjudications", type=Path)
    parser.add_argument("--decision", choices=("go_expand_eval", "redesign", "stop"), required=True)
    parser.add_argument("--decision-note", required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = build_final_report(
            freeze_path=args.freeze,
            log_dir=args.log_dir,
            review_bundle_dir=args.review_bundle,
            primary_review_csv=args.primary_reviews,
            secondary_bundle_dir=args.secondary_bundle,
            secondary_review_csv=args.secondary_reviews,
            revealed_reviews_path=args.revealed_reviews,
            disagreements_path=args.disagreements,
            adjudications_path=args.adjudications,
            decision=args.decision,
            decision_note=args.decision_note,
        )
    except (
        FinalReportError,
        FreezeError,
        ReviewError,
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        parser.error(str(exc))
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_bytes(canonical_json_bytes(report))
    args.markdown_output.write_text(final_markdown(report), encoding="utf-8")
    sys.stdout.buffer.write(canonical_json_bytes(report))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an evidence-based MVP readiness report")
    parser.add_argument("--quality-receipt", type=Path)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = build_readiness_report(quality_receipt_path=args.quality_receipt)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_bytes(canonical_json_bytes(report))
    args.markdown_output.write_text(readiness_markdown(report), encoding="utf-8")
    sys.stdout.buffer.write(canonical_json_bytes(report))


if __name__ == "__main__":
    main()
