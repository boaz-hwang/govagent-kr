from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from .canonical import canonical_json_bytes, content_sha256
from .constants import PACKAGE_ROOT, PROFILE_ID, PROFILE_VERSION
from .dataset import dataset_integrity_report
from .diagnostics import run_diagnostics
from .fixtures import FixtureStore
from .model_config import normalize_frozen_model_config
from .strict_json import StrictJSONError, parse_strict_json_object


class FreezeError(RuntimeError):
    """Raised when the requested freeze would overstate evaluation readiness."""


_SENSITIVE_CONFIG_KEYS = {
    "api_key",
    "apikey",
    "access_token",
    "refresh_token",
    "bearer_token",
    "password",
    "secret",
    "credentials",
}
_SENSITIVE_HEADER_KEYS = {
    "authorization",
    "default_headers",
    "extra_headers",
    "headers",
    "proxy_authorization",
    "x_api_key",
}
_SENSITIVE_KEY_SUFFIXES = ("_api_key", "_password", "_secret", "_token")


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _reject_secrets(value: Any, path: str = "model_config") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).casefold().replace("-", "_")
            if normalized in _SENSITIVE_HEADER_KEYS:
                raise FreezeError(
                    f"freeze manifest must not contain provider headers: {path}.{key}"
                )
            if normalized in _SENSITIVE_CONFIG_KEYS or normalized.endswith(_SENSITIVE_KEY_SUFFIXES):
                raise FreezeError(f"freeze manifest must not contain credentials: {path}.{key}")
            _reject_secrets(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_secrets(child, f"{path}[{index}]")


def _file_record(path: Path, root: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise FreezeError(f"critical artifact must not be a symbolic link: {path}")
    resolved = path.resolve(strict=True)
    if resolved != root and root not in resolved.parents:
        raise FreezeError(f"critical artifact escapes the freeze root: {path}")
    payload = path.read_bytes()
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": content_sha256(payload),
        "bytes": len(payload),
    }


def _critical_files(root: Path) -> list[Path]:
    explicit = [
        root / "pyproject.toml",
        root / "uv.lock",
        root / "profile.yaml",
        root / "compose.yaml",
        root / "compose.frozen.yaml",
        root / "data" / "cases.jsonl",
        root / "data" / "gold.jsonl",
    ]
    result = [path for path in explicit if path.is_file()]
    for directory in (root / "src", root / "runtime", root / "fixtures"):
        if not directory.exists():
            continue
        result.extend(
            path
            for path in directory.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts and not path.name.endswith(".pyc")
        )
    unique = {path.resolve(): path for path in result}
    return sorted(unique.values(), key=lambda path: path.relative_to(root).as_posix())


def _git_value(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _container_record(image: str | None) -> dict[str, Any] | None:
    if image is None:
        return None
    completed = subprocess.run(
        ["docker", "image", "inspect", image, "--format", "{{json .}}"],
        check=True,
        capture_output=True,
        text=True,
    )
    document = json.loads(completed.stdout)
    repo_digests = sorted(document.get("RepoDigests") or [])
    return {
        "requested_image": image,
        "image_id": document["Id"],
        "repo_digests": repo_digests,
    }


def _validate_analysis_quality_receipt(
    receipt: dict[str, Any] | None,
    *,
    artifact_set_sha256: str,
    git_commit: str,
    container: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(receipt, dict):
        raise FreezeError("analysis freeze requires a quality receipt")
    recorded_hash = receipt.get("receipt_sha256")
    unsigned = dict(receipt)
    unsigned.pop("receipt_sha256", None)
    if recorded_hash != content_sha256(canonical_json_bytes(unsigned)):
        raise FreezeError("quality receipt hash is invalid")
    if receipt.get("schema_version") != "govagent-kr/quality-receipt-v0.1":
        raise FreezeError("quality receipt schema is unsupported")
    if receipt.get("profile_id") != PROFILE_ID or receipt.get("profile_version") != PROFILE_VERSION:
        raise FreezeError("quality receipt profile differs from the freeze")
    if receipt.get("ok") is not True or receipt.get("analysis_quality_eligible") is not True:
        raise FreezeError("quality receipt is not analysis-eligible")
    expected_commands = [
        ["uv", "run", "govagent-validate"],
        ["uv", "run", "govagent-diagnostics"],
        ["uv", "run", "ruff", "check", "."],
        ["uv", "run", "ruff", "format", "--check", "."],
        ["uv", "run", "pytest", "-q"],
    ]
    gates = receipt.get("gates")
    if not isinstance(gates, list) or len(gates) != len(expected_commands):
        raise FreezeError("quality receipt does not contain the exact required gates")
    for gate, expected_command in zip(gates, expected_commands, strict=True):
        if not isinstance(gate, dict) or gate.get("command") != expected_command:
            raise FreezeError("quality receipt gate command differs from the required suite")
        if type(gate.get("exit_code")) is not int or gate["exit_code"] != 0:
            raise FreezeError("quality receipt contains a failed gate")
        if not _is_sha256(gate.get("stdout_sha256")) or not _is_sha256(gate.get("stderr_sha256")):
            raise FreezeError("quality receipt gate output hash is invalid")
        if not isinstance(gate.get("stdout_tail"), str) or not isinstance(
            gate.get("stderr_tail"), str
        ):
            raise FreezeError("quality receipt gate output excerpt is invalid")
    if (
        receipt.get("docker_smoke_included") is not True
        or receipt.get("docker_smoke_compose") != "compose.frozen.yaml"
    ):
        raise FreezeError("quality receipt did not smoke the frozen compose environment")
    if receipt.get("artifact_set_sha256") != artifact_set_sha256:
        raise FreezeError("quality receipt artifact set differs from the freeze")
    git = receipt.get("git")
    if not isinstance(git, dict) or git.get("commit") != git_commit:
        raise FreezeError("quality receipt git commit differs from the freeze")
    if git.get("gov_agent_worktree_status") != "":
        raise FreezeError("quality receipt was not produced from a clean gov-agent worktree")
    receipt_container = receipt.get("container")
    if (
        container is None
        or not isinstance(receipt_container, dict)
        or receipt_container.get("requested_image") != container.get("image_id")
        or receipt_container.get("image_id") != container.get("image_id")
    ):
        raise FreezeError("quality receipt SUT image differs from the freeze")
    return receipt


def _validate_gold_review_receipt(
    receipt: dict[str, Any] | None,
    *,
    root: Path,
) -> dict[str, Any]:
    if not isinstance(receipt, dict):
        raise FreezeError("analysis freeze requires a gold review receipt")
    recorded_hash = receipt.get("receipt_sha256")
    unsigned = dict(receipt)
    unsigned.pop("receipt_sha256", None)
    if recorded_hash != content_sha256(canonical_json_bytes(unsigned)):
        raise FreezeError("gold review receipt hash is invalid")
    if receipt.get("schema_version") != "govagent-kr/gold-review-receipt-v0.1":
        raise FreezeError("gold review receipt schema is unsupported")
    if type(receipt.get("item_count")) is not int or receipt["item_count"] != 20:
        raise FreezeError("gold review receipt must cover exactly 20 items")
    required_flags = {
        "two_distinct_ids_validated": True,
        "reviewer_identity_authenticated": False,
        "human_adjudication_required": True,
        "automatic_agreement": False,
        "ai_review": False,
        "analysis_ready": True,
    }
    if any(receipt.get(key) is not expected for key, expected in required_flags.items()):
        raise FreezeError("gold review receipt governance flags are invalid")
    reviewer_1 = receipt.get("reviewer_1")
    reviewer_2 = receipt.get("reviewer_2")
    if (
        not isinstance(reviewer_1, str)
        or not reviewer_1
        or reviewer_1 != reviewer_1.strip()
        or not isinstance(reviewer_2, str)
        or not reviewer_2
        or reviewer_2 != reviewer_2.strip()
        or reviewer_1.casefold() == reviewer_2.casefold()
    ):
        raise FreezeError("gold review receipt does not identify two distinct reviewers")
    adjudicator_ids = receipt.get("adjudicator_ids")
    if (
        not isinstance(adjudicator_ids, list)
        or not adjudicator_ids
        or any(
            not isinstance(value, str) or not value or value != value.strip()
            for value in adjudicator_ids
        )
        or len(adjudicator_ids) != len(set(adjudicator_ids))
    ):
        raise FreezeError("gold review receipt adjudicator IDs are invalid")
    review_hash_keys = (
        "review_bundle_manifest_sha256",
        "review_1_sha256",
        "review_2_sha256",
        "adjudication_sha256",
    )
    if any(not _is_sha256(receipt.get(key)) for key in review_hash_keys):
        raise FreezeError("gold review receipt contains an invalid review artifact hash")
    review_hashes = {key: receipt[key] for key in review_hash_keys}
    if receipt.get("review_artifact_set_sha256") != content_sha256(
        canonical_json_bytes(review_hashes)
    ):
        raise FreezeError("gold review artifact set hash is invalid")
    for key in ("source_candidate_gold_sha256",):
        if not _is_sha256(receipt.get(key)):
            raise FreezeError(f"gold review receipt hash is invalid: {key}")
    cases_hash = content_sha256((root / "data" / "cases.jsonl").read_bytes())
    gold_hash = content_sha256((root / "data" / "gold.jsonl").read_bytes())
    if receipt.get("source_cases_sha256") != cases_hash:
        raise FreezeError("gold review receipt cases differ from the freeze")
    if receipt.get("output_gold_sha256") != gold_hash:
        raise FreezeError("gold review receipt final gold differs from the freeze")
    return receipt


def _validate_analysis_manifest_semantics(manifest: dict[str, Any]) -> None:
    if (
        manifest.get("profile_id") != PROFILE_ID
        or manifest.get("profile_version") != PROFILE_VERSION
    ):
        raise FreezeError("analysis freeze profile differs from this runtime")
    if (
        manifest.get("freeze_kind") != "analysis_freeze"
        or manifest.get("analysis_eligible") is not True
    ):
        raise FreezeError("analysis freeze kind or eligibility is invalid")
    requested_model = manifest.get("requested_model")
    if (
        not isinstance(requested_model, str)
        or not requested_model
        or requested_model != requested_model.strip()
    ):
        raise FreezeError("analysis freeze requested_model is invalid")
    try:
        normalized_model_config = normalize_frozen_model_config(manifest.get("model_config"))
    except ValueError as exc:
        raise FreezeError("analysis freeze model_config is invalid") from exc
    if manifest.get("model_config") != normalized_model_config:
        raise FreezeError("analysis freeze model_config is not in canonical frozen form")

    dataset_review = manifest.get("dataset_review")
    if not isinstance(dataset_review, dict) or dataset_review != {
        "analysis_ready": True,
        "pending_human_review": 0,
        "excluded_items": 0,
    }:
        raise FreezeError("analysis freeze dataset review is incomplete")
    git = manifest.get("git")
    if not isinstance(git, dict) or git.get("gov_agent_worktree_status") != "":
        raise FreezeError("analysis freeze was not created from a clean worktree")
    runtime = manifest.get("runtime")
    try:
        expected_runtime = {
            "python": platform.python_version(),
            "inspect_ai": importlib.metadata.version("inspect-ai"),
            "openai": importlib.metadata.version("openai"),
            "platform": platform.platform(),
        }
    except importlib.metadata.PackageNotFoundError as exc:
        raise FreezeError("analysis freeze runtime packages are not installed") from exc
    if runtime != expected_runtime:
        raise FreezeError("analysis freeze runtime differs from the current runtime")
    if not _is_sha256(manifest.get("diagnostics_sha256")):
        raise FreezeError("analysis freeze diagnostics hash is invalid")
    container = manifest.get("container")
    image_id = container.get("image_id") if isinstance(container, dict) else None
    if (
        not isinstance(image_id, str)
        or not image_id.startswith("sha256:")
        or not _is_sha256(image_id.removeprefix("sha256:"))
        or container.get("requested_image") != image_id
    ):
        raise FreezeError("analysis freeze container identity is not immutable")


def build_freeze_manifest(
    *,
    root: Path = PACKAGE_ROOT,
    requested_model: str,
    model_config: dict[str, Any],
    mode: str,
    container_image: str | None,
    quality_receipt: dict[str, Any] | None = None,
    gold_review_receipt: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    if mode not in {"candidate", "analysis"}:
        raise FreezeError(f"unsupported freeze mode: {mode}")
    if not requested_model.strip():
        raise FreezeError("requested_model must be non-empty")
    _reject_secrets(model_config)
    if mode == "analysis" and not model_config:
        raise FreezeError("analysis freeze requires an explicit model configuration")
    stored_model_config = model_config
    if mode == "analysis":
        try:
            stored_model_config = normalize_frozen_model_config(model_config)
        except ValueError as exc:
            raise FreezeError("analysis freeze model_config is invalid") from exc
    diagnostics = run_diagnostics(
        root / "fixtures" / "local-contract-decree",
        root / "data" / "cases.jsonl",
        root / "data" / "gold.jsonl",
    )
    if not diagnostics["engineering_ready"]:
        raise FreezeError("engineering diagnostics are not ready")
    dataset = dataset_integrity_report(
        FixtureStore(root / "fixtures" / "local-contract-decree"),
        root / "data" / "cases.jsonl",
        root / "data" / "gold.jsonl",
    )
    if mode == "analysis" and not dataset["analysis_ready"]:
        raise FreezeError("analysis freeze requires two-person gold adjudication for every item")
    if mode == "analysis" and container_image is None:
        raise FreezeError("analysis freeze requires an immutable SUT container image")

    files = [_file_record(path, root) for path in _critical_files(root)]
    file_set_hash = content_sha256(canonical_json_bytes(files))
    container = _container_record(container_image)
    if mode == "analysis" and container is not None and container_image != container["image_id"]:
        raise FreezeError(
            "analysis freeze requires the exact sha256 image ID, not a mutable tag or alias"
        )
    repository_root = Path(_git_value(root, "rev-parse", "--show-toplevel"))
    scoped_path = root.relative_to(repository_root).as_posix()
    scoped_status = _git_value(repository_root, "status", "--short", "--", scoped_path)
    if mode == "analysis" and scoped_status:
        raise FreezeError("analysis freeze requires a clean gov-agent worktree")
    git_commit = _git_value(repository_root, "rev-parse", "HEAD")
    validated_quality = (
        _validate_analysis_quality_receipt(
            quality_receipt,
            artifact_set_sha256=file_set_hash,
            git_commit=git_commit,
            container=container,
        )
        if mode == "analysis"
        else None
    )
    validated_gold_review = (
        _validate_gold_review_receipt(gold_review_receipt, root=root)
        if mode == "analysis"
        else None
    )
    base: dict[str, Any] = {
        "profile_id": PROFILE_ID,
        "profile_version": PROFILE_VERSION,
        "freeze_kind": "analysis_freeze" if mode == "analysis" else "engineering_candidate",
        "analysis_eligible": mode == "analysis",
        "requested_model": requested_model,
        "model_config": stored_model_config,
        "runtime": {
            "python": platform.python_version(),
            "inspect_ai": importlib.metadata.version("inspect-ai"),
            "openai": importlib.metadata.version("openai"),
            "platform": platform.platform(),
        },
        "container": container,
        "git": {
            "commit": git_commit,
            "gov_agent_worktree_status": scoped_status,
        },
        "quality_receipt": validated_quality,
        "gold_review_receipt": validated_gold_review,
        "artifacts": files,
        "artifact_set_sha256": file_set_hash,
        "diagnostics_sha256": diagnostics["report_sha256"],
        "dataset_review": {
            "analysis_ready": dataset["analysis_ready"],
            "pending_human_review": dataset["pending_human_review"],
            "excluded_items": dataset["excluded_items"],
        },
    }
    base["freeze_id"] = content_sha256(canonical_json_bytes(base))
    if mode == "analysis":
        _validate_analysis_manifest_semantics(base)
    return base


def verify_freeze_manifest(
    manifest_path: Path,
    *,
    root: Path = PACKAGE_ROOT,
    require_analysis: bool = False,
) -> dict[str, Any]:
    root = root.resolve()
    try:
        manifest = parse_strict_json_object(manifest_path.read_bytes(), max_bytes=4 * 1024 * 1024)
    except (OSError, StrictJSONError) as exc:
        raise FreezeError(f"invalid freeze manifest: {manifest_path}") from exc
    recorded_freeze_id = manifest.get("freeze_id")
    unsigned = dict(manifest)
    unsigned.pop("freeze_id", None)
    if recorded_freeze_id != content_sha256(canonical_json_bytes(unsigned)):
        raise FreezeError("freeze_id does not match manifest content")
    if require_analysis and manifest.get("analysis_eligible") is not True:
        raise FreezeError("an analysis-eligible freeze is required")
    _reject_secrets(manifest.get("model_config"))
    if manifest.get("analysis_eligible") is True:
        _validate_analysis_manifest_semantics(manifest)
    elif (
        manifest.get("profile_id") != PROFILE_ID
        or manifest.get("profile_version") != PROFILE_VERSION
        or manifest.get("freeze_kind") != "engineering_candidate"
    ):
        raise FreezeError("candidate freeze schema or profile is invalid")

    expected_files = [_file_record(path, root) for path in _critical_files(root)]
    if manifest.get("artifacts") != expected_files:
        raise FreezeError("critical artifact set or content differs from the freeze")
    if manifest.get("artifact_set_sha256") != content_sha256(canonical_json_bytes(expected_files)):
        raise FreezeError("artifact set hash is invalid")

    repository_root = Path(_git_value(root, "rev-parse", "--show-toplevel"))
    scoped_path = root.relative_to(repository_root).as_posix()
    current_status = _git_value(repository_root, "status", "--short", "--", scoped_path)
    if manifest.get("git", {}).get("commit") != _git_value(repository_root, "rev-parse", "HEAD"):
        raise FreezeError("git commit differs from the freeze")
    if manifest.get("analysis_eligible") is True and current_status:
        raise FreezeError("analysis freeze verification requires a clean gov-agent worktree")
    container = manifest.get("container")
    current: dict[str, Any] | None = None
    if container is not None:
        if not isinstance(container, dict) or not isinstance(container.get("requested_image"), str):
            raise FreezeError("freeze container record is invalid")
        if manifest.get("analysis_eligible") is True and container.get(
            "requested_image"
        ) != container.get("image_id"):
            raise FreezeError("analysis freeze does not use its exact sha256 image ID")
        current = _container_record(container["requested_image"])
        if current is None or current["image_id"] != container.get("image_id"):
            raise FreezeError("SUT container image differs from the freeze")
    elif require_analysis or manifest.get("analysis_eligible") is True:
        raise FreezeError("analysis freeze has no SUT container")
    if manifest.get("analysis_eligible") is True:
        diagnostics = run_diagnostics(
            root / "fixtures" / "local-contract-decree",
            root / "data" / "cases.jsonl",
            root / "data" / "gold.jsonl",
        )
        if diagnostics.get("analysis_ready") is not True or diagnostics.get(
            "report_sha256"
        ) != manifest.get("diagnostics_sha256"):
            raise FreezeError("current diagnostics differ from the analysis freeze")
        _validate_analysis_quality_receipt(
            manifest.get("quality_receipt"),
            artifact_set_sha256=content_sha256(canonical_json_bytes(expected_files)),
            git_commit=_git_value(repository_root, "rev-parse", "HEAD"),
            container=current,
        )
        _validate_gold_review_receipt(manifest.get("gold_review_receipt"), root=root)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a deterministic environment freeze receipt"
    )
    parser.add_argument("--requested-model", required=True)
    parser.add_argument("--model-config", type=Path)
    parser.add_argument("--mode", choices=("candidate", "analysis"), default="candidate")
    parser.add_argument("--container-image")
    parser.add_argument("--quality-receipt", type=Path)
    parser.add_argument("--gold-review-receipt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model_config: dict[str, Any] = {}
    if args.model_config:
        try:
            model_config = parse_strict_json_object(
                args.model_config.read_bytes(), max_bytes=1_000_000
            )
        except (OSError, StrictJSONError) as exc:
            parser.error(f"invalid --model-config: {exc}")
    quality_receipt: dict[str, Any] | None = None
    if args.quality_receipt:
        try:
            quality_receipt = parse_strict_json_object(
                args.quality_receipt.read_bytes(), max_bytes=8 * 1024 * 1024
            )
        except (OSError, StrictJSONError) as exc:
            parser.error(f"invalid --quality-receipt: {exc}")
    gold_review_receipt: dict[str, Any] | None = None
    if args.gold_review_receipt:
        try:
            gold_review_receipt = parse_strict_json_object(
                args.gold_review_receipt.read_bytes(), max_bytes=8 * 1024 * 1024
            )
        except (OSError, StrictJSONError) as exc:
            parser.error(f"invalid --gold-review-receipt: {exc}")
    try:
        manifest = build_freeze_manifest(
            requested_model=args.requested_model,
            model_config=model_config,
            mode=args.mode,
            container_image=args.container_image,
            quality_receipt=quality_receipt,
            gold_review_receipt=gold_review_receipt,
        )
    except (FreezeError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(manifest))
    sys.stdout.buffer.write(canonical_json_bytes(manifest))


def verify_main() -> None:
    parser = argparse.ArgumentParser(description="Verify a freeze against the current environment")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--require-analysis", action="store_true")
    args = parser.parse_args()
    try:
        manifest = verify_freeze_manifest(
            args.manifest,
            require_analysis=args.require_analysis,
        )
    except FreezeError as exc:
        parser.error(str(exc))
    sys.stdout.buffer.write(
        canonical_json_bytes(
            {
                "ok": True,
                "freeze_id": manifest["freeze_id"],
                "analysis_eligible": manifest["analysis_eligible"],
            }
        )
    )


if __name__ == "__main__":
    main()
