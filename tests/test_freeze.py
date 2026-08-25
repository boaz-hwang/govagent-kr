from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from govagent_kr import freeze
from govagent_kr.canonical import canonical_json_bytes, content_sha256
from govagent_kr.freeze import FreezeError, build_freeze_manifest


def _project_tree(tmp_path: Path) -> tuple[Path, Path]:
    repository_root = tmp_path / "repository"
    root = repository_root / "gov-agent"
    (root / "src" / "govagent_kr").mkdir(parents=True)
    (root / "data").mkdir()
    (root / "fixtures" / "local-contract-decree").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")
    (root / "profile.yaml").write_text("version: 1\n", encoding="utf-8")
    (root / "src" / "govagent_kr" / "agent.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "data" / "cases.jsonl").write_text("{}\n", encoding="utf-8")
    (root / "data" / "gold.jsonl").write_text("{}\n", encoding="utf-8")
    return repository_root, root


def _patch_ready_environment(
    monkeypatch: pytest.MonkeyPatch,
    repository_root: Path,
    *,
    analysis_ready: bool = True,
    worktree_status: str = "",
) -> list[tuple[Path, tuple[str, ...]]]:
    monkeypatch.setattr(
        freeze,
        "run_diagnostics",
        lambda *args: {
            "engineering_ready": True,
            "analysis_ready": analysis_ready,
            "report_sha256": "d" * 64,
        },
    )
    monkeypatch.setattr(freeze, "FixtureStore", lambda path: object())
    monkeypatch.setattr(
        freeze,
        "dataset_integrity_report",
        lambda *args: {
            "analysis_ready": analysis_ready,
            "pending_human_review": 0 if analysis_ready else 20,
            "excluded_items": 0,
        },
    )
    monkeypatch.setattr(
        freeze.importlib.metadata,
        "version",
        lambda package: {"inspect-ai": "0.3.260", "openai": "3.3.1"}[package],
    )
    monkeypatch.setattr(
        freeze,
        "_container_record",
        lambda image: (
            None
            if image is None
            else {
                "requested_image": image,
                "image_id": "sha256:" + "c" * 64,
                "repo_digests": ["example@sha256:" + "c" * 64],
            }
        ),
    )
    calls: list[tuple[Path, tuple[str, ...]]] = []

    def fake_git_value(cwd: Path, *args: str) -> str:
        calls.append((cwd, args))
        if args == ("rev-parse", "--show-toplevel"):
            return str(repository_root)
        if args == ("status", "--short", "--", "gov-agent"):
            return worktree_status
        if args == ("rev-parse", "HEAD"):
            return "a" * 40
        raise AssertionError(f"unexpected git invocation: {cwd=} {args=}")

    monkeypatch.setattr(freeze, "_git_value", fake_git_value)
    return calls


def _quality_receipt(root: Path) -> dict[str, Any]:
    artifacts = [freeze._file_record(path, root) for path in freeze._critical_files(root)]
    commands = [
        ["uv", "run", "govagent-validate"],
        ["uv", "run", "govagent-diagnostics"],
        ["uv", "run", "ruff", "check", "."],
        ["uv", "run", "ruff", "format", "--check", "."],
        ["uv", "run", "pytest", "-q"],
    ]
    gates = [
        {
            "command": command,
            "exit_code": 0,
            "stdout_sha256": "1" * 64,
            "stderr_sha256": "2" * 64,
            "stdout_tail": "ok",
            "stderr_tail": "",
        }
        for command in commands
    ]
    base: dict[str, Any] = {
        "schema_version": "govagent-kr/quality-receipt-v0.1",
        "profile_id": "govagent-kr-temporal-citation",
        "profile_version": "0.1.0",
        "docker_smoke_included": True,
        "docker_smoke_compose": "compose.frozen.yaml",
        "git": {"commit": "a" * 40, "gov_agent_worktree_status": ""},
        "artifact_set_sha256": content_sha256(canonical_json_bytes(artifacts)),
        "container": {
            "requested_image": IMMUTABLE_IMAGE_ID,
            "image_id": IMMUTABLE_IMAGE_ID,
            "repo_digests": ["example@sha256:" + "c" * 64],
        },
        "gates": gates,
        "ok": True,
        "analysis_quality_eligible": True,
    }
    base["receipt_sha256"] = content_sha256(canonical_json_bytes(base))
    return base


def _gold_review_receipt(root: Path) -> dict[str, Any]:
    review_hashes = {
        "review_bundle_manifest_sha256": "3" * 64,
        "review_1_sha256": "4" * 64,
        "review_2_sha256": "5" * 64,
        "adjudication_sha256": "6" * 64,
    }
    base: dict[str, Any] = {
        "schema_version": "govagent-kr/gold-review-receipt-v0.1",
        "item_count": 20,
        "reviewer_1": "sme-a",
        "reviewer_2": "sme-b",
        "two_distinct_ids_validated": True,
        "reviewer_identity_authenticated": False,
        "adjudicator_ids": ["sme-c"],
        "human_adjudication_required": True,
        "automatic_agreement": False,
        "ai_review": False,
        "source_cases_sha256": content_sha256((root / "data" / "cases.jsonl").read_bytes()),
        "source_candidate_gold_sha256": "7" * 64,
        **review_hashes,
        "review_artifact_set_sha256": content_sha256(canonical_json_bytes(review_hashes)),
        "output_gold_sha256": content_sha256((root / "data" / "gold.jsonl").read_bytes()),
        "analysis_ready": True,
    }
    base["receipt_sha256"] = content_sha256(canonical_json_bytes(base))
    return base


def _build(
    root: Path,
    *,
    mode: str = "candidate",
    image: str | None = None,
    include_quality: bool = True,
    include_gold_review: bool = True,
) -> dict[str, Any]:
    return build_freeze_manifest(
        root=root,
        requested_model="frontier-model",
        model_config={"temperature": 0, "reasoning_effort": "high"},
        mode=mode,
        container_image=image,
        quality_receipt=(
            _quality_receipt(root) if mode == "analysis" and include_quality else None
        ),
        gold_review_receipt=(
            _gold_review_receipt(root) if mode == "analysis" and include_gold_review else None
        ),
    )


IMMUTABLE_IMAGE_ID = "sha256:" + "c" * 64


def test_candidate_manifest_is_deterministic_and_scopes_git_to_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository_root, root = _project_tree(tmp_path)
    calls = _patch_ready_environment(monkeypatch, repository_root, worktree_status="M note.txt")

    first = _build(root)
    second = _build(root)

    assert first == second
    assert first["analysis_eligible"] is False
    assert first["git"]["gov_agent_worktree_status"] == "M note.txt"
    assert any(
        cwd == repository_root and args == ("status", "--short", "--", "gov-agent")
        for cwd, args in calls
    )


def test_artifact_change_changes_both_artifact_set_and_freeze_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository_root, root = _project_tree(tmp_path)
    _patch_ready_environment(monkeypatch, repository_root)
    first = _build(root)

    (root / "src" / "govagent_kr" / "agent.py").write_text("VALUE = 2\n", encoding="utf-8")
    second = _build(root)

    assert first["artifact_set_sha256"] != second["artifact_set_sha256"]
    assert first["freeze_id"] != second["freeze_id"]


def test_analysis_freeze_requires_review_container_and_clean_worktree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository_root, root = _project_tree(tmp_path)
    _patch_ready_environment(monkeypatch, repository_root, analysis_ready=False)
    with pytest.raises(FreezeError, match="two-person gold adjudication"):
        _build(root, mode="analysis", image="agent@sha256:digest")

    _patch_ready_environment(monkeypatch, repository_root, analysis_ready=True)
    with pytest.raises(FreezeError, match="immutable SUT container"):
        _build(root, mode="analysis")

    _patch_ready_environment(monkeypatch, repository_root, worktree_status="M src/agent.py")
    with pytest.raises(FreezeError, match="clean gov-agent worktree"):
        _build(root, mode="analysis", image=IMMUTABLE_IMAGE_ID)


def test_analysis_freeze_records_resolved_container_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository_root, root = _project_tree(tmp_path)
    _patch_ready_environment(monkeypatch, repository_root)

    manifest = _build(root, mode="analysis", image=IMMUTABLE_IMAGE_ID)

    assert manifest["analysis_eligible"] is True
    assert manifest["freeze_kind"] == "analysis_freeze"
    assert manifest["container"]["image_id"].startswith("sha256:")
    assert manifest["quality_receipt"]["analysis_quality_eligible"] is True
    assert manifest["gold_review_receipt"]["analysis_ready"] is True
    assert manifest["model_config"]["generate"]["max_retries"] == 0
    assert manifest["model_config"]["model_args"] == {}


def test_analysis_freeze_requires_current_analysis_quality_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository_root, root = _project_tree(tmp_path)
    _patch_ready_environment(monkeypatch, repository_root)

    with pytest.raises(FreezeError, match="requires a quality receipt"):
        _build(
            root,
            mode="analysis",
            image=IMMUTABLE_IMAGE_ID,
            include_quality=False,
        )

    stale = _quality_receipt(root)
    stale["artifact_set_sha256"] = "0" * 64
    unsigned = dict(stale)
    unsigned.pop("receipt_sha256")
    stale["receipt_sha256"] = content_sha256(canonical_json_bytes(unsigned))
    with pytest.raises(FreezeError, match="artifact set differs"):
        build_freeze_manifest(
            root=root,
            requested_model="frontier-model",
            model_config={"temperature": 0},
            mode="analysis",
            container_image=IMMUTABLE_IMAGE_ID,
            quality_receipt=stale,
        )

    incomplete = _quality_receipt(root)
    incomplete["gates"] = []
    unsigned = dict(incomplete)
    unsigned.pop("receipt_sha256")
    incomplete["receipt_sha256"] = content_sha256(canonical_json_bytes(unsigned))
    with pytest.raises(FreezeError, match="exact required gates"):
        build_freeze_manifest(
            root=root,
            requested_model="frontier-model",
            model_config={"temperature": 0},
            mode="analysis",
            container_image=IMMUTABLE_IMAGE_ID,
            quality_receipt=incomplete,
            gold_review_receipt=_gold_review_receipt(root),
        )


def test_analysis_freeze_requires_current_gold_review_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository_root, root = _project_tree(tmp_path)
    _patch_ready_environment(monkeypatch, repository_root)

    with pytest.raises(FreezeError, match="requires a gold review receipt"):
        _build(
            root,
            mode="analysis",
            image=IMMUTABLE_IMAGE_ID,
            include_gold_review=False,
        )

    stale = _gold_review_receipt(root)
    stale["output_gold_sha256"] = "0" * 64
    unsigned = dict(stale)
    unsigned.pop("receipt_sha256")
    stale["receipt_sha256"] = content_sha256(canonical_json_bytes(unsigned))
    with pytest.raises(FreezeError, match="final gold differs"):
        build_freeze_manifest(
            root=root,
            requested_model="frontier-model",
            model_config={"temperature": 0},
            mode="analysis",
            container_image=IMMUTABLE_IMAGE_ID,
            quality_receipt=_quality_receipt(root),
            gold_review_receipt=stale,
        )


def test_analysis_freeze_rejects_mutable_or_indirect_container_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository_root, root = _project_tree(tmp_path)
    _patch_ready_environment(monkeypatch, repository_root)

    with pytest.raises(FreezeError, match="exact sha256 image ID"):
        _build(root, mode="analysis", image="govagent-kr-sut:latest")


def test_empty_model_name_is_rejected_before_freezing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, root = _project_tree(tmp_path)
    monkeypatch.setattr(
        freeze,
        "run_diagnostics",
        lambda *args: pytest.fail("diagnostics should not run for invalid input"),
    )

    with pytest.raises(FreezeError, match="requested_model must be non-empty"):
        build_freeze_manifest(
            root=root,
            requested_model="   ",
            model_config={},
            mode="candidate",
            container_image=None,
        )


def test_provider_extra_headers_are_never_written_to_a_freeze(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, root = _project_tree(tmp_path)
    monkeypatch.setattr(
        freeze,
        "run_diagnostics",
        lambda *args: pytest.fail("diagnostics should not run for secret-bearing config"),
    )

    with pytest.raises(FreezeError, match="must not contain provider headers"):
        build_freeze_manifest(
            root=root,
            requested_model="frontier-model",
            model_config={"generate": {"extra_headers": {"Authorization": "Bearer secret"}}},
            mode="candidate",
            container_image=None,
        )


@pytest.mark.parametrize(
    "model_config",
    [
        {"model_args": {"admin_api_key": "secret"}},
        {"model_args": {"default_headers": {"Authorization": "Bearer secret"}}},
        {"model_args": {"headers": {"X-API-Key": "secret"}}},
    ],
)
def test_provider_credentials_hidden_in_model_args_are_rejected_before_freezing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_config: dict[str, Any],
) -> None:
    _, root = _project_tree(tmp_path)
    monkeypatch.setattr(
        freeze,
        "run_diagnostics",
        lambda *args: pytest.fail("diagnostics should not run for secret-bearing config"),
    )

    with pytest.raises(FreezeError, match="must not contain"):
        build_freeze_manifest(
            root=root,
            requested_model="frontier-model",
            model_config=model_config,
            mode="candidate",
            container_image=None,
        )


@pytest.mark.parametrize(
    "model_config",
    [
        {"generate": {"fallback_models": ["other/model"]}, "model_args": {}},
        {"generate": {"internal_tools": True}, "model_args": {}},
        {"generate": {}, "model_args": {"provider_option": "value"}},
    ],
)
def test_analysis_freeze_rejects_non_single_sut_model_capabilities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_config: dict[str, Any],
) -> None:
    _, root = _project_tree(tmp_path)
    monkeypatch.setattr(
        freeze,
        "run_diagnostics",
        lambda *args: pytest.fail("diagnostics should not run for invalid analysis config"),
    )

    with pytest.raises(FreezeError, match="model_config is invalid"):
        build_freeze_manifest(
            root=root,
            requested_model="frontier-model",
            model_config=model_config,
            mode="analysis",
            container_image=IMMUTABLE_IMAGE_ID,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.__setitem__("profile_version", "other"), "profile"),
        (lambda value: value.__setitem__("freeze_kind", "engineering_candidate"), "kind"),
        (lambda value: value.__setitem__("requested_model", ""), "requested_model"),
        (
            lambda value: value["dataset_review"].__setitem__("pending_human_review", 1),
            "dataset review",
        ),
        (
            lambda value: value["git"].__setitem__("gov_agent_worktree_status", "M task.py"),
            "clean worktree",
        ),
        (lambda value: value["runtime"].__setitem__("inspect_ai", "0.0.0"), "runtime"),
        (lambda value: value.__setitem__("diagnostics_sha256", "e" * 64), "diagnostics"),
    ],
)
def test_verify_analysis_freeze_rechecks_semantics_after_self_rehash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: Any,
    message: str,
) -> None:
    repository_root, root = _project_tree(tmp_path)
    _patch_ready_environment(monkeypatch, repository_root)
    manifest = _build(root, mode="analysis", image=IMMUTABLE_IMAGE_ID)
    changed = deepcopy(manifest)
    mutation(changed)
    changed.pop("freeze_id")
    changed["freeze_id"] = content_sha256(canonical_json_bytes(changed))
    path = tmp_path / "analysis-freeze.json"
    path.write_bytes(canonical_json_bytes(changed))

    with pytest.raises(FreezeError, match=message):
        freeze.verify_freeze_manifest(path, root=root, require_analysis=True)


def test_symbolic_link_cannot_smuggle_external_content_into_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository_root, root = _project_tree(tmp_path)
    _patch_ready_environment(monkeypatch, repository_root)
    outside = tmp_path / "secret.py"
    outside.write_text("SECRET = 'outside-root'\n", encoding="utf-8")
    (root / "src" / "govagent_kr" / "linked.py").symlink_to(outside)

    with pytest.raises(FreezeError, match="symbolic link"):
        _build(root)
