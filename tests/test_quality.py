from pathlib import Path
from typing import Any

from govagent_kr import quality
from govagent_kr.canonical import canonical_json_bytes, content_sha256

IMAGE_ID = "sha256:" + "d" * 64


def _patch_quality_environment(
    monkeypatch: Any,
    tmp_path: Path,
    *,
    worktree_status: str = "",
) -> list[dict[str, Any]]:
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("fixed\n", encoding="utf-8")
    monkeypatch.setattr(quality, "PACKAGE_ROOT", tmp_path)
    monkeypatch.setattr(quality, "_critical_files", lambda root: [artifact])
    monkeypatch.setattr(
        quality,
        "_container_record",
        lambda image: (
            None
            if image is None
            else {"requested_image": image, "image_id": IMAGE_ID, "repo_digests": []}
        ),
    )

    def fake_git_value(root: Path, *args: str) -> str:
        if args == ("rev-parse", "--show-toplevel"):
            return str(tmp_path)
        if args == ("status", "--short", "--", "."):
            return worktree_status
        if args == ("rev-parse", "HEAD"):
            return "a" * 40
        raise AssertionError(args)

    monkeypatch.setattr(quality, "_git_value", fake_git_value)
    calls: list[dict[str, Any]] = []

    def fake_run(
        command: list[str],
        *,
        docker_smoke: bool = False,
        frozen_image: str | None = None,
    ) -> dict[str, Any]:
        calls.append(
            {
                "command": command,
                "docker_smoke": docker_smoke,
                "frozen_image": frozen_image,
            }
        )
        return {"command": command, "exit_code": 0}

    monkeypatch.setattr(quality, "_run", fake_run)
    return calls


def test_analysis_quality_receipt_binds_clean_artifacts_commit_and_exact_image(
    monkeypatch: Any, tmp_path: Path
) -> None:
    calls = _patch_quality_environment(monkeypatch, tmp_path)

    receipt = quality.run_quality_gates(
        include_docker_smoke=True,
        container_image=IMAGE_ID,
    )

    assert receipt["ok"] is True
    assert receipt["analysis_quality_eligible"] is True
    assert receipt["docker_smoke_compose"] == "compose.frozen.yaml"
    assert receipt["container"]["image_id"] == IMAGE_ID
    unsigned = dict(receipt)
    recorded_hash = unsigned.pop("receipt_sha256")
    assert recorded_hash == content_sha256(canonical_json_bytes(unsigned))
    pytest_call = calls[-1]
    assert pytest_call["docker_smoke"] is True
    assert pytest_call["frozen_image"] == IMAGE_ID


def test_quality_receipt_is_not_analysis_eligible_without_clean_exact_frozen_image(
    monkeypatch: Any, tmp_path: Path
) -> None:
    _patch_quality_environment(monkeypatch, tmp_path, worktree_status="M src/agent.py")

    dirty = quality.run_quality_gates(
        include_docker_smoke=True,
        container_image=IMAGE_ID,
    )
    assert dirty["ok"] is True
    assert dirty["analysis_quality_eligible"] is False

    mutable = quality.run_quality_gates(
        include_docker_smoke=True,
        container_image="govagent:latest",
    )
    assert mutable["analysis_quality_eligible"] is False
