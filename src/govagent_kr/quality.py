from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from .canonical import canonical_json_bytes, content_sha256
from .constants import PACKAGE_ROOT, PROFILE_ID, PROFILE_VERSION
from .freeze import _container_record, _critical_files, _file_record, _git_value


def _run(
    command: list[str],
    *,
    docker_smoke: bool = False,
    frozen_image: str | None = None,
) -> dict[str, Any]:
    environment = dict(os.environ)
    if docker_smoke:
        environment["GOVAGENT_RUN_DOCKER_SMOKE"] = "1"
        if frozen_image is not None:
            environment["GOVAGENT_SUT_IMAGE"] = frozen_image
            environment["GOVAGENT_SMOKE_COMPOSE_PATH"] = str(PACKAGE_ROOT / "compose.frozen.yaml")
    completed = subprocess.run(
        command,
        cwd=PACKAGE_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    stdout = completed.stdout.encode("utf-8")
    stderr = completed.stderr.encode("utf-8")
    return {
        "command": command,
        "exit_code": completed.returncode,
        "stdout_sha256": content_sha256(stdout),
        "stderr_sha256": content_sha256(stderr),
        "stdout_tail": completed.stdout[-2_000:],
        "stderr_tail": completed.stderr[-2_000:],
    }


def run_quality_gates(
    *,
    include_docker_smoke: bool,
    container_image: str | None = None,
) -> dict[str, Any]:
    gates = [
        _run(["uv", "run", "govagent-validate"]),
        _run(["uv", "run", "govagent-diagnostics"]),
        _run(["uv", "run", "ruff", "check", "."]),
        _run(["uv", "run", "ruff", "format", "--check", "."]),
        _run(
            ["uv", "run", "pytest", "-q"],
            docker_smoke=include_docker_smoke,
            frozen_image=container_image,
        ),
    ]
    repository_root = Path(_git_value(PACKAGE_ROOT, "rev-parse", "--show-toplevel"))
    scoped_path = PACKAGE_ROOT.relative_to(repository_root).as_posix()
    scoped_status = _git_value(repository_root, "status", "--short", "--", scoped_path)
    git_commit = _git_value(repository_root, "rev-parse", "HEAD")
    artifacts = [_file_record(path, PACKAGE_ROOT) for path in _critical_files(PACKAGE_ROOT)]
    artifact_set_sha256 = content_sha256(canonical_json_bytes(artifacts))
    container = _container_record(container_image)
    exact_image = bool(container is not None and container_image == container.get("image_id"))
    gates_ok = all(gate["exit_code"] == 0 for gate in gates)
    result: dict[str, Any] = {
        "schema_version": "govagent-kr/quality-receipt-v0.1",
        "profile_id": PROFILE_ID,
        "profile_version": PROFILE_VERSION,
        "docker_smoke_included": include_docker_smoke,
        "docker_smoke_compose": (
            "compose.frozen.yaml" if container_image is not None else "compose.yaml"
        ),
        "git": {
            "commit": git_commit,
            "gov_agent_worktree_status": scoped_status,
        },
        "artifact_set_sha256": artifact_set_sha256,
        "container": container,
        "gates": gates,
    }
    result["ok"] = gates_ok
    result["analysis_quality_eligible"] = bool(
        gates_ok
        and include_docker_smoke
        and container_image is not None
        and exact_image
        and not scoped_status
    )
    result["receipt_sha256"] = content_sha256(canonical_json_bytes(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run and receipt all engineering quality gates")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-docker-smoke", action="store_true")
    parser.add_argument(
        "--container-image",
        help="exact sha256 image ID to exercise through compose.frozen.yaml",
    )
    args = parser.parse_args()
    result = run_quality_gates(
        include_docker_smoke=not args.skip_docker_smoke,
        container_image=args.container_image,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(result))
    sys.stdout.buffer.write(canonical_json_bytes(result))
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
