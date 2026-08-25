from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
from typing import Any

from inspect_ai import Epochs, Task, task
from inspect_ai.agent import Agent, AgentState, agent, sandbox_agent_bridge
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, GenerateConfig, get_model
from inspect_ai.scorer import Score, Scorer, Target, mean, scorer
from inspect_ai.solver import TaskState
from inspect_ai.util import sandbox

from govagent_kr.canonical import canonical_json_bytes
from govagent_kr.constants import (
    DEFAULT_CASES_PATH,
    DEFAULT_FIXTURE_ROOT,
    DEFAULT_GOLD_PATH,
    PROFILE_ID,
    PROFILE_VERSION,
)
from govagent_kr.dataset import load_cases, load_gold
from govagent_kr.fixtures import FixtureStore
from govagent_kr.freeze import verify_freeze_manifest
from govagent_kr.model_config import (
    REQUIRED_GENERATE_CONFIG,
    frozen_model_config_sha256,
    normalize_frozen_model_config,
)
from govagent_kr.models import SUT_TERMINAL_ERROR_GRADES, EpisodeOutcome, GoldRecord, ReviewStatus
from govagent_kr.scorer import grade_submission

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COMPOSE_PATH = PROJECT_ROOT / "compose.yaml"
DEFAULT_FROZEN_COMPOSE_PATH = PROJECT_ROOT / "compose.frozen.yaml"

MAX_MODEL_REQUESTS = 12
MAX_TOOL_CALLS = 24
WALL_CLOCK_SECONDS = 180

SYSTEM_PROMPT = """\
You are solving a Korean-law temporal-citation evaluation. Work only from the case
JSON and the four offline law tools exposed in this environment. Do not use current
law as a substitute for the version in force at the controlling event.

Available functions:
- law_list_versions(instrument_id)
- law_search_provisions(history_id, query)
- law_get_provision(history_id, provision_locator)
- law_get_supplementary_provisions(history_id)

Before answering, identify the controlling date, select the applicable history, and
retrieve every passage you quote. A quote must be the complete `passage` returned by
the exact history_id and provision_locator; do not paraphrase or repair it.
When the tool exposes synthetic child locators such as `#본문` or `#3)`, cite the
smallest locator passages needed for the conclusion. A parent aggregate passage is
not a substitute for its available child passages, and unnecessary overcitation can
fail the task.

Return exactly one JSON object, without Markdown or surrounding prose, with this
shape:
{
  "status": "answered | insufficient_facts | no_applicable_provision",
  "conclusion": "affirmative | negative | indeterminate",
  "selected_provisions": [
    {
      "instrument_id": "...",
      "history_id": "...",
      "provision_locator": "...",
      "quote": "complete tool-returned passage"
    }
  ],
  "missing_facts": [],
  "explanation": "brief reasoning"
}

For `answered`, selected_provisions must be non-empty, missing_facts empty, and
conclusion must be affirmative or negative according to the substantive answer. For
`no_applicable_provision`, conclusion must be negative and both arrays must be empty.
For `insufficient_facts`, conclusion must be indeterminate, selected_provisions must
be empty, and missing_facts must contain only the applicable stable key(s) from this
contract:
- vulnerable_employment_ratio_compliance
- statutory_digital_service_selection
- pre_effective_estimate_received
- unfinished_at_effective_date
"""

_SANDBOX_PREFLIGHT = """\
from pathlib import Path

assert Path('/opt/law/index.json').is_file(), 'public fixture is not mounted'
for candidate in (
    '/opt/gold',
    '/data/gold.jsonl',
    '/app/gold.jsonl',
    '/workspace/gov-agent/data/gold.jsonl',
):
    assert not Path(candidate).exists(), f'gold path is visible: {candidate}'
ipv4_routes = Path('/proc/net/route').read_text().splitlines()[1:]
assert all(line.split()[0] == 'lo' for line in ipv4_routes if line.split()), \
    f'non-loopback IPv4 route(s): {ipv4_routes}'
ipv6_routes = Path('/proc/net/ipv6_route').read_text().splitlines()
assert all(line.split()[-1] == 'lo' for line in ipv6_routes if line.split()), \
    f'non-loopback IPv6 route(s): {ipv6_routes}'
"""


def _path(value: str | Path | None, default: Path) -> Path:
    path = Path(value) if value is not None else default
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _validate_eval_freeze(
    freeze: dict[str, Any],
    *,
    freeze_kind: str,
    analysis_eligible: bool,
    smoke_eligible: bool,
) -> dict[str, dict[str, Any]]:
    label = "analysis" if analysis_eligible else "smoke"
    if freeze.get("profile_id") != PROFILE_ID or freeze.get("profile_version") != PROFILE_VERSION:
        raise ValueError(f"{label} freeze profile differs from this task")
    if (
        freeze.get("freeze_kind") != freeze_kind
        or freeze.get("analysis_eligible") is not analysis_eligible
        or freeze.get("smoke_eligible") is not smoke_eligible
    ):
        raise ValueError(f"{label} freeze has an invalid eligibility declaration")
    requested_model = freeze.get("requested_model")
    if not isinstance(requested_model, str) or not requested_model.strip():
        raise ValueError(f"{label} freeze requested_model must be non-empty")
    dataset_review = freeze.get("dataset_review")
    if not isinstance(dataset_review, dict) or dataset_review.get("analysis_ready") is not True:
        raise ValueError(f"{label} freeze does not record a completed dataset review")
    if dataset_review.get("pending_human_review") != 0:
        raise ValueError(f"{label} freeze records pending human gold review")
    git = freeze.get("git")
    if not isinstance(git, dict) or git.get("gov_agent_worktree_status") != "":
        raise ValueError(f"{label} freeze was not created from a clean gov-agent worktree")
    runtime = freeze.get("runtime")
    if not isinstance(runtime, dict) or runtime.get("inspect_ai") != importlib.metadata.version(
        "inspect-ai"
    ):
        raise ValueError(f"{label} freeze Inspect version differs from the runtime")
    container = freeze.get("container")
    image_id = container.get("image_id") if isinstance(container, dict) else None
    if not isinstance(image_id, str) or not image_id.startswith("sha256:"):
        raise ValueError(f"{label} freeze has no immutable SUT image ID")
    normalized = normalize_frozen_model_config(freeze.get("model_config"))
    if any(
        normalized["generate"].get(key) != value for key, value in REQUIRED_GENERATE_CONFIG.items()
    ):
        raise ValueError(
            f"{label} freeze must fix max_connections=1, max_retries=0, "
            "timeout=120, and attempt_timeout=120"
        )
    return normalized


def _validate_analysis_freeze(freeze: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return _validate_eval_freeze(
        freeze,
        freeze_kind="analysis_freeze",
        analysis_eligible=True,
        smoke_eligible=False,
    )


def _validate_smoke_freeze(freeze: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return _validate_eval_freeze(
        freeze,
        freeze_kind="smoke_freeze",
        analysis_eligible=False,
        smoke_eligible=True,
    )


def _message_for_bridge(message: Any) -> dict[str, Any]:
    """Serialize an Inspect message without forwarding Inspect-only identifiers."""

    payload = message.model_dump(mode="json", exclude_none=True)
    payload.pop("id", None)
    payload.pop("source", None)
    return payload


def _stderr_excerpt(value: str, *, limit: int = 4_000) -> str:
    value = value.strip()
    return value if len(value) <= limit else value[:limit] + "...<truncated>"


@agent(name="govagent_sandbox_agent")
def govagent_sandbox_agent(
    *,
    max_model_requests: int = MAX_MODEL_REQUESTS,
    max_tool_calls: int = MAX_TOOL_CALLS,
    timeout_seconds: int = WALL_CLOCK_SECONDS - 5,
) -> Agent:
    """Run the SUT in Docker while Inspect proxies only its model requests."""

    if max_model_requests < 1 or max_tool_calls < 1 or timeout_seconds < 1:
        raise ValueError("agent budgets must be positive")

    async def run(state: AgentState) -> AgentState:
        preflight = await sandbox().exec(
            ["python", "-I", "-c", _SANDBOX_PREFLIGHT],
            timeout=15,
            timeout_retry=False,
        )
        if not preflight.success:
            raise RuntimeError(
                "sandbox isolation preflight failed: " + _stderr_excerpt(preflight.stderr)
            )

        request = json.dumps(
            {"messages": [_message_for_bridge(message) for message in state.messages]},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        bridged_state: AgentState
        async with sandbox_agent_bridge(
            state,
            web_search=False,
            code_execution=False,
            client_mcp_servers=False,
            forward_generation_config=False,
        ) as bridge:
            result = await sandbox().exec(
                [
                    "python",
                    "/app/agent.py",
                    "--max-model-requests",
                    str(max_model_requests),
                    "--max-tool-calls",
                    str(max_tool_calls),
                ],
                input=request,
                env={
                    "OPENAI_API_KEY": "inspect-sandbox-bridge",
                    "OPENAI_BASE_URL": f"http://127.0.0.1:{bridge.port}/v1",
                    "GOVAGENT_FIXTURE_ROOT": "/opt/law",
                    "NO_PROXY": "127.0.0.1,localhost",
                },
                timeout=timeout_seconds,
                timeout_retry=False,
            )
            if not result.success:
                raise RuntimeError(
                    f"sandbox agent exited {result.returncode}: "
                    + _stderr_excerpt(result.stderr or result.stdout)
                )
            bridged_state = bridge.state

        if not bridged_state.output.completion.strip():
            raise RuntimeError("sandbox agent completed without a final submission")
        return bridged_state

    return run


@scorer(metrics=[mean()], name="govagent_strict_scorer")
def govagent_strict_scorer(*, fixture_root: str = str(DEFAULT_FIXTURE_ROOT)) -> Scorer:
    """Adapt the pure deterministic grader to Inspect's scorer protocol."""

    store = FixtureStore(Path(fixture_root))

    async def score(state: TaskState, target: Target) -> Score:
        gold = GoldRecord.model_validate_json(target.text, strict=True)
        raw_submission = state.output.completion
        result = grade_submission(
            raw_submission,
            gold,
            store,
            episode_outcome=EpisodeOutcome.COMPLETED,
        )
        value = result.core_success if result.core_success is not None else 0
        explanation = "pass" if value == 1 else ",".join(result.failure_codes)
        return Score(
            value=value,
            answer=raw_submission,
            explanation=explanation,
            metadata=result.model_dump(mode="json"),
        )

    return score


def _samples(
    *,
    cases_path: Path,
    gold_path: Path,
    allow_pending_gold: bool,
) -> list[Sample]:
    cases = load_cases(cases_path)
    gold_records = load_gold(gold_path)
    gold_by_id = {record.item_id: record for record in gold_records}
    if len(gold_by_id) != len(gold_records):
        raise ValueError("gold item_id values are not unique")
    case_ids = {case.item_id for case in cases}
    if case_ids != set(gold_by_id):
        raise ValueError("cases and gold must have a one-to-one item_id mapping")

    pending = [
        gold.item_id for gold in gold_records if gold.review.status != ReviewStatus.ADJUDICATED
    ]
    if pending and not allow_pending_gold:
        raise ValueError(
            "gold is not fully adjudicated; development smoke runs must explicitly set "
            f"allow_pending_gold=True ({len(pending)} item(s))"
        )

    samples: list[Sample] = []
    for case in cases:
        gold = gold_by_id[case.item_id]
        samples.append(
            Sample(
                id=case.item_id,
                input=[
                    ChatMessageSystem(content=SYSTEM_PROMPT),
                    ChatMessageUser(
                        content=canonical_json_bytes(case.input.model_dump(mode="json")).decode(
                            "utf-8"
                        )
                    ),
                ],
                target=canonical_json_bytes(gold.model_dump(mode="json")).decode("utf-8"),
                metadata={
                    "family_id": case.family_id,
                    "boundary_id": case.metadata.boundary_id,
                    "variant": case.metadata.variant.value,
                    "candidate_gold": gold.review.status != ReviewStatus.ADJUDICATED,
                    "case_originated_as_candidate": case.metadata.candidate_gold,
                },
            )
        )
    return samples


@task(name="govagent_kr")
def govagent_kr(
    cases_path: str | None = None,
    gold_path: str | None = None,
    fixture_root: str | None = None,
    compose_path: str | None = None,
    freeze_manifest: str | None = None,
    smoke_freeze_manifest: str | None = None,
    trials: int = 3,
    allow_pending_gold: bool = False,
) -> Task:
    """Create the frozen temporal-citation evaluation task.

    Pending candidate gold is rejected by default. Set ``allow_pending_gold=True``
    only for development and mock-model smoke tests; it does not make the resulting
    scores analysis-ready.
    """

    if trials < 1:
        raise ValueError("trials must be positive")
    resolved_cases = _path(cases_path, DEFAULT_CASES_PATH)
    resolved_gold = _path(gold_path, DEFAULT_GOLD_PATH)
    resolved_fixture = _path(fixture_root, DEFAULT_FIXTURE_ROOT)
    resolved_compose = _path(compose_path, DEFAULT_COMPOSE_PATH)
    for required in (resolved_cases, resolved_gold, resolved_fixture, resolved_compose):
        if not required.exists():
            raise FileNotFoundError(required)

    if freeze_manifest is not None and smoke_freeze_manifest is not None:
        raise ValueError("analysis and smoke freeze manifests are mutually exclusive")
    if smoke_freeze_manifest is not None and allow_pending_gold:
        raise ValueError("smoke receipt runs require adjudicated gold")

    samples = _samples(
        cases_path=resolved_cases,
        gold_path=resolved_gold,
        allow_pending_gold=allow_pending_gold,
    )
    freeze: dict[str, Any] | None = None
    frozen_model_config: dict[str, dict[str, Any]] | None = None
    if freeze_manifest is not None:
        freeze = verify_freeze_manifest(
            Path(freeze_manifest),
            require_analysis=True,
        )
        frozen_model_config = _validate_analysis_freeze(freeze)
    elif smoke_freeze_manifest is not None:
        freeze = verify_freeze_manifest(
            Path(smoke_freeze_manifest),
            require_smoke=True,
        )
        frozen_model_config = _validate_smoke_freeze(freeze)
    elif not allow_pending_gold:
        raise ValueError("an analysis freeze manifest is required for a non-development run")
    if freeze is not None:
        assert freeze is not None
        expected_paths = {
            "cases_path": (resolved_cases, DEFAULT_CASES_PATH.resolve()),
            "gold_path": (resolved_gold, DEFAULT_GOLD_PATH.resolve()),
            "fixture_root": (resolved_fixture, DEFAULT_FIXTURE_ROOT.resolve()),
            "compose_path": (resolved_compose, DEFAULT_FROZEN_COMPOSE_PATH.resolve()),
        }
        for name, (actual, expected) in expected_paths.items():
            if actual != expected:
                raise ValueError(f"frozen run {name} differs from the project path")
        expected_trials = 3 if freeze["analysis_eligible"] else 1
        if trials != expected_trials:
            noun = "trial" if expected_trials == 1 else "trials"
            raise ValueError(
                f"{freeze['freeze_kind']} requires exactly {expected_trials} {noun} per item"
            )
        frozen_image_id = freeze["container"]["image_id"]
        if os.environ.get("GOVAGENT_SUT_IMAGE") != frozen_image_id:
            raise ValueError(
                "GOVAGENT_SUT_IMAGE must equal the immutable image ID in the frozen manifest"
            )

    model = None
    generate_config = GenerateConfig()
    if frozen_model_config is not None:
        model = get_model(
            freeze["requested_model"],
            **frozen_model_config["model_args"],
        )
        generate_config = GenerateConfig.model_validate(frozen_model_config["generate"])
    return Task(
        dataset=MemoryDataset(samples, name="govagent-kr-v0.1"),
        solver=govagent_sandbox_agent(),
        scorer=govagent_strict_scorer(fixture_root=str(resolved_fixture)),
        model=model,
        config=generate_config,
        sandbox=("docker", str(resolved_compose)),
        epochs=Epochs(trials, reducer=[]),
        fail_on_error=False,
        score_on_error=False,
        message_limit=80,
        time_limit=WALL_CLOCK_SECONDS,
        version=PROFILE_VERSION,
        metadata={
            "profile_version": PROFILE_VERSION,
            "candidate_gold_allowed": allow_pending_gold,
            "model_request_budget": MAX_MODEL_REQUESTS,
            "tool_call_budget": MAX_TOOL_CALLS,
            "network": "none",
            "provider_web_search": False,
            "provider_code_execution": False,
            "remote_mcp": False,
            "freeze_id": freeze["freeze_id"] if freeze else None,
            "analysis_eligible": bool(freeze and freeze["analysis_eligible"]),
            "smoke_eligible": bool(freeze and freeze.get("smoke_eligible")),
            "frozen_requested_model": freeze["requested_model"] if freeze else None,
            "frozen_model_config": frozen_model_config,
            "frozen_model_config_sha256": (
                frozen_model_config_sha256(freeze["model_config"]) if freeze else None
            ),
            "frozen_container_image_id": (
                freeze["container"]["image_id"] if freeze and freeze.get("container") else None
            ),
        },
    )


__all__ = [
    "DEFAULT_COMPOSE_PATH",
    "DEFAULT_FROZEN_COMPOSE_PATH",
    "SUT_TERMINAL_ERROR_GRADES",
    "SYSTEM_PROMPT",
    "frozen_model_config_sha256",
    "govagent_kr",
    "govagent_sandbox_agent",
    "govagent_strict_scorer",
    "normalize_frozen_model_config",
]
