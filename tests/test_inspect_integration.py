from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPOSE_PATH = PROJECT_ROOT / "compose.yaml"
DOCKER_SMOKE_COMPOSE_PATH = Path(os.environ.get("GOVAGENT_SMOKE_COMPOSE_PATH", str(COMPOSE_PATH)))
RUNTIME_STORE_PATH = PROJECT_ROOT / "runtime" / "law_store.py"
RUNTIME_AGENT_PATH = PROJECT_ROOT / "runtime" / "agent.py"
FIXTURE_ROOT = PROJECT_ROOT / "fixtures" / "local-contract-decree"


def _load_runtime_store_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("runtime_law_store", RUNTIME_STORE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def test_runtime_store_exposes_only_public_fields() -> None:
    module = _load_runtime_store_module()
    store = module.LawStore(FIXTURE_ROOT)
    versions = store.list_versions("KR-LAW-010098")
    assert len(versions) == 7
    assert versions == sorted(versions, key=lambda item: item["effective_date"])

    history_id = "KR-LAW-010098:204108:20180724"
    result = store.get_provision(history_id, "제25조제1항제5호마목")
    assert set(result) == {
        "instrument_id",
        "history_id",
        "provision_locator",
        "passage",
    }
    assert "exact_passage_hash" not in result
    assert store.search_provisions(history_id, "사회적기업")
    supplementary = store.get_supplementary_provisions(history_id)
    assert [item["provision_locator"] for item in supplementary] == ["부칙(대통령령 제29059호)"]


def test_runtime_disables_hidden_openai_client_retries() -> None:
    source = RUNTIME_AGENT_PATH.read_text(encoding="utf-8")
    assert "max_retries=0" in source
    assert "max_retries=5" not in source


def test_compose_has_no_network_and_mounts_no_gold() -> None:
    compose = yaml.safe_load(COMPOSE_PATH.read_text(encoding="utf-8"))
    service = compose["services"]["default"]
    assert service["network_mode"] == "none"
    assert service["read_only"] is True
    assert service["build"]["context"] == "./runtime"
    assert set(service.get("environment", {})).isdisjoint(
        {"OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"}
    )
    serialized = json.dumps(service, ensure_ascii=False).casefold()
    assert "gold" not in serialized
    volumes = service["volumes"]
    assert volumes == [
        {
            "type": "bind",
            "source": "./fixtures/local-contract-decree",
            "target": "/opt/law",
            "read_only": True,
        }
    ]


def test_task_keeps_gold_out_of_agent_input() -> None:
    pytest.importorskip("inspect_ai")
    from govagent_kr.task import govagent_kr

    eval_task = govagent_kr(trials=3, allow_pending_gold=True)
    assert eval_task.epochs == 3
    assert eval_task.epochs_reducer == []
    sample = eval_task.dataset[0]
    input_text = "\n".join(message.text for message in sample.input)
    target_text = sample.target
    assert sample.id not in input_text
    assert "expected_status" not in input_text
    assert "required_citations" not in input_text
    assert "expected_status" in target_text
    assert "required_citations" in target_text
    assert sample.files is None


def test_pending_gold_is_rejected_by_default(tmp_path: Path) -> None:
    pytest.importorskip("inspect_ai")
    from govagent_kr.task import govagent_kr

    pending_gold = tmp_path / "pending-gold.jsonl"
    records = []
    for line in (PROJECT_ROOT / "data" / "gold.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        record["review"] = {
            "status": "pending",
            "reviewer_1": None,
            "reviewer_2": None,
            "reviewer_1_verdict": None,
            "reviewer_2_verdict": None,
            "resolution": None,
        }
        records.append(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
    pending_gold.write_text("\n".join(records) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="not fully adjudicated"):
        govagent_kr(trials=1, gold_path=str(pending_gold))


def test_relative_task_paths_are_resolved_from_project_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("inspect_ai")
    from govagent_kr.task import DEFAULT_FROZEN_COMPOSE_PATH, _path

    monkeypatch.chdir(tmp_path)

    assert _path("compose.frozen.yaml", COMPOSE_PATH) == DEFAULT_FROZEN_COMPOSE_PATH


@pytest.mark.skipif(
    os.environ.get("GOVAGENT_RUN_DOCKER_SMOKE") != "1",
    reason="set GOVAGENT_RUN_DOCKER_SMOKE=1 to run the Docker/Agent Bridge smoke test",
)
def test_mockllm_sandbox_agent_bridge_smoke(tmp_path: Path) -> None:
    pytest.importorskip("inspect_ai")
    from inspect_ai import eval as inspect_eval

    from govagent_kr.task import govagent_kr

    subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            str(DOCKER_SMOKE_COMPOSE_PATH),
            "config",
            "--quiet",
        ],
        check=True,
        cwd=PROJECT_ROOT,
    )
    logs = inspect_eval(
        govagent_kr(
            trials=3,
            allow_pending_gold=True,
            compose_path=str(DOCKER_SMOKE_COMPOSE_PATH),
        ),
        model="mockllm/model",
        limit=1,
        display="none",
        log_dir=str(tmp_path),
        log_format="eval",
        log_buffer=1,
        log_model_api=True,
        max_samples=1,
    )
    assert len(logs) == 1
    log = logs[0]
    assert log.status == "success"
    assert log.samples is not None and len(log.samples) == 3
    assert {sample.epoch for sample in log.samples} == {1, 2, 3}
    for sample in log.samples:
        assert sample.error is None
        assert sample.output.completion == "Default output from mockllm/model"
        assert sample.scores is not None
        score = next(iter(sample.scores.values()))
        assert score.value == 0
        assert score.metadata is not None
        assert score.metadata["failure_codes"] == ["SCHEMA_INVALID"]
        assert any(event.event == "model" for event in sample.events)
        assert any(event.event == "sandbox" for event in sample.events)


@pytest.mark.skipif(
    os.environ.get("GOVAGENT_RUN_DOCKER_SMOKE") != "1",
    reason="set GOVAGENT_RUN_DOCKER_SMOKE=1 to run the Docker/Agent Bridge smoke test",
)
def test_mockllm_law_tool_round_trip_scores_positive(tmp_path: Path) -> None:
    pytest.importorskip("inspect_ai")
    from inspect_ai import eval as inspect_eval
    from inspect_ai.model import ModelOutput, get_model

    from govagent_kr.task import govagent_kr

    final_submission = json.dumps(
        {
            "status": "no_applicable_provision",
            "conclusion": "negative",
            "selected_provisions": [],
            "missing_facts": [],
            "explanation": "기준일에는 명시 근거가 아직 시행되지 않았다.",
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    model = get_model(
        "mockllm/model",
        memoize=False,
        custom_outputs=[
            ModelOutput.for_tool_call(
                model="mockllm/model",
                tool_name="law_list_versions",
                tool_arguments={"instrument_id": "KR-LAW-010098"},
                tool_call_id="law-versions-smoke",
            ),
            ModelOutput.from_content(model="mockllm/model", content=final_submission),
        ],
    )
    logs = inspect_eval(
        govagent_kr(
            trials=1,
            allow_pending_gold=True,
            compose_path=str(DOCKER_SMOKE_COMPOSE_PATH),
        ),
        model=model,
        limit=1,
        display="none",
        log_dir=str(tmp_path),
        log_format="eval",
        log_buffer=1,
        log_model_api=True,
        max_samples=1,
    )
    log = logs[0]
    assert log.status == "success"
    assert log.samples is not None and len(log.samples) == 1
    sample = log.samples[0]
    assert sample.error is None
    assert sample.scores is not None
    score = next(iter(sample.scores.values()))
    assert score.value == 1
    assert score.metadata is not None
    assert score.metadata["failure_codes"] == []
    tool_messages = [message for message in sample.messages if message.role == "tool"]
    assert len(tool_messages) == 1
    assert "KR-LAW-010098:201260:20180118" in tool_messages[0].text
    assert "exact_passage_hash" not in tool_messages[0].text
    assert sum(event.event == "model" for event in sample.events) == 2
