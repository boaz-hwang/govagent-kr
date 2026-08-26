import csv
import json
import stat
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from govagent_kr import review
from govagent_kr.review import (
    REVIEW_COLUMNS,
    SECONDARY_CLEAN_SAMPLE_SIZE,
    ReviewError,
    export_adjudication_template,
    export_blind_bundle,
    export_secondary_bundle,
    export_secondary_disagreements,
    reveal_grades,
    validate_adjudications,
    validate_primary_reviews,
    validate_secondary_reviews,
)


class FakeScore(BaseModel):
    value: int
    metadata: dict[str, Any]


def _sample(sample_id: str = "real-case-01", *, core_success: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        id=sample_id,
        epoch=1,
        uuid=f"uuid-{sample_id}",
        input={"question": "기준일의 조문을 찾으시오."},
        target={"secret_gold": True},
        messages=[{"role": "assistant", "content": "도구를 사용했습니다."}],
        output={"status": "answered", "selected_provisions": []},
        error=None,
        total_time=1.25,
        working_time=1.0,
        model_usage={"input_tokens": 100, "output_tokens": 20},
        turn_count=2,
        limit=None,
        scores={
            "govagent_strict_scorer": FakeScore(
                value=core_success,
                metadata={"core_success": core_success},
            )
        },
    )


def _patch_inspect_logs(monkeypatch: pytest.MonkeyPatch, samples: list[SimpleNamespace]) -> None:
    import inspect_ai.log

    log_info = SimpleNamespace(name="/frozen/run.eval")
    header = SimpleNamespace(eval=SimpleNamespace(run_id="run-001", model="frontier-model"))
    monkeypatch.setattr(inspect_ai.log, "list_eval_logs", lambda path, formats=None: [log_info])
    monkeypatch.setattr(
        inspect_ai.log,
        "read_eval_log",
        lambda info, header_only: header,
    )
    monkeypatch.setattr(
        inspect_ai.log,
        "read_eval_log_samples",
        lambda info: iter(samples),
    )

    def read_sample(log_file: str, *, id: str, epoch: int) -> SimpleNamespace:
        return next(
            sample for sample in samples if str(sample.id) == str(id) and sample.epoch == epoch
        )

    monkeypatch.setattr(inspect_ai.log, "read_eval_log_sample", read_sample)


def _export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    samples: list[SimpleNamespace] | None = None,
    *,
    name: str = "bundle",
) -> Path:
    samples = samples or [_sample()]
    _patch_inspect_logs(monkeypatch, samples)
    key_file = tmp_path / f"{name}.key"
    key_file.write_bytes(b"k" * 32)
    bundle = tmp_path / name
    export_blind_bundle(tmp_path / "logs", bundle, key_file)
    return bundle


def _complete_review(
    template: Path,
    *,
    verdict: str = "correct",
    environment_valid: str = "true",
    sut_error: str = "false",
    labels: str = "",
    primary_cause: str = "",
    reviewer_id: str = "reviewer-a",
) -> None:
    with template.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row.update(
            reviewer_id=reviewer_id,
            human_task_verdict=verdict,
            environment_valid=environment_valid,
            sut_error=sut_error,
            labels=labels,
            primary_cause=primary_cause,
            note="blind first-pass review",
        )
    with template.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _read_review_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_review_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _secondary_workflow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    item_count: int = 15,
    labeled_count: int = 3,
    name: str = "secondary-source",
) -> tuple[Path, Path, Path]:
    samples = [_sample(f"case-{index:02d}") for index in range(item_count)]
    source_bundle = _export(tmp_path, monkeypatch, samples, name=name)
    primary_csv = source_bundle / "primary-review.csv"
    _complete_review(primary_csv)
    rows = _read_review_rows(primary_csv)
    for row in rows[:labeled_count]:
        row.update(
            human_task_verdict="incorrect",
            sut_error="true",
            labels="SUT_ERROR",
            primary_cause="SUT_ERROR",
        )
    _write_review_rows(primary_csv, rows)
    secondary_bundle = tmp_path / f"{name}-bundle"
    export_secondary_bundle(primary_csv, source_bundle, secondary_bundle)
    return source_bundle, primary_csv, secondary_bundle


def test_export_is_blind_deterministic_with_fixed_key_and_private_map_is_protected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _export(tmp_path, monkeypatch, name="first")
    second = _export(tmp_path, monkeypatch, name="second")

    assert (first / "blind-items.jsonl").read_bytes() == (second / "blind-items.jsonl").read_bytes()
    public = json.loads((first / "blind-items.jsonl").read_text(encoding="utf-8"))
    assert public["blind_id"].startswith("R-")
    assert "real-case-01" not in json.dumps(public)
    assert "target" not in public
    assert "scores" not in public
    private_path = first / "private" / "identity-map.jsonl"
    assert stat.S_IMODE(private_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(private_path.parent.stat().st_mode) == 0o700
    manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["contains_target"] is False
    assert manifest["contains_grader_output"] is False
    assert "identity_map_sha256" in manifest


def test_export_rejects_duplicate_identity_and_blind_id_collision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    duplicate = _sample()
    _patch_inspect_logs(monkeypatch, [duplicate, duplicate])
    key = tmp_path / "duplicate.key"
    key.write_bytes(b"k" * 32)
    with pytest.raises(ReviewError, match="duplicate run/sample/epoch"):
        export_blind_bundle(tmp_path / "logs", tmp_path / "duplicate", key)

    samples = [_sample("case-a"), _sample("case-b")]
    _patch_inspect_logs(monkeypatch, samples)
    monkeypatch.setattr(review, "_blind_digest", lambda key, identity: "a" * 64)
    with pytest.raises(ReviewError, match="blind_id collision"):
        export_blind_bundle(tmp_path / "logs", tmp_path / "collision", key)


def test_export_rejects_short_key_and_nonempty_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_inspect_logs(monkeypatch, [_sample()])
    short_key = tmp_path / "short.key"
    short_key.write_bytes(b"short")
    with pytest.raises(ReviewError, match="at least 32 bytes"):
        export_blind_bundle(tmp_path / "logs", tmp_path / "short", short_key)

    output = tmp_path / "occupied"
    output.mkdir()
    (output / "existing").write_text("do not overwrite", encoding="utf-8")
    with pytest.raises(ReviewError, match="must be empty"):
        export_blind_bundle(tmp_path / "logs", output, short_key)


def test_export_requires_exactly_one_eval_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import inspect_ai.log

    monkeypatch.setattr(
        inspect_ai.log,
        "list_eval_logs",
        lambda path, formats=None: [
            SimpleNamespace(name="first.eval"),
            SimpleNamespace(name="second.eval"),
        ],
    )
    key = tmp_path / "blind.key"
    key.write_bytes(b"k" * 32)

    with pytest.raises(ReviewError, match=r"exactly one Inspect \.eval log"):
        export_blind_bundle(tmp_path / "logs", tmp_path / "bundle", key)


@pytest.mark.parametrize("bad_core", [True, "1", 2, -1])
def test_export_rejects_malformed_automatic_core_score(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad_core: object
) -> None:
    sample = _sample()
    sample.scores["govagent_strict_scorer"].metadata["core_success"] = bad_core
    _patch_inspect_logs(monkeypatch, [sample])
    key = tmp_path / "malformed-score.key"
    key.write_bytes(b"k" * 32)

    with pytest.raises(ReviewError, match="must be exactly 0, 1, or null"):
        export_blind_bundle(tmp_path / "logs", tmp_path / "malformed-score", key)


def test_valid_complete_primary_review_is_normalized_and_sorted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _export(tmp_path, monkeypatch)
    review_csv = bundle / "primary-review.csv"
    _complete_review(review_csv)

    records = validate_primary_reviews(review_csv, bundle / "private" / "identity-map.jsonl")

    assert len(records) == 1
    assert records[0]["grader_visible"] is False
    assert records[0]["human_task_verdict"] == "correct"
    assert records[0]["labels"] == []
    assert records[0]["primary_cause"] is None


@pytest.mark.parametrize("malformation", ["short", "extra"])
def test_malformed_csv_rows_raise_review_error_not_python_exceptions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, malformation: str
) -> None:
    bundle = _export(tmp_path, monkeypatch)
    template = bundle / "primary-review.csv"
    with template.open(encoding="utf-8", newline="") as handle:
        blind_id = next(csv.DictReader(handle))["blind_id"]
    header = ",".join(REVIEW_COLUMNS)
    if malformation == "short":
        row = f"{blind_id},reviewer-a,primary"
    else:
        row = f"{blind_id},reviewer-a,primary,false,correct,true,false,,,note,unexpected"
    template.write_text(f"{header}\n{row}\n", encoding="utf-8")

    with pytest.raises(ReviewError, match="malformed review CSV row"):
        validate_primary_reviews(template, bundle / "private" / "identity-map.jsonl")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("grader_visible", "true", "saw grader output"),
        ("review_stage", "secondary", "stage=primary"),
        ("human_task_verdict", "mostly_correct", "invalid human verdict"),
        ("sut_error", "maybe", "invalid sut_error"),
        ("labels", "NOT_A_CAUSE", "invalid root-cause label"),
        ("labels", "SUT_ERROR;SUT_ERROR", "duplicate root-cause label"),
        ("labels", "GRADER_ERROR", "grader-blind review cannot assign"),
        ("primary_cause", "SUT_ERROR", "primary_cause must be one of labels"),
    ],
)
def test_review_invariants_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
    message: str,
) -> None:
    bundle = _export(tmp_path, monkeypatch)
    review_csv = bundle / "primary-review.csv"
    _complete_review(review_csv)
    with review_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0][field] = value
    with review_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ReviewError, match=message):
        validate_primary_reviews(review_csv, bundle / "private" / "identity-map.jsonl")


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"human_task_verdict": "incorrect"}, "incorrect verdict requires a cause label"),
        (
            {"environment_valid": "false"},
            "environment_valid=false requires an environment error label",
        ),
        ({"sut_error": "true"}, "sut_error=true requires SUT_ERROR label"),
        (
            {"labels": "SUT_ERROR", "primary_cause": "SUT_ERROR"},
            "SUT_ERROR label requires sut_error=true",
        ),
    ],
)
def test_error_fields_cannot_be_mislabeled_as_clean_before_secondary_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changes: dict[str, str],
    message: str,
) -> None:
    bundle = _export(tmp_path, monkeypatch)
    review_csv = bundle / "primary-review.csv"
    _complete_review(review_csv)
    rows = _read_review_rows(review_csv)
    rows[0].update(changes)
    _write_review_rows(review_csv, rows)

    with pytest.raises(ReviewError, match=message):
        validate_primary_reviews(review_csv, bundle / "private" / "identity-map.jsonl")


@pytest.mark.parametrize("label", ["TOOL_OR_ENV_ERROR", "INFRA_ERROR"])
def test_environment_error_labels_require_invalid_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    label: str,
) -> None:
    bundle = _export(tmp_path, monkeypatch)
    review_csv = bundle / "primary-review.csv"
    _complete_review(
        review_csv,
        verdict="incorrect",
        environment_valid="true",
        labels=label,
        primary_cause=label,
    )

    with pytest.raises(ReviewError, match="environment error labels require"):
        validate_primary_reviews(review_csv, bundle / "private" / "identity-map.jsonl")


def test_reveal_joins_fixed_human_verdict_and_frozen_grade(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    samples = [_sample(core_success=1)]
    bundle = _export(tmp_path, monkeypatch, samples)
    review_csv = bundle / "primary-review.csv"
    _complete_review(review_csv)
    output = tmp_path / "revealed.jsonl"

    summary = reveal_grades(review_csv, bundle, output)

    assert summary["reviewed_items"] == 1
    assert summary["grader_agreement_items"] == 1
    assert summary["grader_agreement_rate"] == 1.0
    revealed = json.loads(output.read_text(encoding="utf-8"))
    assert revealed["grader_agrees"] is True
    assert revealed["automatic_grade"]["core_success"] == 1


def test_reveal_marks_invalid_environment_as_not_applicable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _export(tmp_path, monkeypatch)
    review_csv = bundle / "primary-review.csv"
    _complete_review(
        review_csv,
        verdict="incorrect",
        environment_valid="false",
        labels="TOOL_OR_ENV_ERROR",
        primary_cause="TOOL_OR_ENV_ERROR",
    )
    output = tmp_path / "revealed-invalid.jsonl"

    summary = reveal_grades(review_csv, bundle, output)

    assert summary["agreement_applicable_items"] == 0
    assert summary["grader_agreement_rate"] is None
    assert json.loads(output.read_text(encoding="utf-8"))["grader_agrees"] == "not_applicable"


def test_reveal_rejects_public_bundle_transcript_or_grade_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    samples = [_sample(core_success=1)]
    public_bundle = _export(tmp_path, monkeypatch, samples, name="public-tamper")
    public_review = public_bundle / "primary-review.csv"
    _complete_review(public_review)
    blind_path = public_bundle / "blind-items.jsonl"
    blind_path.write_bytes(blind_path.read_bytes() + b" \n")
    with pytest.raises(ReviewError, match="blind item bundle hash mismatch"):
        reveal_grades(public_review, public_bundle, tmp_path / "public-output.jsonl")

    transcript_bundle = _export(tmp_path, monkeypatch, samples, name="transcript-tamper")
    transcript_review = transcript_bundle / "primary-review.csv"
    _complete_review(transcript_review)
    samples[0].output = {"status": "insufficient_facts"}
    with pytest.raises(ReviewError, match="source transcript changed"):
        reveal_grades(
            transcript_review,
            transcript_bundle,
            tmp_path / "transcript-output.jsonl",
        )

    samples[0] = _sample(core_success=1)
    grade_bundle = _export(tmp_path, monkeypatch, samples, name="grade-tamper")
    grade_review = grade_bundle / "primary-review.csv"
    _complete_review(grade_review)
    samples[0].scores["govagent_strict_scorer"] = FakeScore(value=0, metadata={"core_success": 0})
    with pytest.raises(ReviewError, match="automatic grade changed"):
        reveal_grades(grade_review, grade_bundle, tmp_path / "grade-output.jsonl")


def test_reveal_rejects_changed_sample_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    samples = [_sample()]
    bundle = _export(tmp_path, monkeypatch, samples)
    review_csv = bundle / "primary-review.csv"
    _complete_review(review_csv)
    samples[0].uuid = "replacement-uuid"

    with pytest.raises(ReviewError, match="source sample identity changed"):
        reveal_grades(review_csv, bundle, tmp_path / "identity-output.jsonl")


def test_secondary_export_selects_all_labeled_plus_deterministic_ten_clean_without_verdicts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, primary_csv, secondary = _secondary_workflow(tmp_path, monkeypatch)
    primary_rows = _read_review_rows(primary_csv)
    labeled_ids = sorted(row["blind_id"] for row in primary_rows if row["labels"])
    clean_ids = sorted(row["blind_id"] for row in primary_rows if not row["labels"])
    expected_ids = sorted([*labeled_ids, *clean_ids[:SECONDARY_CLEAN_SAMPLE_SIZE]])
    items = [
        json.loads(line)
        for line in (secondary / "secondary-items.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert [item["blind_id"] for item in items] == expected_ids
    assert all("primary_review" not in item and "labels" not in item for item in items)
    manifest = json.loads((secondary / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["secondary_item_count"] == 13
    assert manifest["labeled_item_count"] == 3
    assert manifest["clean_sample_selected"] == 10
    assert manifest["contains_primary_review"] is False
    assert manifest["contains_grader_output"] is False
    template_rows = _read_review_rows(secondary / "secondary-review.csv")
    assert all(row["review_stage"] == "secondary" for row in template_rows)
    assert all(row["grader_visible"] == "false" for row in template_rows)

    repeated = tmp_path / "secondary-repeat"
    export_secondary_bundle(primary_csv, source, repeated)
    assert (secondary / "secondary-items.jsonl").read_bytes() == (
        repeated / "secondary-items.jsonl"
    ).read_bytes()
    assert (secondary / "secondary-review.csv").read_bytes() == (
        repeated / "secondary-review.csv"
    ).read_bytes()
    assert (secondary / "manifest.json").read_bytes() == (repeated / "manifest.json").read_bytes()


def test_secondary_review_requires_an_independent_reviewer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, primary_csv, secondary = _secondary_workflow(tmp_path, monkeypatch)
    review_csv = secondary / "secondary-review.csv"
    _complete_review(review_csv, reviewer_id=" REVIEWER-A ")

    with pytest.raises(ReviewError, match="must differ from primary reviewer"):
        validate_secondary_reviews(review_csv, primary_csv, source, secondary)

    _complete_review(review_csv, reviewer_id="reviewer-b")
    records = validate_secondary_reviews(review_csv, primary_csv, source, secondary)
    assert len(records) == 13
    assert all(record["review_stage"] == "secondary" for record in records)
    assert all(record["grader_visible"] is False for record in records)
    assert all(record["reviewer_id"] == "reviewer-b" for record in records)


@pytest.mark.parametrize(
    ("malformation", "message"),
    [
        ("stage", "stage=secondary"),
        ("grader", "secondary reviewer saw grader output"),
        ("missing", "review is incomplete"),
        ("extra", "unknown blind_id"),
    ],
)
def test_secondary_review_stage_blinding_and_target_completeness_are_enforced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    malformation: str,
    message: str,
) -> None:
    source, primary_csv, secondary = _secondary_workflow(tmp_path, monkeypatch)
    review_csv = secondary / "secondary-review.csv"
    _complete_review(review_csv, reviewer_id="reviewer-b")
    rows = _read_review_rows(review_csv)
    if malformation == "stage":
        rows[0]["review_stage"] = "primary"
    elif malformation == "grader":
        rows[0]["grader_visible"] = "true"
    elif malformation == "missing":
        rows.pop()
    else:
        source_ids = {
            json.loads(line)["blind_id"]
            for line in (source / "blind-items.jsonl").read_text(encoding="utf-8").splitlines()
        }
        selected_ids = {row["blind_id"] for row in rows}
        rows[0]["blind_id"] = sorted(source_ids - selected_ids)[0]
    _write_review_rows(review_csv, rows)

    with pytest.raises(ReviewError, match=message):
        validate_secondary_reviews(review_csv, primary_csv, source, secondary)


def test_secondary_validation_rejects_changed_primary_or_selected_transcript(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, primary_csv, secondary = _secondary_workflow(
        tmp_path, monkeypatch, name="fixed-primary"
    )
    review_csv = secondary / "secondary-review.csv"
    _complete_review(review_csv, reviewer_id="reviewer-b")
    primary_rows = _read_review_rows(primary_csv)
    primary_rows[0]["note"] = "changed after secondary selection"
    _write_review_rows(primary_csv, primary_rows)
    with pytest.raises(ReviewError, match="primary reviews changed after secondary selection"):
        validate_secondary_reviews(review_csv, primary_csv, source, secondary)

    source, primary_csv, secondary = _secondary_workflow(
        tmp_path, monkeypatch, name="fixed-transcript"
    )
    review_csv = secondary / "secondary-review.csv"
    _complete_review(review_csv, reviewer_id="reviewer-b")
    items_path = secondary / "secondary-items.jsonl"
    items_path.write_bytes(items_path.read_bytes() + b" \n")
    with pytest.raises(ReviewError, match="secondary item bundle hash mismatch"):
        validate_secondary_reviews(review_csv, primary_csv, source, secondary)


def test_disagreement_export_is_deterministic_and_never_auto_adjudicates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, primary_csv, secondary = _secondary_workflow(tmp_path, monkeypatch)
    primary_by_id = {row["blind_id"]: row for row in _read_review_rows(primary_csv)}
    review_csv = secondary / "secondary-review.csv"
    secondary_rows = _read_review_rows(review_csv)
    for row in secondary_rows:
        primary = primary_by_id[row["blind_id"]]
        row.update(
            reviewer_id="reviewer-b",
            human_task_verdict=primary["human_task_verdict"],
            environment_valid=primary["environment_valid"],
            sut_error=primary["sut_error"],
            labels=primary["labels"],
            primary_cause=primary["primary_cause"],
            note="independent second-pass review",
        )
    _write_review_rows(review_csv, secondary_rows)
    no_difference_path = tmp_path / "disagreements-none.json"
    no_difference = export_secondary_disagreements(
        primary_csv, review_csv, source, secondary, no_difference_path
    )
    assert no_difference["disagreement_items"] == 0
    assert no_difference["requires_adjudication"] is False
    assert json.loads(no_difference_path.read_text(encoding="utf-8"))["disagreements"] == []

    changed = secondary_rows[0]
    if changed["human_task_verdict"] == "correct":
        changed.update(
            human_task_verdict="incorrect",
            sut_error="true",
            labels="SUT_ERROR",
            primary_cause="SUT_ERROR",
        )
    else:
        changed.update(
            human_task_verdict="correct",
            sut_error="false",
            labels="",
            primary_cause="",
        )
    _write_review_rows(review_csv, secondary_rows)

    first_path = tmp_path / "disagreements-first.json"
    second_path = tmp_path / "disagreements-second.json"
    first = export_secondary_disagreements(primary_csv, review_csv, source, secondary, first_path)
    second = export_secondary_disagreements(primary_csv, review_csv, source, secondary, second_path)

    assert first == second
    assert first["disagreement_items"] == 1
    assert first["requires_adjudication"] is True
    assert first["automatic_adjudication"] is False
    assert first_path.read_bytes() == second_path.read_bytes()
    document = json.loads(first_path.read_text(encoding="utf-8"))
    assert document["automatic_adjudication"] is False
    assert len(document["disagreements"]) == 1
    disagreement = document["disagreements"][0]
    assert disagreement["blind_id"] == changed["blind_id"]
    assert "human_task_verdict" in disagreement["differing_fields"]
    assert "adjudicated" not in disagreement
    assert "automatic_grade" not in disagreement


def test_blind_bundle_never_exports_raw_sample_error_or_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sample = _sample()
    sample.error = SimpleNamespace(
        message="RuntimeError('SECRET GOLD DETAIL scorer exploded')",
        traceback="traceback contains SECRET GOLD DETAIL",
    )
    sample.scores = {}

    bundle = _export(tmp_path, monkeypatch, [sample], name="safe-error")
    payload = (bundle / "blind-items.jsonl").read_text(encoding="utf-8")

    assert "SECRET GOLD DETAIL" not in payload
    assert json.loads(payload)["infrastructure_error"] == {
        "present": True,
        "classification": "opaque_error",
    }


def test_blank_review_note_cannot_reach_grade_reveal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _export(tmp_path, monkeypatch)
    review_csv = bundle / "primary-review.csv"
    _complete_review(review_csv)
    rows = _read_review_rows(review_csv)
    rows[0]["note"] = "   "
    _write_review_rows(review_csv, rows)

    with pytest.raises(ReviewError, match="review note is required"):
        reveal_grades(review_csv, bundle, tmp_path / "revealed.jsonl")


def _disagreement_document() -> dict[str, Any]:
    primary = {
        "blind_id": "R-blind",
        "reviewer_id": "primary",
        "review_stage": "primary",
        "grader_visible": False,
        "human_task_verdict": "correct",
        "environment_valid": True,
        "sut_error": False,
        "labels": [],
        "primary_cause": None,
        "note": "primary evidence",
    }
    secondary = {
        **primary,
        "reviewer_id": "secondary",
        "review_stage": "secondary",
        "human_task_verdict": "incorrect",
        "sut_error": True,
        "labels": ["SUT_ERROR"],
        "primary_cause": "SUT_ERROR",
        "note": "secondary evidence",
    }
    return {
        "schema_version": "govagent-kr/review-disagreements-v0.1",
        "automatic_adjudication": False,
        "disagreements": [
            {
                "blind_id": "R-blind",
                "differing_fields": [
                    "human_task_verdict",
                    "sut_error",
                    "labels",
                    "primary_cause",
                ],
                "primary_review": primary,
                "secondary_review": secondary,
            }
        ],
    }


def test_adjudication_template_and_validator_are_blind_independent_and_complete(
    tmp_path: Path,
) -> None:
    disagreements = tmp_path / "disagreements.json"
    disagreements.write_text(json.dumps(_disagreement_document()), encoding="utf-8")
    template = tmp_path / "adjudications.jsonl"
    result = export_adjudication_template(disagreements, template)
    assert result["adjudication_items"] == 1
    record = json.loads(template.read_text(encoding="utf-8"))
    assert record["review_stage"] == "adjudication"
    assert record["grader_visible"] is False

    record.update(
        adjudicator_id="independent",
        human_task_verdict="incorrect",
        environment_valid=True,
        sut_error=True,
        labels=["SUT_ERROR"],
        primary_cause="SUT_ERROR",
        note="resolved from transcript evidence",
    )
    template.write_text(json.dumps(record) + "\n", encoding="utf-8")
    assert len(validate_adjudications(template, disagreements)) == 1

    record["adjudicator_id"] = "PRIMARY"
    template.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ReviewError, match="independent"):
        validate_adjudications(template, disagreements)


def test_unresolved_disagreement_and_primary_only_cli_cannot_reveal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    disagreements = tmp_path / "disagreements.json"
    disagreements.write_text(json.dumps(_disagreement_document()), encoding="utf-8")
    with pytest.raises(ReviewError, match="require human adjudication"):
        validate_adjudications(None, disagreements)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "govagent-review",
            "reveal",
            "--reviews",
            "primary.csv",
            "--bundle",
            "bundle",
            "--output",
            "revealed.jsonl",
        ],
    )
    with pytest.raises(SystemExit, match="2"):
        review.main()
