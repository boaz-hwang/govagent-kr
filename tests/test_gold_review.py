import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from govagent_kr.canonical import canonical_json_bytes, content_sha256
from govagent_kr.dataset import build_dataset, dataset_integrity_report, load_gold
from govagent_kr.fixtures import FixtureStore
from govagent_kr.gold_review import (
    REVIEW_COLUMNS,
    GoldReviewError,
    export_gold_review_bundle,
    finalize_gold_reviews,
)
from govagent_kr.models import ReviewStatus


@dataclass(frozen=True)
class Workflow:
    cases: Path
    candidate_gold: Path
    bundle: Path


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_bytes(b"".join(canonical_json_bytes(record) for record in records))


def _workflow(tmp_path: Path) -> Workflow:
    cases = tmp_path / "cases.jsonl"
    candidate_gold = tmp_path / "candidate-gold.jsonl"
    build_dataset(FixtureStore(), cases_path=cases, gold_path=candidate_gold)
    bundle = tmp_path / "review-bundle"
    export_gold_review_bundle(
        cases_path=cases,
        candidate_gold_path=candidate_gold,
        output_dir=bundle,
    )
    return Workflow(cases, candidate_gold, bundle)


def _complete_csv_review(
    template: Path,
    output: Path,
    reviewer_id: str,
    *,
    verdict_overrides: dict[str, str] | None = None,
) -> None:
    with template.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["reviewer_id"] = reviewer_id
        row["verdict"] = (verdict_overrides or {}).get(row["item_id"], "correct")
        row["note"] = f"독립 검수: {reviewer_id}"
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _complete_jsonl_review(
    template: Path,
    output: Path,
    reviewer_id: str,
    *,
    verdict_overrides: dict[str, str] | None = None,
) -> None:
    records = _jsonl(template)
    for record in records:
        record["reviewer_id"] = reviewer_id
        record["verdict"] = (verdict_overrides or {}).get(record["item_id"], "correct")
        record["note"] = f"독립 검수: {reviewer_id}"
    _write_jsonl(output, records)


def _complete_adjudication(
    template: Path,
    output: Path,
    *,
    note_change: tuple[str, str] | None = None,
    mutation: str | None = None,
) -> None:
    records = _jsonl(template)
    if note_change is not None:
        item_id, note = note_change
        target = next(record for record in records if record["item_id"] == item_id)
        target["final_gold"]["adjudication_note"] = note
    if mutation == "event":
        records[0]["final_gold"]["controlling_event_hash"] = "0" * 64
    elif mutation == "conclusion":
        target = next(
            record
            for record in records
            if record["final_gold"]["expected_status"] == "no_applicable_provision"
        )
        target["final_gold"]["expected_conclusion"] = "indeterminate"
    elif mutation in {"hash", "fixture"}:
        target = next(record for record in records if record["final_gold"]["required_citations"])
        citation = target["final_gold"]["required_citations"][0]
        if mutation == "hash":
            citation["exact_passage_hash"] = "0" * 64
        else:
            citation["provision_locator"] = "제999조"
    for record in records:
        record["adjudicator_id"] = "sme-adjudicator-c"
        record["resolution"] = f"두 검토 의견과 원문을 대조하여 {record['item_id']}을 확정함."
    _write_jsonl(output, records)


def _reviews(workflow: Workflow, tmp_path: Path) -> tuple[Path, Path]:
    review_1 = tmp_path / "review-1.jsonl"
    review_2 = tmp_path / "review-2.csv"
    _complete_jsonl_review(workflow.bundle / "review-items.jsonl", review_1, "sme-legal-a")
    _complete_csv_review(workflow.bundle / "review-items.csv", review_2, "sme-legal-b")
    return review_1, review_2


def test_export_contains_full_case_candidate_and_family_fixture_context(
    tmp_path: Path,
) -> None:
    workflow = _workflow(tmp_path)
    records = _jsonl(workflow.bundle / "review-items.jsonl")
    csv_rows: list[dict[str, str]]
    with (workflow.bundle / "review-items.csv").open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    adjudications = _jsonl(workflow.bundle / "adjudication-template.jsonl")
    manifest = json.loads((workflow.bundle / "manifest.json").read_text(encoding="utf-8"))

    assert len(records) == len(csv_rows) == len(adjudications) == 20
    assert records == sorted(records, key=lambda record: record["item_id"])
    assert all(record["case"]["item_id"] == record["item_id"] for record in records)
    assert all(record["candidate_gold"]["item_id"] == record["item_id"] for record in records)
    assert all(record["review_subject_sha256"] for record in records)
    assert all(json.loads(row["case_json"])["item_id"] == row["item_id"] for row in csv_rows)
    assert all(
        json.loads(row["candidate_gold_json"])["item_id"] == row["item_id"] for row in csv_rows
    )
    pre_item = next(
        record
        for record in records
        if record["item_id"] == "social-enterprise-direct-award--pre_effective"
    )
    assert pre_item["candidate_gold"]["required_citations"] == []
    assert pre_item["fixture_passages"]
    assert all("passage" in passage for passage in pre_item["fixture_passages"])
    assert all(record["resolution"] == "" for record in adjudications)
    assert all(record["adjudicator_id"] == "" for record in adjudications)
    assert manifest["contains_full_cases"] is True
    assert manifest["contains_full_candidate_gold"] is True
    assert manifest["contains_fixture_passages"] is True
    assert manifest["fixture_passages_are_candidate_navigation_context"] is True
    assert manifest["independent_official_source_search_required"] is True
    assert manifest["human_adjudication_required"] is True
    assert manifest["automatic_agreement"] is False
    assert manifest["ai_review"] is False
    assert manifest["distinct_reviewer_ids_are_identity_authentication"] is False

    second_bundle = tmp_path / "second-review-bundle"
    export_gold_review_bundle(
        cases_path=workflow.cases,
        candidate_gold_path=workflow.candidate_gold,
        output_dir=second_bundle,
    )
    for name in (
        "review-items.jsonl",
        "review-items.csv",
        "adjudication-template.jsonl",
        "manifest.json",
    ):
        assert (workflow.bundle / name).read_bytes() == (second_bundle / name).read_bytes()


def test_reviews_require_twenty_items_and_two_distinct_reviewer_ids(tmp_path: Path) -> None:
    workflow = _workflow(tmp_path)
    review_1 = tmp_path / "review-1.csv"
    review_2 = tmp_path / "review-2.csv"
    adjudication = tmp_path / "adjudication.jsonl"
    output = tmp_path / "final-gold.jsonl"
    _complete_csv_review(workflow.bundle / "review-items.csv", review_1, "Same-SME")
    _complete_csv_review(workflow.bundle / "review-items.csv", review_2, "same-sme")
    _complete_adjudication(workflow.bundle / "adjudication-template.jsonl", adjudication)

    with pytest.raises(GoldReviewError, match="two different people"):
        finalize_gold_reviews(
            cases_path=workflow.cases,
            candidate_gold_path=workflow.candidate_gold,
            review_1_path=review_1,
            review_2_path=review_2,
            adjudication_path=adjudication,
            review_manifest_path=workflow.bundle / "manifest.json",
            output_gold_path=output,
        )
    assert not output.exists()

    _complete_csv_review(workflow.bundle / "review-items.csv", review_2, "other-sme")
    with review_2.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    with review_2.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows[:-1])
    with pytest.raises(GoldReviewError, match="review is incomplete: 1 of 20"):
        finalize_gold_reviews(
            cases_path=workflow.cases,
            candidate_gold_path=workflow.candidate_gold,
            review_1_path=review_1,
            review_2_path=review_2,
            adjudication_path=adjudication,
            review_manifest_path=workflow.bundle / "manifest.json",
            output_gold_path=output,
        )
    assert not output.exists()


def test_disagreement_is_not_auto_resolved_and_explicit_human_adjudication_is_required(
    tmp_path: Path,
) -> None:
    workflow = _workflow(tmp_path)
    first_item = _jsonl(workflow.bundle / "review-items.jsonl")[0]["item_id"]
    review_1 = tmp_path / "review-1.jsonl"
    review_2 = tmp_path / "review-2.csv"
    _complete_jsonl_review(workflow.bundle / "review-items.jsonl", review_1, "sme-a")
    _complete_csv_review(
        workflow.bundle / "review-items.csv",
        review_2,
        "sme-b",
        verdict_overrides={first_item: "incorrect"},
    )
    output = tmp_path / "final-gold.jsonl"

    with pytest.raises(GoldReviewError, match="human resolution is required"):
        finalize_gold_reviews(
            cases_path=workflow.cases,
            candidate_gold_path=workflow.candidate_gold,
            review_1_path=review_1,
            review_2_path=review_2,
            adjudication_path=workflow.bundle / "adjudication-template.jsonl",
            review_manifest_path=workflow.bundle / "manifest.json",
            output_gold_path=output,
        )
    assert not output.exists()

    adjudication = tmp_path / "resolution-without-adjudicator.jsonl"
    records = _jsonl(workflow.bundle / "adjudication-template.jsonl")
    for record in records:
        record["resolution"] = "두 의견을 사람이 대조하여 확정함."
    _write_jsonl(adjudication, records)
    with pytest.raises(GoldReviewError, match="adjudicator_id is required"):
        finalize_gold_reviews(
            cases_path=workflow.cases,
            candidate_gold_path=workflow.candidate_gold,
            review_1_path=review_1,
            review_2_path=review_2,
            adjudication_path=adjudication,
            review_manifest_path=workflow.bundle / "manifest.json",
            output_gold_path=output,
        )
    assert not output.exists()


def test_explicit_adjudication_can_change_candidate_and_publishes_complete_gold(
    tmp_path: Path,
) -> None:
    workflow = _workflow(tmp_path)
    first_item = _jsonl(workflow.bundle / "review-items.jsonl")[0]["item_id"]
    review_1 = tmp_path / "review-1.jsonl"
    review_2 = tmp_path / "review-2.csv"
    _complete_jsonl_review(workflow.bundle / "review-items.jsonl", review_1, "sme-legal-a")
    _complete_csv_review(
        workflow.bundle / "review-items.csv",
        review_2,
        "sme-legal-b",
        verdict_overrides={first_item: "incorrect"},
    )
    adjudication = tmp_path / "adjudication.jsonl"
    changed_note = "SME가 원문을 대조하여 수정한 최종 adjudication note"
    _complete_adjudication(
        workflow.bundle / "adjudication-template.jsonl",
        adjudication,
        note_change=(first_item, changed_note),
    )
    output = tmp_path / "final-gold.jsonl"
    receipt_path = tmp_path / "gold-review-receipt.json"
    candidate_before = content_sha256(workflow.candidate_gold.read_bytes())

    receipt = finalize_gold_reviews(
        cases_path=workflow.cases,
        candidate_gold_path=workflow.candidate_gold,
        review_1_path=review_1,
        review_2_path=review_2,
        adjudication_path=adjudication,
        review_manifest_path=workflow.bundle / "manifest.json",
        output_gold_path=output,
        receipt_output_path=receipt_path,
    )

    final = load_gold(output)
    final_by_id = {record.item_id: record for record in final}
    report = dataset_integrity_report(FixtureStore(), workflow.cases, output)
    assert len(final) == 20
    assert report["ok"] is True
    assert report["analysis_ready"] is True
    assert all(record.review.status == ReviewStatus.ADJUDICATED for record in final)
    assert all(record.review.reviewer_1 == "sme-legal-a" for record in final)
    assert all(record.review.reviewer_2 == "sme-legal-b" for record in final)
    assert final_by_id[first_item].review.reviewer_1_verdict == "correct"
    assert final_by_id[first_item].review.reviewer_2_verdict == "incorrect"
    assert final_by_id[first_item].adjudication_note == changed_note
    assert final_by_id[first_item].review.resolution.startswith(
        "[adjudicator_id=sme-adjudicator-c] "
    )
    assert receipt["automatic_agreement"] is False
    assert receipt["ai_review"] is False
    assert receipt["reviewer_identity_authenticated"] is False
    assert receipt["adjudicator_ids"] == ["sme-adjudicator-c"]
    assert receipt["analysis_ready"] is True
    assert receipt["schema_version"] == "govagent-kr/gold-review-receipt-v0.1"
    assert receipt["review_1_sha256"] == content_sha256(review_1.read_bytes())
    assert receipt["review_2_sha256"] == content_sha256(review_2.read_bytes())
    assert receipt["adjudication_sha256"] == content_sha256(adjudication.read_bytes())
    assert receipt["review_bundle_manifest_sha256"] == content_sha256(
        (workflow.bundle / "manifest.json").read_bytes()
    )
    review_artifact_hashes = {
        key: receipt[key]
        for key in (
            "review_bundle_manifest_sha256",
            "review_1_sha256",
            "review_2_sha256",
            "adjudication_sha256",
        )
    }
    assert receipt["review_artifact_set_sha256"] == content_sha256(
        canonical_json_bytes(review_artifact_hashes)
    )
    assert json.loads(receipt_path.read_text(encoding="utf-8")) == receipt
    unsigned = dict(receipt)
    receipt_hash = unsigned.pop("receipt_sha256")
    assert receipt_hash == content_sha256(canonical_json_bytes(unsigned))
    assert content_sha256(workflow.candidate_gold.read_bytes()) == candidate_before


def test_finalization_rejects_tampered_review_bundle_artifact(tmp_path: Path) -> None:
    workflow = _workflow(tmp_path)
    review_1, review_2 = _reviews(workflow, tmp_path)
    adjudication = tmp_path / "adjudication.jsonl"
    _complete_adjudication(workflow.bundle / "adjudication-template.jsonl", adjudication)
    with (workflow.bundle / "review-items.jsonl").open("ab") as handle:
        handle.write(b"\n")

    with pytest.raises(GoldReviewError, match="review bundle artifact differs"):
        finalize_gold_reviews(
            cases_path=workflow.cases,
            candidate_gold_path=workflow.candidate_gold,
            review_1_path=review_1,
            review_2_path=review_2,
            adjudication_path=adjudication,
            review_manifest_path=workflow.bundle / "manifest.json",
            output_gold_path=tmp_path / "final-gold.jsonl",
        )


@pytest.mark.parametrize("mutation", ["event", "conclusion", "hash", "fixture"])
def test_integrity_failure_preserves_existing_output_and_removes_staging_file(
    tmp_path: Path, mutation: str
) -> None:
    workflow = _workflow(tmp_path)
    review_1, review_2 = _reviews(workflow, tmp_path)
    adjudication = tmp_path / "adjudication.jsonl"
    _complete_adjudication(
        workflow.bundle / "adjudication-template.jsonl",
        adjudication,
        mutation=mutation,
    )
    output = tmp_path / "final-gold.jsonl"
    sentinel = b"existing output must survive\n"
    output.write_bytes(sentinel)

    with pytest.raises(GoldReviewError, match="final gold integrity failed"):
        finalize_gold_reviews(
            cases_path=workflow.cases,
            candidate_gold_path=workflow.candidate_gold,
            review_1_path=review_1,
            review_2_path=review_2,
            adjudication_path=adjudication,
            review_manifest_path=workflow.bundle / "manifest.json",
            output_gold_path=output,
        )

    assert output.read_bytes() == sentinel
    assert list(tmp_path.glob(".final-gold.jsonl.*")) == []


def test_review_subject_hash_rejects_stale_case_or_candidate(tmp_path: Path) -> None:
    workflow = _workflow(tmp_path)
    review_1, review_2 = _reviews(workflow, tmp_path)
    adjudication = tmp_path / "adjudication.jsonl"
    _complete_adjudication(workflow.bundle / "adjudication-template.jsonl", adjudication)
    cases = _jsonl(workflow.cases)
    cases[0]["input"]["question"] += " 변경됨"
    _write_jsonl(workflow.cases, cases)

    with pytest.raises(GoldReviewError, match="review bundle cases differ"):
        finalize_gold_reviews(
            cases_path=workflow.cases,
            candidate_gold_path=workflow.candidate_gold,
            review_1_path=review_1,
            review_2_path=review_2,
            adjudication_path=adjudication,
            review_manifest_path=workflow.bundle / "manifest.json",
            output_gold_path=tmp_path / "final-gold.jsonl",
        )


@pytest.mark.parametrize(
    "column",
    ["case_json", "candidate_gold_json", "fixture_passages_json"],
)
def test_reviewer_csv_cannot_tamper_with_displayed_evidence(tmp_path: Path, column: str) -> None:
    workflow = _workflow(tmp_path)
    review_1 = tmp_path / "review-1.jsonl"
    review_2 = tmp_path / "review-2.csv"
    _complete_jsonl_review(workflow.bundle / "review-items.jsonl", review_1, "sme-legal-a")
    _complete_csv_review(workflow.bundle / "review-items.csv", review_2, "sme-legal-b")
    with review_2.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    evidence = json.loads(rows[0][column])
    if column == "case_json":
        evidence["input"]["question"] += " 변조"
    elif column == "candidate_gold_json":
        evidence["adjudication_note"] += " 변조"
    else:
        evidence[0]["passage"] += " 변조"
    rows[0][column] = json.dumps(
        evidence, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    with review_2.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    adjudication = tmp_path / "adjudication.jsonl"
    _complete_adjudication(workflow.bundle / "adjudication-template.jsonl", adjudication)

    with pytest.raises(GoldReviewError, match="reviewer evidence was modified"):
        finalize_gold_reviews(
            cases_path=workflow.cases,
            candidate_gold_path=workflow.candidate_gold,
            review_1_path=review_1,
            review_2_path=review_2,
            adjudication_path=adjudication,
            review_manifest_path=workflow.bundle / "manifest.json",
            output_gold_path=tmp_path / "final-gold.jsonl",
        )


def test_review_verdict_must_be_a_closed_human_enum(tmp_path: Path) -> None:
    workflow = _workflow(tmp_path)
    review_1 = tmp_path / "review-1.csv"
    review_2 = tmp_path / "review-2.csv"
    _complete_csv_review(workflow.bundle / "review-items.csv", review_1, "sme-a")
    _complete_csv_review(workflow.bundle / "review-items.csv", review_2, "sme-b")
    with review_1.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["verdict"] = "mostly_correct"
    with review_1.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    adjudication = tmp_path / "adjudication.jsonl"
    _complete_adjudication(workflow.bundle / "adjudication-template.jsonl", adjudication)

    with pytest.raises(GoldReviewError, match="invalid reviewer verdict"):
        finalize_gold_reviews(
            cases_path=workflow.cases,
            candidate_gold_path=workflow.candidate_gold,
            review_1_path=review_1,
            review_2_path=review_2,
            adjudication_path=adjudication,
            review_manifest_path=workflow.bundle / "manifest.json",
            output_gold_path=tmp_path / "final-gold.jsonl",
        )
