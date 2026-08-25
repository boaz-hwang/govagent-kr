from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from collections.abc import Iterable
from contextlib import suppress
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .canonical import canonical_json_bytes, content_sha256
from .constants import DEFAULT_CASES_PATH, DEFAULT_FIXTURE_ROOT, DEFAULT_GOLD_PATH
from .dataset import dataset_integrity_report, load_cases, load_gold
from .fixtures import FixtureStore
from .models import (
    CaseRecord,
    GoldRecord,
    GoldReview,
    HumanTaskVerdict,
    ReviewStatus,
)
from .strict_json import StrictJSONError, parse_strict_json_object

EXPORT_SCHEMA_VERSION = "govagent-kr/gold-review-item-v0.1"
ADJUDICATION_SCHEMA_VERSION = "govagent-kr/gold-adjudication-v0.1"
MANIFEST_SCHEMA_VERSION = "govagent-kr/gold-review-bundle-v0.1"
FINAL_RECEIPT_SCHEMA_VERSION = "govagent-kr/gold-review-receipt-v0.1"

REVIEW_COLUMNS = (
    "schema_version",
    "item_id",
    "review_subject_sha256",
    "case_json",
    "candidate_gold_json",
    "fixture_passages_json",
    "reviewer_id",
    "verdict",
    "note",
)


class GoldReviewError(RuntimeError):
    """Raised when a candidate-gold human-review invariant is not satisfied."""


@dataclass(frozen=True)
class ReviewerBatch:
    reviewer_id: str
    verdicts: dict[str, HumanTaskVerdict]
    notes: dict[str, str]


@dataclass(frozen=True)
class Adjudication:
    final_gold: GoldRecord
    resolution: str
    adjudicator_id: str


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _jsonl_bytes(records: Iterable[dict[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(record) for record in records)


def _source_hash(path: Path) -> str:
    try:
        return content_sha256(path.read_bytes())
    except OSError as exc:
        raise GoldReviewError(f"cannot read review source: {path}") from exc


def _subject_evidence(
    case: CaseRecord,
    candidate: GoldRecord,
    fixture_passages: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "case": case.model_dump(mode="json"),
        "candidate_gold": candidate.model_dump(mode="json"),
        "fixture_passages": fixture_passages,
    }


def _subject_hash(evidence: dict[str, Any]) -> str:
    return content_sha256(canonical_json_bytes(evidence))


def _validated_candidate(
    store: FixtureStore, cases_path: Path, candidate_gold_path: Path
) -> tuple[dict[str, CaseRecord], dict[str, GoldRecord], dict[str, Any]]:
    report = dataset_integrity_report(store, cases_path, candidate_gold_path)
    if not report.get("ok"):
        errors = report.get("errors") or ["candidate dataset integrity failed"]
        raise GoldReviewError("candidate dataset is invalid: " + "; ".join(map(str, errors)))
    cases = load_cases(cases_path)
    candidates = load_gold(candidate_gold_path)
    if len(cases) != 20 or len(candidates) != 20:
        raise GoldReviewError("gold review requires exactly 20 cases and 20 candidate records")
    case_by_id = {record.item_id: record for record in cases}
    candidate_by_id = {record.item_id: record for record in candidates}
    if len(case_by_id) != 20 or len(candidate_by_id) != 20:
        raise GoldReviewError("gold review item_id values must be unique")
    if set(case_by_id) != set(candidate_by_id):
        raise GoldReviewError("cases and candidate gold must have identical item_id values")
    return case_by_id, candidate_by_id, report


def _family_fixture_passages(
    *,
    item_id: str,
    case_by_id: dict[str, CaseRecord],
    candidate_by_id: dict[str, GoldRecord],
    store: FixtureStore,
) -> list[dict[str, Any]]:
    family_id = case_by_id[item_id].family_id
    family_items = sorted(
        other_id for other_id, case in case_by_id.items() if case.family_id == family_id
    )
    locators: set[str] = set()
    histories: set[str] = set()
    for other_id in family_items:
        candidate = candidate_by_id[other_id]
        locators.update(
            citation.provision_locator
            for citation in candidate.required_citations + candidate.allowed_citations
        )
        histories.update(
            citation.history_id
            for citation in candidate.required_citations + candidate.allowed_citations
        )
        histories.update(candidate.forbidden_history_ids)

    candidate = candidate_by_id[item_id]
    required = {
        (citation.history_id, citation.provision_locator)
        for citation in candidate.required_citations
    }
    allowed = {
        (citation.history_id, citation.provision_locator)
        for citation in candidate.allowed_citations
    }
    passages: list[dict[str, Any]] = []
    for history_id in sorted(histories):
        for locator in sorted(locators):
            provision = store.maybe_get_provision(history_id, locator)
            if provision is None:
                continue
            identity = (history_id, locator)
            roles: list[str] = []
            if identity in required:
                roles.append("candidate_required")
            if identity in allowed:
                roles.append("candidate_allowed")
            if not roles:
                roles.append("family_context")
            passages.append(
                {
                    "roles": roles,
                    "instrument_id": provision["instrument_id"],
                    "history_id": provision["history_id"],
                    "provision_locator": provision["provision_locator"],
                    "exact_passage_hash": provision["exact_passage_hash"],
                    "passage": provision["passage"],
                }
            )
    return passages


def export_gold_review_bundle(
    *,
    fixtures_root: Path = DEFAULT_FIXTURE_ROOT,
    cases_path: Path = DEFAULT_CASES_PATH,
    candidate_gold_path: Path = DEFAULT_GOLD_PATH,
    output_dir: Path,
) -> dict[str, Any]:
    store = FixtureStore(fixtures_root)
    case_by_id, candidate_by_id, _ = _validated_candidate(store, cases_path, candidate_gold_path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise GoldReviewError(f"output directory must be empty: {output_dir}")

    records: list[dict[str, Any]] = []
    adjudications: list[dict[str, Any]] = []
    for item_id in sorted(case_by_id):
        passages = _family_fixture_passages(
            item_id=item_id,
            case_by_id=case_by_id,
            candidate_by_id=candidate_by_id,
            store=store,
        )
        evidence = _subject_evidence(case_by_id[item_id], candidate_by_id[item_id], passages)
        case_payload = evidence["case"]
        candidate_payload = evidence["candidate_gold"]
        subject_hash = _subject_hash(evidence)
        records.append(
            {
                "schema_version": EXPORT_SCHEMA_VERSION,
                "item_id": item_id,
                "review_subject_sha256": subject_hash,
                "case": case_payload,
                "candidate_gold": candidate_payload,
                "fixture_passages": passages,
                "reviewer_id": "",
                "verdict": "",
                "note": "",
            }
        )
        adjudications.append(
            {
                "schema_version": ADJUDICATION_SCHEMA_VERSION,
                "item_id": item_id,
                "review_subject_sha256": subject_hash,
                "final_gold": candidate_payload,
                "adjudicator_id": "",
                "resolution": "",
            }
        )

    jsonl_payload = _jsonl_bytes(records)
    adjudication_payload = _jsonl_bytes(adjudications)
    csv_rows = [
        {
            "schema_version": record["schema_version"],
            "item_id": record["item_id"],
            "review_subject_sha256": record["review_subject_sha256"],
            "case_json": _compact_json(record["case"]),
            "candidate_gold_json": _compact_json(record["candidate_gold"]),
            "fixture_passages_json": _compact_json(record["fixture_passages"]),
            "reviewer_id": "",
            "verdict": "",
            "note": "",
        }
        for record in records
    ]
    csv_buffer = StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=REVIEW_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(csv_rows)
    csv_payload = csv_buffer.getvalue().encode("utf-8")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "review-items.jsonl").write_bytes(jsonl_payload)
    (output_dir / "review-items.csv").write_bytes(csv_payload)
    (output_dir / "adjudication-template.jsonl").write_bytes(adjudication_payload)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "item_count": len(records),
        "source_cases_sha256": _source_hash(cases_path),
        "source_candidate_gold_sha256": _source_hash(candidate_gold_path),
        "review_items_jsonl_sha256": content_sha256(jsonl_payload),
        "review_items_csv_sha256": content_sha256(csv_payload),
        "adjudication_template_sha256": content_sha256(adjudication_payload),
        "contains_full_cases": True,
        "contains_full_candidate_gold": True,
        "contains_fixture_passages": True,
        "fixture_passages_are_candidate_navigation_context": True,
        "independent_official_source_search_required": True,
        "human_adjudication_required": True,
        "automatic_agreement": False,
        "ai_review": False,
        "distinct_reviewer_ids_are_identity_authentication": False,
    }
    (output_dir / "manifest.json").write_bytes(canonical_json_bytes(manifest))
    return manifest


def _validate_review_bundle_manifest(
    path: Path,
    *,
    cases_sha256: str,
    candidate_gold_sha256: str,
) -> str:
    try:
        payload = path.read_bytes()
        manifest = parse_strict_json_object(payload, max_bytes=1_000_000)
    except (OSError, StrictJSONError) as exc:
        raise GoldReviewError(f"invalid review bundle manifest: {path}") from exc
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise GoldReviewError("review bundle manifest schema is unsupported")
    if type(manifest.get("item_count")) is not int or manifest["item_count"] != 20:
        raise GoldReviewError("review bundle manifest must contain exactly 20 items")
    if manifest.get("source_cases_sha256") != cases_sha256:
        raise GoldReviewError("review bundle cases differ from the finalization input")
    if manifest.get("source_candidate_gold_sha256") != candidate_gold_sha256:
        raise GoldReviewError("review bundle candidate gold differs from the finalization input")
    required_flags = {
        "contains_full_cases": True,
        "contains_full_candidate_gold": True,
        "contains_fixture_passages": True,
        "fixture_passages_are_candidate_navigation_context": True,
        "independent_official_source_search_required": True,
        "human_adjudication_required": True,
        "automatic_agreement": False,
        "ai_review": False,
        "distinct_reviewer_ids_are_identity_authentication": False,
    }
    if any(manifest.get(key) is not expected for key, expected in required_flags.items()):
        raise GoldReviewError("review bundle manifest governance flags are invalid")
    sibling_hashes = {
        "review_items_jsonl_sha256": path.parent / "review-items.jsonl",
        "review_items_csv_sha256": path.parent / "review-items.csv",
        "adjudication_template_sha256": path.parent / "adjudication-template.jsonl",
    }
    for key, sibling in sibling_hashes.items():
        if manifest.get(key) != _source_hash(sibling):
            raise GoldReviewError(f"review bundle artifact differs from manifest: {sibling.name}")
    return content_sha256(payload)


def _review_rows_from_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != REVIEW_COLUMNS:
            raise GoldReviewError("review CSV columns differ from the exported template")
        rows: list[dict[str, Any]] = []
        for row_number, row in enumerate(reader, 2):
            if None in row or any(row.get(column) is None for column in REVIEW_COLUMNS):
                raise GoldReviewError(f"malformed review CSV row at row {row_number}")
            rows.append(row)
    return rows


def _review_rows_from_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_bytes().splitlines()
    except OSError as exc:
        raise GoldReviewError(f"cannot read reviewer file: {path}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            rows.append(parse_strict_json_object(line, max_bytes=1_000_000))
        except StrictJSONError as exc:
            raise GoldReviewError(f"invalid reviewer JSONL at line {line_number}") from exc
    return rows


def load_reviewer_verdicts(
    path: Path,
    *,
    expected_subjects: dict[str, str],
    expected_evidence: dict[str, dict[str, Any]],
) -> ReviewerBatch:
    is_csv = path.suffix.casefold() == ".csv"
    if is_csv:
        rows = _review_rows_from_csv(path)
    elif path.suffix.casefold() == ".jsonl":
        rows = _review_rows_from_jsonl(path)
    else:
        raise GoldReviewError("reviewer file must use .csv or .jsonl")

    seen: set[str] = set()
    reviewer_ids: set[str] = set()
    verdicts: dict[str, HumanTaskVerdict] = {}
    notes: dict[str, str] = {}
    for row_number, row in enumerate(rows, 1):
        item_id = row.get("item_id")
        if not isinstance(item_id, str) or item_id not in expected_subjects:
            raise GoldReviewError(f"unknown or invalid item_id in reviewer row {row_number}")
        if item_id in seen:
            raise GoldReviewError(f"duplicate reviewer verdict for item_id: {item_id}")
        seen.add(item_id)
        if row.get("schema_version") != EXPORT_SCHEMA_VERSION:
            raise GoldReviewError(f"review schema mismatch for item_id: {item_id}")
        if row.get("review_subject_sha256") != expected_subjects[item_id]:
            raise GoldReviewError(f"stale review subject for item_id: {item_id}")
        evidence = expected_evidence[item_id]
        if is_csv:
            evidence_matches = (
                row.get("case_json") == _compact_json(evidence["case"])
                and row.get("candidate_gold_json") == _compact_json(evidence["candidate_gold"])
                and row.get("fixture_passages_json") == _compact_json(evidence["fixture_passages"])
            )
        else:
            evidence_matches = (
                row.get("case") == evidence["case"]
                and row.get("candidate_gold") == evidence["candidate_gold"]
                and row.get("fixture_passages") == evidence["fixture_passages"]
            )
        if not evidence_matches:
            raise GoldReviewError(f"reviewer evidence was modified for item_id: {item_id}")
        reviewer_id = row.get("reviewer_id")
        if not isinstance(reviewer_id, str) or not reviewer_id:
            raise GoldReviewError(f"missing reviewer_id for item_id: {item_id}")
        if reviewer_id != reviewer_id.strip():
            raise GoldReviewError(f"reviewer_id must not have surrounding whitespace: {item_id}")
        reviewer_ids.add(reviewer_id)
        verdict_raw = row.get("verdict")
        if not isinstance(verdict_raw, str):
            raise GoldReviewError(f"missing reviewer verdict for item_id: {item_id}")
        try:
            verdict = HumanTaskVerdict(verdict_raw)
        except ValueError as exc:
            raise GoldReviewError(f"invalid reviewer verdict for item_id: {item_id}") from exc
        note = row.get("note", "")
        if not isinstance(note, str):
            raise GoldReviewError(f"reviewer note must be text for item_id: {item_id}")
        verdicts[item_id] = verdict
        notes[item_id] = note

    missing = sorted(set(expected_subjects) - seen)
    if missing:
        raise GoldReviewError(f"review is incomplete: {len(missing)} of 20 items missing")
    if len(seen) != 20:
        raise GoldReviewError("review must contain exactly 20 unique items")
    if len(reviewer_ids) != 1:
        raise GoldReviewError("one review file must contain exactly one reviewer_id")
    return ReviewerBatch(next(iter(reviewer_ids)), verdicts, notes)


def _gold_from_object(value: Any, *, item_id: str) -> GoldRecord:
    if not isinstance(value, dict):
        raise GoldReviewError(f"final_gold must be an object for item_id: {item_id}")
    try:
        return GoldRecord.model_validate_json(canonical_json_bytes(value), strict=True)
    except ValidationError as exc:
        raise GoldReviewError(f"invalid final_gold for item_id: {item_id}") from exc


def load_adjudications(
    path: Path,
    *,
    expected_subjects: dict[str, str],
) -> dict[str, Adjudication]:
    try:
        lines = path.read_bytes().splitlines()
    except OSError as exc:
        raise GoldReviewError(f"cannot read adjudication file: {path}") from exc
    result: dict[str, Adjudication] = {}
    expected_keys = {
        "schema_version",
        "item_id",
        "review_subject_sha256",
        "final_gold",
        "adjudicator_id",
        "resolution",
    }
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record = parse_strict_json_object(line, max_bytes=1_000_000)
        except StrictJSONError as exc:
            raise GoldReviewError(f"invalid adjudication JSONL at line {line_number}") from exc
        if set(record) != expected_keys:
            raise GoldReviewError(
                f"adjudication keys differ from the template at line {line_number}"
            )
        item_id = record.get("item_id")
        if not isinstance(item_id, str) or item_id not in expected_subjects:
            raise GoldReviewError(f"unknown adjudication item_id at line {line_number}")
        if item_id in result:
            raise GoldReviewError(f"duplicate adjudication item_id: {item_id}")
        if record.get("schema_version") != ADJUDICATION_SCHEMA_VERSION:
            raise GoldReviewError(f"adjudication schema mismatch for item_id: {item_id}")
        if record.get("review_subject_sha256") != expected_subjects[item_id]:
            raise GoldReviewError(f"stale adjudication subject for item_id: {item_id}")
        resolution = record.get("resolution")
        if not isinstance(resolution, str) or not resolution.strip():
            raise GoldReviewError(f"human resolution is required for item_id: {item_id}")
        adjudicator_id = record.get("adjudicator_id")
        if not isinstance(adjudicator_id, str) or not adjudicator_id:
            raise GoldReviewError(f"adjudicator_id is required for item_id: {item_id}")
        if adjudicator_id != adjudicator_id.strip():
            raise GoldReviewError(f"adjudicator_id must not have surrounding whitespace: {item_id}")
        final_gold = _gold_from_object(record.get("final_gold"), item_id=item_id)
        if final_gold.item_id != item_id:
            raise GoldReviewError(f"outer and final_gold item_id differ for item_id: {item_id}")
        if final_gold.review != GoldReview():
            raise GoldReviewError(
                f"final_gold review must remain the pending placeholder for item_id: {item_id}"
            )
        result[item_id] = Adjudication(final_gold, resolution.strip(), adjudicator_id)
    missing = sorted(set(expected_subjects) - set(result))
    if missing:
        raise GoldReviewError(f"adjudication is incomplete: {len(missing)} of 20 items missing")
    if len(result) != 20:
        raise GoldReviewError("adjudication must contain exactly 20 unique items")
    return result


def _write_validated_gold_atomically(
    *,
    records: list[GoldRecord],
    output_gold_path: Path,
    cases_path: Path,
    store: FixtureStore,
    expected_source_hashes: tuple[str, str],
    candidate_gold_path: Path,
) -> tuple[str, dict[str, Any]]:
    payload = _jsonl_bytes(record.model_dump(mode="json") for record in records)
    output_gold_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_gold_path.name}.", dir=output_gold_path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            os.fchmod(handle.fileno(), 0o644)
        report = dataset_integrity_report(store, cases_path, temporary_path)
        if not report.get("ok") or not report.get("analysis_ready"):
            errors = report.get("errors") or ["final dataset is not analysis-ready"]
            raise GoldReviewError("final gold integrity failed: " + "; ".join(map(str, errors)))
        staged = load_gold(temporary_path)
        if len(staged) != 20 or any(
            record.review.status != ReviewStatus.ADJUDICATED for record in staged
        ):
            raise GoldReviewError("all 20 final gold records must be adjudicated")
        for record in staged:
            try:
                HumanTaskVerdict(record.review.reviewer_1_verdict)
                HumanTaskVerdict(record.review.reviewer_2_verdict)
            except ValueError as exc:
                raise GoldReviewError("final gold contains an invalid human verdict") from exc
        current_source_hashes = (_source_hash(cases_path), _source_hash(candidate_gold_path))
        if current_source_hashes != expected_source_hashes:
            raise GoldReviewError("cases or candidate gold changed during finalization")
        os.replace(temporary_path, output_gold_path)
        directory_descriptor = os.open(output_gold_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        with suppress(FileNotFoundError):
            temporary_path.unlink()
        raise
    return content_sha256(payload), report


def finalize_gold_reviews(
    *,
    fixtures_root: Path = DEFAULT_FIXTURE_ROOT,
    cases_path: Path = DEFAULT_CASES_PATH,
    candidate_gold_path: Path = DEFAULT_GOLD_PATH,
    review_1_path: Path,
    review_2_path: Path,
    adjudication_path: Path,
    review_manifest_path: Path,
    output_gold_path: Path,
    receipt_output_path: Path | None = None,
) -> dict[str, Any]:
    store = FixtureStore(fixtures_root)
    source_hashes = (_source_hash(cases_path), _source_hash(candidate_gold_path))
    review_manifest_hash = _validate_review_bundle_manifest(
        review_manifest_path,
        cases_sha256=source_hashes[0],
        candidate_gold_sha256=source_hashes[1],
    )
    case_by_id, candidate_by_id, _ = _validated_candidate(store, cases_path, candidate_gold_path)
    expected_evidence: dict[str, dict[str, Any]] = {}
    for item_id in sorted(case_by_id):
        passages = _family_fixture_passages(
            item_id=item_id,
            case_by_id=case_by_id,
            candidate_by_id=candidate_by_id,
            store=store,
        )
        expected_evidence[item_id] = _subject_evidence(
            case_by_id[item_id], candidate_by_id[item_id], passages
        )
    expected_subjects = {
        item_id: _subject_hash(evidence) for item_id, evidence in expected_evidence.items()
    }
    if review_1_path.resolve() == review_2_path.resolve():
        raise GoldReviewError("reviewers must submit two separate review files")
    review_1 = load_reviewer_verdicts(
        review_1_path,
        expected_subjects=expected_subjects,
        expected_evidence=expected_evidence,
    )
    review_2 = load_reviewer_verdicts(
        review_2_path,
        expected_subjects=expected_subjects,
        expected_evidence=expected_evidence,
    )
    if review_1.reviewer_id.casefold() == review_2.reviewer_id.casefold():
        raise GoldReviewError("reviewer IDs must identify two different people")
    adjudications = load_adjudications(adjudication_path, expected_subjects=expected_subjects)

    final_records: list[GoldRecord] = []
    for item_id in sorted(expected_subjects):
        adjudication = adjudications[item_id]
        review = GoldReview(
            status=ReviewStatus.ADJUDICATED,
            reviewer_1=review_1.reviewer_id,
            reviewer_2=review_2.reviewer_id,
            reviewer_1_verdict=review_1.verdicts[item_id].value,
            reviewer_2_verdict=review_2.verdicts[item_id].value,
            resolution=(
                f"[adjudicator_id={adjudication.adjudicator_id}] {adjudication.resolution}"
            ),
        )
        final_records.append(adjudication.final_gold.model_copy(update={"review": review}))

    output_hash, report = _write_validated_gold_atomically(
        records=final_records,
        output_gold_path=output_gold_path,
        cases_path=cases_path,
        store=store,
        expected_source_hashes=source_hashes,
        candidate_gold_path=candidate_gold_path,
    )
    input_hashes = {
        "review_bundle_manifest_sha256": review_manifest_hash,
        "review_1_sha256": _source_hash(review_1_path),
        "review_2_sha256": _source_hash(review_2_path),
        "adjudication_sha256": _source_hash(adjudication_path),
    }
    receipt: dict[str, Any] = {
        "schema_version": FINAL_RECEIPT_SCHEMA_VERSION,
        "item_count": len(final_records),
        "reviewer_1": review_1.reviewer_id,
        "reviewer_2": review_2.reviewer_id,
        "two_distinct_ids_validated": True,
        "reviewer_identity_authenticated": False,
        "adjudicator_ids": sorted(
            {adjudication.adjudicator_id for adjudication in adjudications.values()}
        ),
        "human_adjudication_required": True,
        "automatic_agreement": False,
        "ai_review": False,
        "source_cases_sha256": source_hashes[0],
        "source_candidate_gold_sha256": source_hashes[1],
        **input_hashes,
        "review_artifact_set_sha256": content_sha256(canonical_json_bytes(input_hashes)),
        "output_gold_sha256": output_hash,
        "analysis_ready": report["analysis_ready"],
    }
    receipt["receipt_sha256"] = content_sha256(canonical_json_bytes(receipt))
    if receipt_output_path is not None:
        protected_paths = {
            review_1_path.resolve(),
            review_2_path.resolve(),
            adjudication_path.resolve(),
            review_manifest_path.resolve(),
            candidate_gold_path.resolve(),
            output_gold_path.resolve(),
        }
        if receipt_output_path.resolve() in protected_paths:
            raise GoldReviewError("receipt output must not overwrite review evidence or gold")
        receipt_output_path.parent.mkdir(parents=True, exist_ok=True)
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{receipt_output_path.name}.", dir=receipt_output_path.parent
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(file_descriptor, "wb") as handle:
                handle.write(canonical_json_bytes(receipt))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, receipt_output_path)
        except BaseException:
            with suppress(FileNotFoundError):
                temporary_path.unlink()
            raise
    return receipt


def _print(value: Any) -> None:
    sys.stdout.buffer.write(canonical_json_bytes(value))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export and finalize two-person human review of candidate gold"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export", help="export reviewer evidence and templates")
    export_parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURE_ROOT)
    export_parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    export_parser.add_argument("--candidate-gold", type=Path, default=DEFAULT_GOLD_PATH)
    export_parser.add_argument("--output", type=Path, required=True)

    finalize_parser = subparsers.add_parser(
        "finalize", help="validate two human reviews and explicit adjudication"
    )
    finalize_parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURE_ROOT)
    finalize_parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    finalize_parser.add_argument("--candidate-gold", type=Path, default=DEFAULT_GOLD_PATH)
    finalize_parser.add_argument("--review-1", type=Path, required=True)
    finalize_parser.add_argument("--review-2", type=Path, required=True)
    finalize_parser.add_argument("--adjudication", type=Path, required=True)
    finalize_parser.add_argument("--review-manifest", type=Path, required=True)
    finalize_parser.add_argument("--output-gold", type=Path, required=True)
    finalize_parser.add_argument("--receipt-output", type=Path, required=True)

    args = parser.parse_args()
    try:
        if args.command == "export":
            result = export_gold_review_bundle(
                fixtures_root=args.fixtures,
                cases_path=args.cases,
                candidate_gold_path=args.candidate_gold,
                output_dir=args.output,
            )
        else:
            result = finalize_gold_reviews(
                fixtures_root=args.fixtures,
                cases_path=args.cases,
                candidate_gold_path=args.candidate_gold,
                review_1_path=args.review_1,
                review_2_path=args.review_2,
                adjudication_path=args.adjudication,
                review_manifest_path=args.review_manifest,
                output_gold_path=args.output_gold,
                receipt_output_path=args.receipt_output,
            )
    except GoldReviewError as exc:
        parser.error(str(exc))
    _print(result)


if __name__ == "__main__":
    main()


__all__ = [
    "ADJUDICATION_SCHEMA_VERSION",
    "EXPORT_SCHEMA_VERSION",
    "FINAL_RECEIPT_SCHEMA_VERSION",
    "REVIEW_COLUMNS",
    "GoldReviewError",
    "export_gold_review_bundle",
    "finalize_gold_reviews",
    "load_adjudications",
    "load_reviewer_verdicts",
    "main",
]
