from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import io
import json
import secrets
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .canonical import canonical_json_bytes, content_sha256
from .models import SUT_TERMINAL_ERROR_GRADES, HumanTaskVerdict, RootCause
from .strict_json import StrictJSONError, parse_strict_json_object


class ReviewError(RuntimeError):
    """Raised when blinding or review invariants are not satisfied."""


REVIEW_COLUMNS = (
    "blind_id",
    "reviewer_id",
    "review_stage",
    "grader_visible",
    "human_task_verdict",
    "environment_valid",
    "sut_error",
    "labels",
    "primary_cause",
    "note",
)

SECONDARY_SCHEMA_VERSION = "govagent-kr/secondary-review-bundle-v0.1"
SECONDARY_SELECTION_ALGORITHM = "all-nonclean-plus-first-10-sorted-hmac-blind-id/v2"
SECONDARY_CLEAN_SAMPLE_SIZE = 10
DISAGREEMENT_SCHEMA_VERSION = "govagent-kr/review-disagreements-v0.1"
REVIEW_COMPARISON_FIELDS = (
    "human_task_verdict",
    "environment_valid",
    "sut_error",
    "labels",
    "primary_cause",
)
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
ENVIRONMENT_ERROR_LABELS = {
    RootCause.TOOL_OR_ENV_ERROR.value,
    RootCause.INFRA_ERROR.value,
}


def _json_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    return value


def _jsonl_bytes(records: Iterable[dict[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(record) for record in records)


def _write_private(path: Path, payload: bytes) -> None:
    path.write_bytes(payload)
    path.chmod(0o600)


def _blind_digest(key: bytes, identity: str) -> str:
    return hmac.new(key, identity.encode("utf-8"), hashlib.sha256).hexdigest()


def _safe_error_summary(error: Any) -> dict[str, Any] | None:
    if error is None:
        return None
    terminal = SUT_TERMINAL_ERROR_GRADES.get(getattr(error, "message", None))
    if terminal is None:
        return {"present": True, "classification": "opaque_error"}
    outcome, failure_code = terminal
    return {
        "present": True,
        "classification": outcome.value,
        "failure_code": failure_code,
    }


def _safe_sample_payload(sample: Any, blind_id: str, run_model: str) -> dict[str, Any]:
    """Export only evidence available before seeing target or grader output."""

    return {
        "blind_id": blind_id,
        "requested_model": run_model,
        "input": _json_value(sample.input),
        "messages": _json_value(sample.messages),
        "final_output": _json_value(sample.output),
        "infrastructure_error": _safe_error_summary(sample.error),
        "timing": {
            "total_time": sample.total_time,
            "working_time": sample.working_time,
        },
        "usage": _json_value(sample.model_usage),
        "turn_count": sample.turn_count,
        "limit": _json_value(sample.limit),
    }


def export_blind_bundle(log_dir: Path, output_dir: Path, key_file: Path | None = None) -> dict:
    from inspect_ai.log import list_eval_logs, read_eval_log, read_eval_log_samples

    if output_dir.exists() and any(output_dir.iterdir()):
        raise ReviewError(f"output directory must be empty: {output_dir}")
    log_infos = sorted(list_eval_logs(str(log_dir), formats=["eval"]), key=lambda item: item.name)
    if len(log_infos) != 1:
        raise ReviewError(f"exactly one Inspect .eval log is required under {log_dir}")
    if key_file is not None:
        key = key_file.read_bytes()
        if len(key) < 32:
            raise ReviewError("blinding key must contain at least 32 bytes")
    else:
        key = secrets.token_bytes(32)

    output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = output_dir / "private"
    private_dir.mkdir(mode=0o700)
    private_dir.chmod(0o700)
    if key_file is None:
        _write_private(private_dir / "blind-key.bin", key)

    staged: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    identities: set[tuple[str, str, int]] = set()
    blind_ids: set[str] = set()
    for log_info in log_infos:
        header = read_eval_log(log_info, header_only=True)
        for sample in read_eval_log_samples(log_info):
            identity_tuple = (header.eval.run_id, str(sample.id), sample.epoch)
            if identity_tuple in identities:
                raise ReviewError(f"duplicate run/sample/epoch identity: {identity_tuple}")
            identities.add(identity_tuple)
            raw_identity = "\x1f".join((header.eval.run_id, str(sample.id), str(sample.epoch)))
            digest = _blind_digest(key, raw_identity)
            blind_id = f"R-{digest[:16]}"
            if blind_id in blind_ids:
                raise ReviewError(f"blind_id collision: {blind_id}")
            blind_ids.add(blind_id)
            public = _safe_sample_payload(sample, blind_id, header.eval.model)
            private = {
                "blind_id": blind_id,
                "log_file": log_info.name,
                "eval_run_id": header.eval.run_id,
                "sample_id": sample.id,
                "epoch": sample.epoch,
                "sample_uuid": sample.uuid,
                "identity_hmac_sha256": digest,
                "public_sample_sha256": content_sha256(canonical_json_bytes(public)),
                "automatic_grade_sha256": content_sha256(
                    canonical_json_bytes(_automatic_grade(sample))
                ),
            }
            staged.append((digest, public, private))

    staged.sort(key=lambda item: item[0])
    public_records = [item[1] for item in staged]
    private_records = [item[2] for item in staged]
    blind_payload = _jsonl_bytes(public_records)
    identity_payload = _jsonl_bytes(private_records)
    (output_dir / "blind-items.jsonl").write_bytes(blind_payload)
    _write_private(private_dir / "identity-map.jsonl", identity_payload)

    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=REVIEW_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for record in public_records:
        writer.writerow(
            {
                "blind_id": record["blind_id"],
                "review_stage": "primary",
                "grader_visible": "false",
            }
        )
    (output_dir / "primary-review.csv").write_text(csv_buffer.getvalue(), encoding="utf-8")
    manifest = {
        "schema_version": "govagent-kr/review-bundle-v0.1",
        "blind_item_count": len(public_records),
        "blind_items_sha256": content_sha256(blind_payload),
        "identity_map_sha256": content_sha256(identity_payload),
        "review_template_sha256": content_sha256(csv_buffer.getvalue().encode("utf-8")),
        "source_log_count": len(log_infos),
        "contains_target": False,
        "contains_grader_output": False,
        "contains_real_case_id": False,
    }
    (output_dir / "manifest.json").write_bytes(canonical_json_bytes(manifest))
    return manifest


def _boolean(value: str, field: str) -> bool:
    lowered = value.strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    raise ReviewError(f"{field} must be true or false")


def _load_identity_map(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            record = parse_strict_json_object(line)
        except StrictJSONError as exc:
            raise ReviewError(f"invalid identity map at line {line_number}") from exc
        blind_id = record.get("blind_id")
        if not isinstance(blind_id, str) or blind_id in result:
            raise ReviewError(f"invalid identity map at line {line_number}")
        result[blind_id] = record
    return result


def _validate_review_csv(
    review_csv: Path,
    expected_ids: set[str],
    *,
    expected_stage: str,
    primary_reviewers: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    with review_csv.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != REVIEW_COLUMNS:
            raise ReviewError("review CSV columns differ from the frozen template")
        for row_number, row in enumerate(reader, 2):
            if None in row or any(row.get(column) is None for column in REVIEW_COLUMNS):
                raise ReviewError(f"malformed review CSV row at row {row_number}")
            blind_id = row["blind_id"]
            if blind_id not in expected_ids:
                raise ReviewError(f"unknown blind_id at row {row_number}: {blind_id}")
            if blind_id in seen:
                raise ReviewError(f"duplicate blind_id at row {row_number}: {blind_id}")
            seen.add(blind_id)
            reviewer_id = row["reviewer_id"].strip()
            if not reviewer_id:
                raise ReviewError(f"missing reviewer_id at row {row_number}")
            if row["review_stage"] != expected_stage:
                prefix = "first-pass" if expected_stage == "primary" else "secondary"
                raise ReviewError(
                    f"{prefix} review must have stage={expected_stage} at row {row_number}"
                )
            if _boolean(row["grader_visible"], "grader_visible"):
                raise ReviewError(
                    f"{expected_stage} reviewer saw grader output at row {row_number}"
                )
            if primary_reviewers is not None:
                primary_reviewer = primary_reviewers[blind_id].strip()
                if reviewer_id.casefold() == primary_reviewer.casefold():
                    raise ReviewError(
                        f"secondary reviewer must differ from primary reviewer at row {row_number}"
                    )
            try:
                verdict = HumanTaskVerdict(row["human_task_verdict"])
            except ValueError as exc:
                raise ReviewError(f"invalid human verdict at row {row_number}") from exc
            environment_valid = _boolean(row["environment_valid"], "environment_valid")
            sut_error_raw = row["sut_error"].strip().lower()
            if sut_error_raw not in {"true", "false", "indeterminate"}:
                raise ReviewError(f"invalid sut_error at row {row_number}")
            labels_raw = [item for item in row["labels"].split(";") if item]
            try:
                labels = [RootCause(value).value for value in labels_raw]
            except ValueError as exc:
                raise ReviewError(f"invalid root-cause label at row {row_number}") from exc
            if len(labels) != len(set(labels)):
                raise ReviewError(f"duplicate root-cause label at row {row_number}")
            if RootCause.GRADER_ERROR.value in labels:
                raise ReviewError(
                    f"grader-blind review cannot assign GRADER_ERROR at row {row_number}"
                )
            if verdict == HumanTaskVerdict.INCORRECT and not labels:
                raise ReviewError(f"incorrect verdict requires a cause label at row {row_number}")
            environment_labels = ENVIRONMENT_ERROR_LABELS.intersection(labels)
            if environment_valid and environment_labels:
                raise ReviewError(
                    f"environment error labels require environment_valid=false at row {row_number}"
                )
            if not environment_valid and not environment_labels:
                raise ReviewError(
                    "environment_valid=false requires an environment error label "
                    f"at row {row_number}"
                )
            if sut_error_raw == "true" and RootCause.SUT_ERROR.value not in labels:
                raise ReviewError(f"sut_error=true requires SUT_ERROR label at row {row_number}")
            if sut_error_raw != "true" and RootCause.SUT_ERROR.value in labels:
                raise ReviewError(f"SUT_ERROR label requires sut_error=true at row {row_number}")
            if sut_error_raw == "true" and (
                verdict != HumanTaskVerdict.INCORRECT or not environment_valid
            ):
                raise ReviewError(
                    "sut_error=true requires incorrect verdict and valid environment "
                    f"at row {row_number}"
                )
            primary = row["primary_cause"] or None
            if primary is not None and primary not in labels:
                raise ReviewError(f"primary_cause must be one of labels at row {row_number}")
            if labels and primary is None:
                raise ReviewError(f"nonempty labels require primary_cause at row {row_number}")
            if not labels and primary is not None:
                raise ReviewError(f"empty labels require blank primary_cause at row {row_number}")
            note = row["note"].strip()
            if not note:
                raise ReviewError(f"review note is required at row {row_number}")
            records.append(
                {
                    "blind_id": blind_id,
                    "reviewer_id": reviewer_id,
                    "review_stage": expected_stage,
                    "grader_visible": False,
                    "human_task_verdict": verdict.value,
                    "environment_valid": environment_valid,
                    "sut_error": (
                        sut_error_raw
                        if sut_error_raw == "indeterminate"
                        else sut_error_raw == "true"
                    ),
                    "labels": labels,
                    "primary_cause": primary,
                    "note": note,
                }
            )
    missing = sorted(expected_ids - seen)
    if missing:
        raise ReviewError(f"review is incomplete: {len(missing)} blind items missing")
    return sorted(records, key=lambda record: record["blind_id"])


def validate_primary_reviews(review_csv: Path, identity_map: Path) -> list[dict[str, Any]]:
    expected = _load_identity_map(identity_map)
    return _validate_review_csv(review_csv, set(expected), expected_stage="primary")


def _automatic_grade(sample: Any) -> dict[str, Any]:
    if getattr(sample, "limit", None) is not None:
        return {
            "core_success": None,
            "sample_error": None,
            "scores": _json_value(sample.scores),
        }
    if sample.error is not None:
        terminal = SUT_TERMINAL_ERROR_GRADES.get(sample.error.message)
        return {
            "core_success": 0 if terminal is not None else None,
            "sample_error": _safe_error_summary(sample.error),
            "scores": None,
        }
    scores = _json_value(sample.scores)
    core_values: list[int | None] = []
    if sample.scores:
        if set(sample.scores) != {"govagent_strict_scorer"}:
            raise ReviewError("automatic grade does not come from the frozen scorer")
        for score in sample.scores.values():
            metadata = score.metadata or {}
            if "core_success" in metadata:
                value = metadata["core_success"]
                if value is not None and (type(value) is not int or value not in {0, 1}):
                    raise ReviewError("automatic core_success must be exactly 0, 1, or null")
                if value is not None and score.value != value:
                    raise ReviewError("automatic score value differs from core_success metadata")
                core_values.append(value)
    if len(core_values) > 1:
        raise ReviewError("multiple automatic core_success values found for one sample")
    core_success = core_values[0] if len(core_values) == 1 else None
    return {"core_success": core_success, "sample_error": None, "scores": scores}


def _load_blind_items(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            record = parse_strict_json_object(line)
        except StrictJSONError as exc:
            raise ReviewError(f"invalid blind item at line {line_number}") from exc
        blind_id = record.get("blind_id")
        if not isinstance(blind_id, str) or blind_id in result:
            raise ReviewError(f"invalid blind item at line {line_number}")
        result[blind_id] = record
    return result


def _verify_bundle_integrity(bundle_dir: Path) -> dict[str, dict[str, Any]]:
    manifest_path = bundle_dir / "manifest.json"
    blind_path = bundle_dir / "blind-items.jsonl"
    identity_path = bundle_dir / "private" / "identity-map.jsonl"
    try:
        manifest = parse_strict_json_object(manifest_path.read_bytes())
    except StrictJSONError as exc:
        raise ReviewError("invalid review bundle manifest") from exc
    if manifest.get("schema_version") != "govagent-kr/review-bundle-v0.1":
        raise ReviewError("unsupported review bundle schema")
    blind_payload = blind_path.read_bytes()
    identity_payload = identity_path.read_bytes()
    if content_sha256(blind_payload) != manifest.get("blind_items_sha256"):
        raise ReviewError("blind item bundle hash mismatch")
    if content_sha256(identity_payload) != manifest.get("identity_map_sha256"):
        raise ReviewError("private identity map hash mismatch")
    public_records = _load_blind_items(blind_path)
    if type(manifest.get("blind_item_count")) is not int or manifest["blind_item_count"] != len(
        public_records
    ):
        raise ReviewError("blind item count mismatch")
    return public_records


def _normalized_reviews_sha256(records: list[dict[str, Any]]) -> str:
    return content_sha256(_jsonl_bytes(records))


def _is_clean_review(record: dict[str, Any]) -> bool:
    return (
        record["human_task_verdict"] == HumanTaskVerdict.CORRECT.value
        and record["environment_valid"] is True
        and record["sut_error"] is False
        and not record["labels"]
    )


def _secondary_selection_ids(primary_reviews: list[dict[str, Any]]) -> list[str]:
    mandatory = sorted(
        record["blind_id"] for record in primary_reviews if not _is_clean_review(record)
    )
    clean = sorted(record["blind_id"] for record in primary_reviews if _is_clean_review(record))
    selected_clean = clean[:SECONDARY_CLEAN_SAMPLE_SIZE]
    return sorted([*mandatory, *selected_clean])


def _review_template_bytes(blind_ids: list[str], stage: str) -> bytes:
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=REVIEW_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for blind_id in blind_ids:
        writer.writerow(
            {
                "blind_id": blind_id,
                "review_stage": stage,
                "grader_visible": "false",
            }
        )
    return csv_buffer.getvalue().encode("utf-8")


def export_secondary_bundle(
    primary_review_csv: Path,
    source_bundle_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Export a grader-blind second pass without exposing primary verdicts."""

    if output_dir.exists() and any(output_dir.iterdir()):
        raise ReviewError(f"output directory must be empty: {output_dir}")
    public_records = _verify_bundle_integrity(source_bundle_dir)
    identity_path = source_bundle_dir / "private" / "identity-map.jsonl"
    primary_reviews = validate_primary_reviews(primary_review_csv, identity_path)
    if set(public_records) != {record["blind_id"] for record in primary_reviews}:
        raise ReviewError("primary review and source bundle blind IDs differ")

    selected_ids = _secondary_selection_ids(primary_reviews)
    selected_records = [public_records[blind_id] for blind_id in selected_ids]
    selected_payload = _jsonl_bytes(selected_records)
    template_payload = _review_template_bytes(selected_ids, "secondary")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "secondary-items.jsonl").write_bytes(selected_payload)
    (output_dir / "secondary-review.csv").write_bytes(template_payload)

    mandatory_count = sum(not _is_clean_review(record) for record in primary_reviews)
    labeled_count = sum(bool(record["labels"]) for record in primary_reviews)
    clean_population = len(primary_reviews) - mandatory_count
    manifest = {
        "schema_version": SECONDARY_SCHEMA_VERSION,
        "selection_algorithm": SECONDARY_SELECTION_ALGORITHM,
        "source_bundle_manifest_sha256": content_sha256(
            (source_bundle_dir / "manifest.json").read_bytes()
        ),
        "source_primary_reviews_sha256": _normalized_reviews_sha256(primary_reviews),
        "source_item_count": len(primary_reviews),
        "secondary_item_count": len(selected_ids),
        "labeled_item_count": labeled_count,
        "mandatory_nonclean_item_count": mandatory_count,
        "clean_population_count": clean_population,
        "clean_sample_requested": SECONDARY_CLEAN_SAMPLE_SIZE,
        "clean_sample_selected": min(SECONDARY_CLEAN_SAMPLE_SIZE, clean_population),
        "secondary_items_sha256": content_sha256(selected_payload),
        "review_template_sha256": content_sha256(template_payload),
        "contains_primary_review": False,
        "contains_grader_output": False,
    }
    (output_dir / "manifest.json").write_bytes(canonical_json_bytes(manifest))
    return manifest


def _verify_secondary_bundle_integrity(
    secondary_bundle_dir: Path,
    source_bundle_dir: Path,
    primary_reviews: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    try:
        manifest = parse_strict_json_object((secondary_bundle_dir / "manifest.json").read_bytes())
    except StrictJSONError as exc:
        raise ReviewError("invalid secondary review bundle manifest") from exc
    if manifest.get("schema_version") != SECONDARY_SCHEMA_VERSION:
        raise ReviewError("unsupported secondary review bundle schema")
    if manifest.get("selection_algorithm") != SECONDARY_SELECTION_ALGORITHM:
        raise ReviewError("secondary selection algorithm mismatch")
    if manifest.get("source_bundle_manifest_sha256") != content_sha256(
        (source_bundle_dir / "manifest.json").read_bytes()
    ):
        raise ReviewError("secondary source bundle changed")
    if manifest.get("source_primary_reviews_sha256") != _normalized_reviews_sha256(primary_reviews):
        raise ReviewError("primary reviews changed after secondary selection")

    path = secondary_bundle_dir / "secondary-items.jsonl"
    payload = path.read_bytes()
    if manifest.get("secondary_items_sha256") != content_sha256(payload):
        raise ReviewError("secondary item bundle hash mismatch")
    secondary_records = _load_blind_items(path)
    expected_ids = set(_secondary_selection_ids(primary_reviews))
    if set(secondary_records) != expected_ids:
        raise ReviewError("secondary review target set is incomplete or contains extra items")
    if type(manifest.get("secondary_item_count")) is not int or manifest[
        "secondary_item_count"
    ] != len(expected_ids):
        raise ReviewError("secondary item count mismatch")

    source_records = _verify_bundle_integrity(source_bundle_dir)
    for blind_id, record in secondary_records.items():
        if record != source_records[blind_id]:
            raise ReviewError(f"secondary transcript differs from source for {blind_id}")
    return secondary_records


def validate_secondary_reviews(
    review_csv: Path,
    primary_review_csv: Path,
    source_bundle_dir: Path,
    secondary_bundle_dir: Path,
) -> list[dict[str, Any]]:
    identity_path = source_bundle_dir / "private" / "identity-map.jsonl"
    primary_reviews = validate_primary_reviews(primary_review_csv, identity_path)
    secondary_records = _verify_secondary_bundle_integrity(
        secondary_bundle_dir, source_bundle_dir, primary_reviews
    )
    primary_reviewers = {record["blind_id"]: record["reviewer_id"] for record in primary_reviews}
    return _validate_review_csv(
        review_csv,
        set(secondary_records),
        expected_stage="secondary",
        primary_reviewers=primary_reviewers,
    )


def _validate_normalized_review_record(
    record: dict[str, Any],
    *,
    expected_stage: str,
    context: str,
) -> None:
    if set(record) != set(REVIEW_COLUMNS):
        raise ReviewError(f"{context}: review fields differ from schema")
    if not isinstance(record.get("blind_id"), str) or not record["blind_id"]:
        raise ReviewError(f"{context}: blind_id is required")
    if not isinstance(record.get("reviewer_id"), str) or not record["reviewer_id"].strip():
        raise ReviewError(f"{context}: reviewer_id is required")
    if record.get("review_stage") != expected_stage or record.get("grader_visible") is not False:
        raise ReviewError(f"{context}: review stage or grader visibility is invalid")
    _validate_human_judgment(record, context=context)


def _validate_human_judgment(record: dict[str, Any], *, context: str) -> None:
    try:
        verdict = HumanTaskVerdict(record.get("human_task_verdict"))
    except ValueError as exc:
        raise ReviewError(f"{context}: invalid human_task_verdict") from exc
    environment_valid = record.get("environment_valid")
    if type(environment_valid) is not bool:
        raise ReviewError(f"{context}: invalid environment_valid")
    sut_error = record.get("sut_error")
    if type(sut_error) is not bool and sut_error != "indeterminate":
        raise ReviewError(f"{context}: invalid sut_error")
    labels = record.get("labels")
    allowed_labels = {label.value for label in RootCause}
    if (
        not isinstance(labels, list)
        or any(type(label) is not str or label not in allowed_labels for label in labels)
        or len(labels) != len(set(labels))
    ):
        raise ReviewError(f"{context}: invalid labels")
    if RootCause.GRADER_ERROR.value in labels:
        raise ReviewError(f"{context}: grader-blind review cannot assign GRADER_ERROR")
    if verdict == HumanTaskVerdict.INCORRECT and not labels:
        raise ReviewError(f"{context}: incorrect verdict requires a cause label")
    environment_labels = ENVIRONMENT_ERROR_LABELS.intersection(labels)
    if environment_valid and environment_labels:
        raise ReviewError(f"{context}: environment error labels require environment_valid=false")
    if not environment_valid and not environment_labels:
        raise ReviewError(f"{context}: environment_valid=false requires an environment error label")
    if sut_error is True and RootCause.SUT_ERROR.value not in labels:
        raise ReviewError(f"{context}: sut_error=true requires SUT_ERROR label")
    if sut_error is not True and RootCause.SUT_ERROR.value in labels:
        raise ReviewError(f"{context}: SUT_ERROR label requires sut_error=true")
    if sut_error is True and (
        verdict != HumanTaskVerdict.INCORRECT or environment_valid is not True
    ):
        raise ReviewError(
            f"{context}: sut_error=true requires incorrect verdict and valid environment"
        )
    primary = record.get("primary_cause")
    if primary is not None and primary not in labels:
        raise ReviewError(f"{context}: primary_cause is not in labels")
    if labels and primary is None:
        raise ReviewError(f"{context}: nonempty labels require primary_cause")
    if not labels and primary is not None:
        raise ReviewError(f"{context}: empty labels require blank primary_cause")
    if not isinstance(record.get("note"), str) or not record["note"].strip():
        raise ReviewError(f"{context}: review note is required")


def _load_disagreement_reviews(
    path: Path,
) -> tuple[set[str], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    try:
        document = parse_strict_json_object(path.read_bytes(), max_bytes=8 * 1024 * 1024)
    except (OSError, StrictJSONError) as exc:
        raise ReviewError("invalid disagreement artifact") from exc
    if set(document) != {"schema_version", "automatic_adjudication", "disagreements"}:
        raise ReviewError("disagreement artifact fields differ from schema")
    disagreements = document.get("disagreements")
    if (
        document.get("schema_version") != DISAGREEMENT_SCHEMA_VERSION
        or document.get("automatic_adjudication") is not False
        or not isinstance(disagreements, list)
    ):
        raise ReviewError("invalid disagreement artifact")
    primary_by_id: dict[str, dict[str, Any]] = {}
    secondary_by_id: dict[str, dict[str, Any]] = {}
    for index, disagreement in enumerate(disagreements, 1):
        context = f"disagreement {index}"
        if not isinstance(disagreement, dict) or set(disagreement) != {
            "blind_id",
            "differing_fields",
            "primary_review",
            "secondary_review",
        }:
            raise ReviewError(f"{context}: fields differ from schema")
        blind_id = disagreement.get("blind_id")
        if not isinstance(blind_id, str) or not blind_id or blind_id in primary_by_id:
            raise ReviewError(f"{context}: duplicate or invalid blind_id")
        primary = disagreement.get("primary_review")
        secondary = disagreement.get("secondary_review")
        if not isinstance(primary, dict) or not isinstance(secondary, dict):
            raise ReviewError(f"{context}: reviews must be objects")
        _validate_normalized_review_record(
            primary, expected_stage="primary", context=f"{blind_id}: primary"
        )
        _validate_normalized_review_record(
            secondary, expected_stage="secondary", context=f"{blind_id}: secondary"
        )
        if primary["blind_id"] != blind_id or secondary["blind_id"] != blind_id:
            raise ReviewError(f"{context}: embedded review identity differs")
        if primary["reviewer_id"].strip().casefold() == secondary["reviewer_id"].strip().casefold():
            raise ReviewError(f"{context}: primary and secondary reviewers must differ")
        expected_fields = [
            field
            for field in REVIEW_COMPARISON_FIELDS
            if (sorted(primary[field]) if field == "labels" else primary[field])
            != (sorted(secondary[field]) if field == "labels" else secondary[field])
        ]
        if not expected_fields or disagreement.get("differing_fields") != expected_fields:
            raise ReviewError(f"{context}: differing fields are not reproducible")
        primary_by_id[blind_id] = primary
        secondary_by_id[blind_id] = secondary
    return set(primary_by_id), primary_by_id, secondary_by_id


def validate_adjudication_records(
    records: list[dict[str, Any]],
    disagreement_ids: set[str],
    primary_by_id: dict[str, dict[str, Any]],
    secondary_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        blind_id = record.get("blind_id")
        if not isinstance(blind_id, str) or blind_id in by_id:
            raise ReviewError("adjudication has a duplicate or invalid blind_id")
        by_id[blind_id] = record
    if set(by_id) != disagreement_ids:
        raise ReviewError("adjudication target set differs from review disagreements")
    for blind_id, record in by_id.items():
        if set(record) != ADJUDICATION_FIELDS:
            raise ReviewError(f"{blind_id}: adjudication fields differ from schema")
        adjudicator_id = record.get("adjudicator_id")
        if not isinstance(adjudicator_id, str) or not adjudicator_id.strip():
            raise ReviewError(f"{blind_id}: adjudicator_id is required")
        if record.get("review_stage") != "adjudication":
            raise ReviewError(f"{blind_id}: review_stage must be adjudication")
        if record.get("grader_visible") is not False:
            raise ReviewError(f"{blind_id}: adjudicator must be grader-blind")
        adjudicator = adjudicator_id.strip().casefold()
        prior_reviewers = {
            primary_by_id[blind_id]["reviewer_id"].strip().casefold(),
            secondary_by_id[blind_id]["reviewer_id"].strip().casefold(),
        }
        if adjudicator in prior_reviewers:
            raise ReviewError(f"{blind_id}: adjudicator must be an independent person")
        _validate_human_judgment(record, context=f"{blind_id}: adjudication")
    return [by_id[blind_id] for blind_id in sorted(by_id)]


def export_adjudication_template(disagreements_path: Path, output_path: Path) -> dict[str, Any]:
    disagreement_ids, _, _ = _load_disagreement_reviews(disagreements_path)
    if output_path.exists():
        raise ReviewError(f"adjudication template output already exists: {output_path}")
    records = [
        {
            "blind_id": blind_id,
            "adjudicator_id": "",
            "review_stage": "adjudication",
            "grader_visible": False,
            "human_task_verdict": "",
            "environment_valid": None,
            "sut_error": "",
            "labels": [],
            "primary_cause": None,
            "note": "",
        }
        for blind_id in sorted(disagreement_ids)
    ]
    payload = _jsonl_bytes(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(payload)
    return {
        "adjudication_items": len(records),
        "grader_visible": False,
        "template_sha256": content_sha256(payload),
    }


def validate_adjudications(
    adjudications_path: Path | None,
    disagreements_path: Path,
) -> list[dict[str, Any]]:
    disagreement_ids, primary_by_id, secondary_by_id = _load_disagreement_reviews(
        disagreements_path
    )
    if adjudications_path is None:
        if disagreement_ids:
            raise ReviewError("review disagreements require human adjudication")
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        adjudications_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        try:
            records.append(parse_strict_json_object(line))
        except StrictJSONError as exc:
            raise ReviewError(f"invalid adjudication at line {line_number}") from exc
    return validate_adjudication_records(
        records,
        disagreement_ids,
        primary_by_id,
        secondary_by_id,
    )


def _secondary_disagreement_document(
    primary_reviews: list[dict[str, Any]],
    secondary_reviews: list[dict[str, Any]],
) -> dict[str, Any]:
    primary_by_id = {record["blind_id"]: record for record in primary_reviews}
    disagreements: list[dict[str, Any]] = []
    for secondary in secondary_reviews:
        primary = primary_by_id[secondary["blind_id"]]
        differing_fields: list[str] = []
        for field in REVIEW_COMPARISON_FIELDS:
            primary_value = sorted(primary[field]) if field == "labels" else primary[field]
            secondary_value = sorted(secondary[field]) if field == "labels" else secondary[field]
            if primary_value != secondary_value:
                differing_fields.append(field)
        if differing_fields:
            disagreements.append(
                {
                    "blind_id": secondary["blind_id"],
                    "differing_fields": differing_fields,
                    "primary_review": primary,
                    "secondary_review": secondary,
                }
            )
    return {
        "schema_version": DISAGREEMENT_SCHEMA_VERSION,
        "automatic_adjudication": False,
        "disagreements": disagreements,
    }


def validate_secondary_disagreement_artifact(
    primary_review_csv: Path,
    secondary_review_csv: Path,
    source_bundle_dir: Path,
    secondary_bundle_dir: Path,
    disagreements_path: Path,
) -> dict[str, Any]:
    identity_path = source_bundle_dir / "private" / "identity-map.jsonl"
    primary_reviews = validate_primary_reviews(primary_review_csv, identity_path)
    secondary_reviews = validate_secondary_reviews(
        secondary_review_csv,
        primary_review_csv,
        source_bundle_dir,
        secondary_bundle_dir,
    )
    expected = _secondary_disagreement_document(primary_reviews, secondary_reviews)
    try:
        actual = parse_strict_json_object(
            disagreements_path.read_bytes(), max_bytes=8 * 1024 * 1024
        )
    except (OSError, StrictJSONError) as exc:
        raise ReviewError("invalid disagreement artifact") from exc
    if actual != expected:
        raise ReviewError(
            "disagreement artifact differs from current primary and secondary reviews"
        )
    return expected


def export_secondary_disagreements(
    primary_review_csv: Path,
    secondary_review_csv: Path,
    source_bundle_dir: Path,
    secondary_bundle_dir: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Write disagreements for human adjudication without resolving them."""

    identity_path = source_bundle_dir / "private" / "identity-map.jsonl"
    primary_reviews = validate_primary_reviews(primary_review_csv, identity_path)
    secondary_reviews = validate_secondary_reviews(
        secondary_review_csv,
        primary_review_csv,
        source_bundle_dir,
        secondary_bundle_dir,
    )
    document = _secondary_disagreement_document(primary_reviews, secondary_reviews)
    disagreements = document["disagreements"]
    payload = canonical_json_bytes(document)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(payload)
    return {
        "secondary_reviewed_items": len(secondary_reviews),
        "disagreement_items": len(disagreements),
        "requires_adjudication": bool(disagreements),
        "automatic_adjudication": False,
        "disagreements_sha256": content_sha256(payload),
    }


def reveal_grades(review_csv: Path, bundle_dir: Path, output_path: Path) -> dict[str, Any]:
    from inspect_ai.log import read_eval_log_sample

    identity_path = bundle_dir / "private" / "identity-map.jsonl"
    public_records = _verify_bundle_integrity(bundle_dir)
    identities = _load_identity_map(identity_path)
    if set(public_records) != set(identities):
        raise ReviewError("public and private blind IDs differ")
    reviews = validate_primary_reviews(review_csv, identity_path)
    revealed: list[dict[str, Any]] = []
    agreements = 0
    applicable = 0
    for review in reviews:
        identity = identities[review["blind_id"]]
        sample = read_eval_log_sample(
            identity["log_file"], id=identity["sample_id"], epoch=identity["epoch"]
        )
        if (
            str(sample.id) != str(identity["sample_id"])
            or sample.epoch != identity["epoch"]
            or sample.uuid != identity["sample_uuid"]
        ):
            raise ReviewError(f"source sample identity changed for {review['blind_id']}")
        public = public_records[review["blind_id"]]
        recomputed_public = _safe_sample_payload(
            sample, review["blind_id"], public["requested_model"]
        )
        if content_sha256(canonical_json_bytes(recomputed_public)) != identity.get(
            "public_sample_sha256"
        ):
            raise ReviewError(f"source transcript changed for {review['blind_id']}")
        automatic = _automatic_grade(sample)
        if content_sha256(canonical_json_bytes(automatic)) != identity.get(
            "automatic_grade_sha256"
        ):
            raise ReviewError(f"automatic grade changed for {review['blind_id']}")
        if (
            review["human_task_verdict"] == HumanTaskVerdict.INDETERMINATE.value
            or not review["environment_valid"]
            or automatic["core_success"] is None
        ):
            grader_agrees: bool | str = "not_applicable"
        else:
            human_correct = review["human_task_verdict"] == HumanTaskVerdict.CORRECT.value
            grader_agrees = human_correct == (automatic["core_success"] == 1)
            applicable += 1
            agreements += int(grader_agrees)
        revealed.append(
            {
                "blind_id": review["blind_id"],
                "primary_review": review,
                "automatic_grade": automatic,
                "grader_agrees": grader_agrees,
            }
        )
    payload = _jsonl_bytes(revealed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(payload)
    return {
        "reviewed_items": len(revealed),
        "agreement_applicable_items": applicable,
        "grader_agreement_items": agreements,
        "grader_agreement_rate": agreements / applicable if applicable else None,
        "revealed_sha256": content_sha256(payload),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Blind transcript review workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export", help="export a grader-blind review bundle")
    export_parser.add_argument("--log-dir", type=Path, required=True)
    export_parser.add_argument("--output", type=Path, required=True)
    export_parser.add_argument("--key-file", type=Path)

    validate_parser = subparsers.add_parser("validate", help="validate completed primary reviews")
    validate_parser.add_argument("--reviews", type=Path, required=True)
    validate_parser.add_argument("--identity-map", type=Path, required=True)

    reveal_parser = subparsers.add_parser("reveal", help="join fixed human verdicts with grades")
    reveal_parser.add_argument("--reviews", type=Path, required=True)
    reveal_parser.add_argument("--bundle", type=Path, required=True)
    reveal_parser.add_argument("--secondary-reviews", type=Path, required=True)
    reveal_parser.add_argument("--secondary-bundle", type=Path, required=True)
    reveal_parser.add_argument("--disagreements", type=Path, required=True)
    reveal_parser.add_argument("--adjudications", type=Path)
    reveal_parser.add_argument("--output", type=Path, required=True)

    secondary_export_parser = subparsers.add_parser(
        "export-secondary", help="export labeled items plus a deterministic clean sample"
    )
    secondary_export_parser.add_argument("--primary-reviews", type=Path, required=True)
    secondary_export_parser.add_argument("--bundle", type=Path, required=True)
    secondary_export_parser.add_argument("--output", type=Path, required=True)

    secondary_validate_parser = subparsers.add_parser(
        "validate-secondary", help="validate an independent grader-blind second review"
    )
    secondary_validate_parser.add_argument("--reviews", type=Path, required=True)
    secondary_validate_parser.add_argument("--primary-reviews", type=Path, required=True)
    secondary_validate_parser.add_argument("--bundle", type=Path, required=True)
    secondary_validate_parser.add_argument("--secondary-bundle", type=Path, required=True)

    disagreement_parser = subparsers.add_parser(
        "disagreements", help="export unresolved primary-secondary disagreements"
    )
    disagreement_parser.add_argument("--primary-reviews", type=Path, required=True)
    disagreement_parser.add_argument("--secondary-reviews", type=Path, required=True)
    disagreement_parser.add_argument("--bundle", type=Path, required=True)
    disagreement_parser.add_argument("--secondary-bundle", type=Path, required=True)
    disagreement_parser.add_argument("--output", type=Path, required=True)

    adjudication_export_parser = subparsers.add_parser(
        "export-adjudication", help="export a grader-blind adjudication template"
    )
    adjudication_export_parser.add_argument("--disagreements", type=Path, required=True)
    adjudication_export_parser.add_argument("--output", type=Path, required=True)

    adjudication_validate_parser = subparsers.add_parser(
        "validate-adjudication", help="validate completed grader-blind adjudications"
    )
    adjudication_validate_parser.add_argument("--disagreements", type=Path, required=True)
    adjudication_validate_parser.add_argument("--adjudications", type=Path, required=True)

    args = parser.parse_args()
    try:
        if args.command == "export":
            result = export_blind_bundle(args.log_dir, args.output, args.key_file)
        elif args.command == "validate":
            reviews = validate_primary_reviews(args.reviews, args.identity_map)
            result = {"ok": True, "reviewed_items": len(reviews)}
        elif args.command == "reveal":
            validate_secondary_disagreement_artifact(
                args.reviews,
                args.secondary_reviews,
                args.bundle,
                args.secondary_bundle,
                args.disagreements,
            )
            validate_adjudications(args.adjudications, args.disagreements)
            result = reveal_grades(args.reviews, args.bundle, args.output)
        elif args.command == "export-secondary":
            result = export_secondary_bundle(args.primary_reviews, args.bundle, args.output)
        elif args.command == "validate-secondary":
            reviews = validate_secondary_reviews(
                args.reviews,
                args.primary_reviews,
                args.bundle,
                args.secondary_bundle,
            )
            result = {"ok": True, "reviewed_items": len(reviews)}
        elif args.command == "disagreements":
            result = export_secondary_disagreements(
                args.primary_reviews,
                args.secondary_reviews,
                args.bundle,
                args.secondary_bundle,
                args.output,
            )
        elif args.command == "export-adjudication":
            result = export_adjudication_template(args.disagreements, args.output)
        else:
            adjudications = validate_adjudications(
                args.adjudications,
                args.disagreements,
            )
            result = {"ok": True, "adjudicated_items": len(adjudications)}
    except (ReviewError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    sys.stdout.buffer.write(canonical_json_bytes(result))


if __name__ == "__main__":
    main()
