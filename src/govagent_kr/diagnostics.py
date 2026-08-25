from __future__ import annotations

from pathlib import Path
from typing import Any

from .canonical import canonical_json_bytes, content_sha256
from .constants import DEFAULT_CASES_PATH, DEFAULT_FIXTURE_ROOT, DEFAULT_GOLD_PATH
from .dataset import dataset_integrity_report, load_gold
from .fixtures import FixtureStore, fixture_integrity_report
from .scorer import (
    current_version_ablation_submission,
    gold_submission,
    grade_submission,
    replay_grade_bytes,
)


def run_diagnostics(
    fixture_root: Path = DEFAULT_FIXTURE_ROOT,
    cases_path: Path = DEFAULT_CASES_PATH,
    gold_path: Path = DEFAULT_GOLD_PATH,
) -> dict[str, Any]:
    store = FixtureStore(fixture_root)
    fixture_report = fixture_integrity_report(fixture_root)
    dataset_report = dataset_integrity_report(store, cases_path, gold_path)
    gold_records = load_gold(gold_path)
    oracle_items: list[dict[str, Any]] = []
    ablation_items: list[dict[str, Any]] = []
    replay_equal = True
    for gold in gold_records:
        oracle_raw = gold_submission(gold, store)
        oracle_grade = grade_submission(oracle_raw, gold, store)
        first = replay_grade_bytes(oracle_raw, gold, store)
        second = replay_grade_bytes(oracle_raw, gold, store)
        replay_equal = replay_equal and first == second
        oracle_items.append(
            {
                "item_id": gold.item_id,
                "core_success": oracle_grade.core_success,
                "failure_codes": oracle_grade.failure_codes,
                "grade_sha256": content_sha256(first),
            }
        )
        ablation_raw = current_version_ablation_submission(gold, store)
        ablation_grade = grade_submission(ablation_raw, gold, store)
        ablation_items.append(
            {
                "item_id": gold.item_id,
                "core_success": ablation_grade.core_success,
                "failure_codes": ablation_grade.failure_codes,
            }
        )
    oracle_successes = sum(item["core_success"] == 1 for item in oracle_items)
    ablation_failures = sum(item["core_success"] == 0 for item in ablation_items)
    report = {
        "fixture_integrity": fixture_report,
        "dataset_integrity": dataset_report,
        "fixture_oracle": {
            "successes": oracle_successes,
            "total": len(oracle_items),
            "items": oracle_items,
        },
        "current_version_ablation": {
            "failures": ablation_failures,
            "total": len(ablation_items),
            "minimum_expected_failures": 3,
            "items": ablation_items,
        },
        "replay_byte_identical": replay_equal,
    }
    report["engineering_ready"] = bool(
        fixture_report["ok"]
        and dataset_report["ok"]
        and oracle_successes == len(oracle_items) == 20
        and ablation_failures >= 3
        and replay_equal
    )
    report["analysis_ready"] = bool(
        report["engineering_ready"] and dataset_report["analysis_ready"]
    )
    report["report_sha256"] = content_sha256(canonical_json_bytes(report))
    return report


__all__ = ["run_diagnostics"]
