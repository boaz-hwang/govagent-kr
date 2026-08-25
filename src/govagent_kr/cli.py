from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from .canonical import canonical_json_bytes
from .constants import DEFAULT_CASES_PATH, DEFAULT_FIXTURE_ROOT, DEFAULT_GOLD_PATH
from .dataset import build_dataset, dataset_integrity_report, load_gold
from .diagnostics import run_diagnostics
from .fixtures import FixtureBuildError, FixtureStore, build_fixtures, fixture_integrity_report
from .models import EpisodeOutcome
from .scorer import grade_submission


def _print(value: Any) -> None:
    sys.stdout.buffer.write(canonical_json_bytes(value))


def build_fixtures_main() -> None:
    parser = argparse.ArgumentParser(description="Freeze official Korean law fixtures")
    parser.add_argument("--output", type=Path, default=DEFAULT_FIXTURE_ROOT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--oc",
        default=os.environ.get("LAW_GO_KR_OC", "test"),
        help="official API key; defaults to LAW_GO_KR_OC or the public test key",
    )
    args = parser.parse_args()
    try:
        _print(build_fixtures(args.output, oc=args.oc, force=args.force))
    except FixtureBuildError as exc:
        parser.error(str(exc))


def build_dataset_main() -> None:
    parser = argparse.ArgumentParser(description="Build the 5-family/20-item candidate dataset")
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURE_ROOT)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD_PATH)
    args = parser.parse_args()
    store = FixtureStore(args.fixtures)
    _print(build_dataset(store, args.cases, args.gold))


def validate_main() -> None:
    parser = argparse.ArgumentParser(description="Validate fixtures, data, and review readiness")
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURE_ROOT)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD_PATH)
    parser.add_argument("--require-human-review", action="store_true")
    args = parser.parse_args()
    store = FixtureStore(args.fixtures)
    report = {
        "fixtures": fixture_integrity_report(args.fixtures),
        "dataset": dataset_integrity_report(store, args.cases, args.gold),
    }
    report["ok"] = report["fixtures"]["ok"] and report["dataset"]["ok"]
    if args.require_human_review:
        report["ok"] = report["ok"] and report["dataset"]["analysis_ready"]
    _print(report)
    raise SystemExit(0 if report["ok"] else 1)


def diagnostics_main() -> None:
    parser = argparse.ArgumentParser(description="Run fixture oracle and current-version ablation")
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURE_ROOT)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD_PATH)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_diagnostics(args.fixtures, args.cases, args.gold)
    payload = canonical_json_bytes(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(payload)
    sys.stdout.buffer.write(payload)
    raise SystemExit(0 if report["engineering_ready"] else 1)


def grade_main() -> None:
    parser = argparse.ArgumentParser(description="Grade one raw submission from a file or stdin")
    parser.add_argument("item_id")
    parser.add_argument("submission", nargs="?", type=Path)
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURE_ROOT)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD_PATH)
    parser.add_argument(
        "--episode-outcome",
        choices=[value.value for value in EpisodeOutcome],
        default=EpisodeOutcome.COMPLETED.value,
    )
    args = parser.parse_args()
    gold_by_id = {record.item_id: record for record in load_gold(args.gold)}
    if args.item_id not in gold_by_id:
        parser.error(f"unknown item_id: {args.item_id}")
    raw = args.submission.read_bytes() if args.submission else sys.stdin.buffer.read()
    result = grade_submission(
        raw,
        gold_by_id[args.item_id],
        FixtureStore(args.fixtures),
        episode_outcome=EpisodeOutcome(args.episode_outcome),
    )
    _print(result.model_dump(mode="json"))
    raise SystemExit(0 if result.core_success == 1 else 1)


def review_main() -> None:
    from .review import main

    main()


if __name__ == "__main__":
    raise SystemExit("Use one of the installed govagent-* entry points")
