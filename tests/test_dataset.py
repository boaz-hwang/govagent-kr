from pathlib import Path

import pytest

import govagent_kr.dataset as dataset_module
from govagent_kr.canonical import canonical_json_bytes
from govagent_kr.dataset import (
    DatasetBuildError,
    build_dataset,
    dataset_integrity_report,
    load_cases,
    load_gold,
)
from govagent_kr.fixtures import SNAPSHOTS, FixtureStore
from govagent_kr.models import (
    AnswerStatus,
    CaseRecord,
    Conclusion,
    GoldRecord,
    GoldReview,
    ReviewStatus,
)

H_2018_POST = "KR-LAW-010098:204108:20180724"
H_2020_PRE = "KR-LAW-010098:221951:20201001"
H_2020_POST = "KR-LAW-010098:222683:20201103"
H_2025_PRE = "KR-LAW-010098:269761:20250312"
H_2025_POST = "KR-LAW-010098:272723:20250708"


def test_every_frozen_official_snapshot_has_a_pinned_raw_hash() -> None:
    assert len(SNAPSHOTS) == 7
    assert all(snapshot.expected_raw_sha256 for snapshot in SNAPSHOTS)


@pytest.fixture(scope="module")
def generated_records(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[dict[str, CaseRecord], dict[str, GoldRecord]]:
    output_dir = tmp_path_factory.mktemp("dataset-contract")
    cases_path = output_dir / "cases.jsonl"
    gold_path = output_dir / "gold.jsonl"

    build_dataset(FixtureStore(), cases_path=cases_path, gold_path=gold_path)

    cases = {record.item_id: record for record in load_cases(cases_path)}
    gold = {record.item_id: record for record in load_gold(gold_path)}
    assert len(cases) == len(gold) == 20
    assert set(cases) == set(gold)
    return cases, gold


def _required_identities(record: GoldRecord) -> set[tuple[str, str]]:
    return {
        (citation.history_id, citation.provision_locator) for citation in record.required_citations
    }


def _write_gold(path: Path, records: list[GoldRecord]) -> None:
    path.write_bytes(
        b"".join(canonical_json_bytes(record.model_dump(mode="json")) for record in records)
    )


def _mark_first_gold_adjudicated(path: Path) -> None:
    records = load_gold(path)
    records[0] = records[0].model_copy(
        update={
            "review": GoldReview(
                status=ReviewStatus.ADJUDICATED,
                reviewer_1="sme-one",
                reviewer_2="sme-two",
                reviewer_1_verdict="correct",
                reviewer_2_verdict="correct",
                resolution="[adjudicator_id=lead-sme] accepted",
            )
        }
    )
    _write_gold(path, records)


def test_regeneration_refuses_reviewed_gold_without_explicit_new_cycle(
    tmp_path: Path,
) -> None:
    cases_path = tmp_path / "cases.jsonl"
    gold_path = tmp_path / "gold.jsonl"
    store = FixtureStore()
    build_dataset(store, cases_path=cases_path, gold_path=gold_path)
    _mark_first_gold_adjudicated(gold_path)
    before = (cases_path.read_bytes(), gold_path.read_bytes())

    with pytest.raises(DatasetBuildError, match="reviewed records"):
        build_dataset(store, cases_path=cases_path, gold_path=gold_path)

    assert (cases_path.read_bytes(), gold_path.read_bytes()) == before


def test_explicit_new_cycle_replaces_reviewed_gold_with_valid_pending_pair(
    tmp_path: Path,
) -> None:
    cases_path = tmp_path / "cases.jsonl"
    gold_path = tmp_path / "gold.jsonl"
    store = FixtureStore()
    build_dataset(store, cases_path=cases_path, gold_path=gold_path)
    _mark_first_gold_adjudicated(gold_path)

    result = build_dataset(
        store,
        cases_path=cases_path,
        gold_path=gold_path,
        force_new_cycle=True,
    )

    report = dataset_integrity_report(store, cases_path, gold_path)
    assert result["force_new_cycle"] is True
    assert result["pair_validated_before_replace"] is True
    assert report["ok"] is True
    assert report["pending_human_review"] == 20
    assert report["candidate_gold_blueprint_match"] is True


def test_staged_validation_failure_preserves_existing_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cases_path = tmp_path / "cases.jsonl"
    gold_path = tmp_path / "gold.jsonl"
    store = FixtureStore()
    build_dataset(store, cases_path=cases_path, gold_path=gold_path)
    before = (cases_path.read_bytes(), gold_path.read_bytes())
    monkeypatch.setattr(
        dataset_module,
        "dataset_integrity_report",
        lambda *_args, **_kwargs: {
            "ok": False,
            "case_count": 20,
            "pending_human_review": 20,
            "excluded_items": 0,
            "case_blueprint_match": True,
            "candidate_gold_blueprint_match": False,
            "errors": ["synthetic staged validation failure"],
        },
    )

    with pytest.raises(DatasetBuildError, match="synthetic staged validation failure"):
        build_dataset(store, cases_path=cases_path, gold_path=gold_path)

    assert (cases_path.read_bytes(), gold_path.read_bytes()) == before


def test_second_replace_failure_rolls_back_both_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cases_path = tmp_path / "cases.jsonl"
    gold_path = tmp_path / "gold.jsonl"
    store = FixtureStore()
    build_dataset(store, cases_path=cases_path, gold_path=gold_path)
    before = (cases_path.read_bytes(), gold_path.read_bytes())
    real_replace = dataset_module.os.replace
    replacements = 0

    def fail_second_replace(source: Path, target: Path) -> None:
        nonlocal replacements
        replacements += 1
        if replacements == 2:
            raise OSError("synthetic second replace failure")
        real_replace(source, target)

    monkeypatch.setattr(dataset_module.os, "replace", fail_second_replace)

    with pytest.raises(DatasetBuildError, match="prior pair was restored"):
        build_dataset(store, cases_path=cases_path, gold_path=gold_path)

    assert (cases_path.read_bytes(), gold_path.read_bytes()) == before


@pytest.mark.parametrize(
    ("target", "old", "new", "expected_error"),
    [
        (
            "cases",
            "행정안전부장관이 정하는 기준에 적합하다.",
            "행정안전부장관이 정하는 기준에 부적합하다.",
            "cases differ from the deterministic dataset blueprint",
        ),
        (
            "gold",
            "CANDIDATE GOLD",
            "CANDIDATE DRAFT",
            "pending candidate gold differs from the deterministic blueprint",
        ),
    ],
)
def test_integrity_report_rejects_stale_checked_data(
    tmp_path: Path,
    target: str,
    old: str,
    new: str,
    expected_error: str,
) -> None:
    cases_path = tmp_path / "cases.jsonl"
    gold_path = tmp_path / "gold.jsonl"
    store = FixtureStore()
    build_dataset(store, cases_path=cases_path, gold_path=gold_path)
    path = cases_path if target == "cases" else gold_path
    payload = path.read_text(encoding="utf-8")
    assert old in payload
    path.write_text(payload.replace(old, new, 1), encoding="utf-8")

    report = dataset_integrity_report(store, cases_path, gold_path)

    assert report["ok"] is False
    assert expected_error in report["errors"]


def test_answered_items_have_exact_substantive_conclusions(
    generated_records: tuple[dict[str, CaseRecord], dict[str, GoldRecord]],
) -> None:
    _, gold = generated_records
    expected = {
        "social-enterprise-direct-award--post_effective": Conclusion.AFFIRMATIVE,
        "social-enterprise-single-quote--post_effective": Conclusion.AFFIRMATIVE,
        "statutory-digital-service--post_effective": Conclusion.AFFIRMATIVE,
        "legacy-innovation-transition--pre_effective": Conclusion.AFFIRMATIVE,
        "legacy-innovation-transition--post_effective": Conclusion.AFFIRMATIVE,
        "legacy-innovation-transition--current_decoy": Conclusion.AFFIRMATIVE,
        "material-price-adjustment-transition--pre_effective": Conclusion.NEGATIVE,
        "material-price-adjustment-transition--post_effective": Conclusion.AFFIRMATIVE,
        "material-price-adjustment-transition--current_decoy": Conclusion.NEGATIVE,
    }
    answered = {
        item_id: record.expected_conclusion
        for item_id, record in gold.items()
        if record.expected_status == AnswerStatus.ANSWERED
    }

    assert answered == expected


def test_status_and_conclusion_invariants_hold_for_every_gold_item(
    generated_records: tuple[dict[str, CaseRecord], dict[str, GoldRecord]],
) -> None:
    _, gold = generated_records
    allowed_by_status = {
        AnswerStatus.ANSWERED: {Conclusion.AFFIRMATIVE, Conclusion.NEGATIVE},
        AnswerStatus.INSUFFICIENT_FACTS: {Conclusion.INDETERMINATE},
        AnswerStatus.NO_APPLICABLE_PROVISION: {Conclusion.NEGATIVE},
    }

    for item_id, record in gold.items():
        assert record.expected_conclusion in allowed_by_status[record.expected_status], item_id


def test_single_quote_cases_state_the_separate_ministerial_criteria_fact(
    generated_records: tuple[dict[str, CaseRecord], dict[str, GoldRecord]],
) -> None:
    cases, _ = generated_records
    family = [case for case in cases.values() if case.family_id == "social-enterprise-single-quote"]
    assert len(family) == 4
    for case in family:
        assert any(
            "제30조제1항제2호나목" in fact and "기준에 적합" in fact for fact in case.input.facts
        )


def test_material_insufficient_case_does_not_leak_transition_fact_and_uses_statutory_terms(
    generated_records: tuple[dict[str, CaseRecord], dict[str, GoldRecord]],
) -> None:
    cases, gold = generated_records
    item_id = "material-price-adjustment-transition--insufficient_facts"
    case = cases[item_id]
    record = gold[item_id]
    facts = case.input.facts
    task_text = "\n".join((case.input.question, *facts))

    assert "완료" not in task_text
    assert "해당 자재는 특정규격의 자재이다." in facts
    assert any(
        "해당 공사비를 구성하는 재료비ㆍ노무비ㆍ경비 합계액의 1000분의 5" in fact for fact in facts
    )
    assert "계약금액의 100분의 1" not in task_text
    assert record.required_missing_fact_keys == ["unfinished_at_effective_date"]


def test_digital_insufficient_case_does_not_assert_statutory_selection(
    generated_records: tuple[dict[str, CaseRecord], dict[str, GoldRecord]],
) -> None:
    cases, gold = generated_records
    item_id = "statutory-digital-service--insufficient_facts"
    case = cases[item_id]
    record = gold[item_id]
    facts_text = "\n".join(case.input.facts)

    assert case.input.facts == [
        "대상은 클라우드 기반 서비스이다.",
        "다른 수의계약 사유에는 해당하지 않는다.",
    ]
    assert "이 디지털서비스" not in case.input.question
    assert "선정" not in facts_text
    assert "심사위원회" not in facts_text
    assert "제8조의2" not in facts_text
    assert record.required_missing_fact_keys == ["statutory_digital_service_selection"]


def test_answered_items_require_exact_granular_temporal_citations(
    generated_records: tuple[dict[str, CaseRecord], dict[str, GoldRecord]],
) -> None:
    _, gold = generated_records
    expected = {
        "social-enterprise-direct-award--post_effective": {
            (H_2018_POST, "제25조제1항"),
            (H_2018_POST, "제25조제1항제5호마목#본문"),
            (H_2018_POST, "제25조제1항제5호마목#3)"),
        },
        "social-enterprise-single-quote--post_effective": {
            (H_2018_POST, "제30조제1항"),
            (H_2018_POST, "제30조제1항제2호"),
            (H_2018_POST, "제30조제1항제2호나목"),
        },
        "statutory-digital-service--post_effective": {
            (H_2020_POST, "제25조제1항"),
            (H_2020_POST, "제25조제1항제8호차목"),
        },
        "legacy-innovation-transition--pre_effective": {
            (H_2020_PRE, "제25조제1항"),
            (H_2020_PRE, "제25조제1항제6호라목#본문"),
            (H_2020_PRE, "제25조제1항제6호라목#8)"),
        },
        "legacy-innovation-transition--post_effective": {
            (H_2020_POST, "제25조제1항"),
            (H_2020_PRE, "제25조제1항제6호라목#본문"),
            (H_2020_PRE, "제25조제1항제6호라목#8)"),
            (H_2020_POST, "부칙(대통령령 제31136호)제2조"),
        },
        "legacy-innovation-transition--current_decoy": {
            (H_2020_POST, "제25조제1항"),
            (H_2020_PRE, "제25조제1항제6호라목#본문"),
            (H_2020_PRE, "제25조제1항제6호라목#8)"),
            (H_2020_POST, "부칙(대통령령 제31136호)제2조"),
        },
        "material-price-adjustment-transition--pre_effective": {
            (H_2025_PRE, "제73조제6항"),
        },
        "material-price-adjustment-transition--post_effective": {
            (H_2025_POST, "제73조제6항"),
            (H_2025_POST, "부칙(대통령령 제35644호)제4조"),
        },
        "material-price-adjustment-transition--current_decoy": {
            (H_2025_PRE, "제73조제6항"),
        },
    }

    actual = {
        item_id: _required_identities(record)
        for item_id, record in gold.items()
        if record.expected_status == AnswerStatus.ANSWERED
    }
    assert actual == expected

    for item_id in (
        "legacy-innovation-transition--post_effective",
        "legacy-innovation-transition--current_decoy",
    ):
        identities = _required_identities(gold[item_id])
        assert (H_2020_POST, "제25조제1항") in identities
        assert (H_2020_PRE, "제25조제1항") not in identities
