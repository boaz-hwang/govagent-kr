from __future__ import annotations

import json
import os
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .canonical import canonical_json_bytes, content_sha256, controlling_event_hash
from .constants import (
    DEFAULT_CASES_PATH,
    DEFAULT_GOLD_PATH,
    INSTRUMENT_ID,
    SCHEMA_VERSION,
    TIMEZONE,
)
from .fixtures import SNAPSHOTS, FixtureLookupError, FixtureStore
from .models import (
    AnswerStatus,
    CaseMetadata,
    CaseRecord,
    Conclusion,
    ControllingEvent,
    GoldCitation,
    GoldRecord,
    GoldReview,
    ReviewStatus,
    TaskInput,
    Variant,
)


class DatasetBuildError(RuntimeError):
    """Raised when candidate dataset regeneration cannot complete safely."""


@dataclass(frozen=True)
class VariantBlueprint:
    variant: Variant
    event_date: str
    facts: tuple[str, ...]
    expected_status: AnswerStatus
    citations: tuple[tuple[str, str], ...] = ()
    allowed_citations: tuple[tuple[str, str], ...] = ()
    required_missing_fact_keys: tuple[str, ...] = ()
    expected_conclusion: Conclusion | None = None
    note: str = ""


@dataclass(frozen=True)
class FamilyBlueprint:
    family_id: str
    boundary_id: str
    effective_at: str
    question: str
    event_kind: str
    variants: tuple[VariantBlueprint, ...]


def _history(lsi_seq: str, effective_date: str) -> str:
    return f"{INSTRUMENT_ID}:{lsi_seq}:{effective_date}"


H_2018_PRE = _history("201260", "20180118")
H_2018_POST = _history("204108", "20180724")
H_2020_PRE = _history("221951", "20201001")
H_2020_POST = _history("222683", "20201103")
H_2025_PRE = _history("269761", "20250312")
H_2025_POST = _history("272723", "20250708")
H_LATEST = _history("286149", "20260603")


COMMON_SOCIAL_FACTS = (
    "계약은 추정가격 3천만원의 용역계약이다.",
    "계약상대 후보는 「사회적기업 육성법」상 사회적기업이다.",
    "해당 기업은 소기업ㆍ소상공인, 여성기업 또는 장애인기업에는 해당하지 않는다.",
    "제25조의 다른 수의계약 사유에는 해당하지 않는다.",
    "행정안전부장관이 고시한 취약계층 고용비율을 충족한다.",
)

SINGLE_QUOTE_SOCIAL_FACTS = (
    *COMMON_SOCIAL_FACTS,
    "해당 기업은 제30조제1항제2호나목에 따라 행정안전부장관이 정하는 기준에 적합하다.",
)

DIGITAL_SERVICE_SELECTION_FACT = (
    "과학기술정보통신부장관은 「클라우드컴퓨팅 발전 및 이용자 보호에 관한 법률 시행령」 "
    "제8조의2제1항에 따라, 제8조의3제1항의 디지털서비스 심사위원회 심사를 거쳐 이 "
    "서비스를 국가기관등이 이용하기에 적합한 디지털서비스로 선정하였다."
)


FAMILIES: tuple[FamilyBlueprint, ...] = (
    FamilyBlueprint(
        family_id="social-enterprise-direct-award",
        boundary_id="social-enterprise-2018-07-24",
        effective_at="2018-07-24",
        question=(
            "이 사회적기업과 수의계약을 체결할 수 있는 시행령상 근거가 기준일 당시 "
            "존재하는지 판단하고, 존재한다면 해당 조문 원문을 정확히 인용하라."
        ),
        event_kind="contract_method_decision",
        variants=(
            VariantBlueprint(
                Variant.PRE_EFFECTIVE,
                "2018-07-23",
                COMMON_SOCIAL_FACTS,
                AnswerStatus.NO_APPLICABLE_PROVISION,
                note="사회적기업 명시 근거 시행 전",
            ),
            VariantBlueprint(
                Variant.POST_EFFECTIVE,
                "2018-07-24",
                COMMON_SOCIAL_FACTS,
                AnswerStatus.ANSWERED,
                (
                    (H_2018_POST, "제25조제1항"),
                    (H_2018_POST, "제25조제1항제5호마목#본문"),
                    (H_2018_POST, "제25조제1항제5호마목#3)"),
                ),
                allowed_citations=(
                    (H_2018_POST, "제25조제1항제5호"),
                    (H_2018_POST, "부칙(대통령령 제29059호)"),
                ),
                note="공포일 시행 후 사회적기업 명시 근거",
            ),
            VariantBlueprint(
                Variant.CURRENT_DECOY,
                "2018-07-23",
                (
                    *COMMON_SOCIAL_FACTS,
                    "담당자의 최신 검색 메모에는 사회적기업도 가능하다고 적혀 있다.",
                ),
                AnswerStatus.NO_APPLICABLE_PROVISION,
                note="최신 메모보다 기준일을 우선해야 함",
            ),
            VariantBlueprint(
                Variant.INSUFFICIENT_FACTS,
                "2018-07-24",
                tuple(fact for fact in COMMON_SOCIAL_FACTS if "취약계층 고용비율" not in fact),
                AnswerStatus.INSUFFICIENT_FACTS,
                required_missing_fact_keys=("vulnerable_employment_ratio_compliance",),
                note="고시상 취약계층 고용비율 충족 여부 누락",
            ),
        ),
    ),
    FamilyBlueprint(
        family_id="social-enterprise-single-quote",
        boundary_id="social-enterprise-2018-07-24",
        effective_at="2018-07-24",
        question=(
            "이 계약에서 1인에게서 받은 견적서만으로 수의계약 상대자를 선정할 수 있는 "
            "시행령상 예외가 기준일 당시 존재하는지 판단하고, 존재한다면 조문 원문을 "
            "정확히 인용하라."
        ),
        event_kind="quotation_collection",
        variants=(
            VariantBlueprint(
                Variant.PRE_EFFECTIVE,
                "2018-07-23",
                SINGLE_QUOTE_SOCIAL_FACTS,
                AnswerStatus.NO_APPLICABLE_PROVISION,
                note="사회적기업 5천만원 이하 1인 견적 예외 시행 전",
            ),
            VariantBlueprint(
                Variant.POST_EFFECTIVE,
                "2018-07-24",
                SINGLE_QUOTE_SOCIAL_FACTS,
                AnswerStatus.ANSWERED,
                (
                    (H_2018_POST, "제30조제1항"),
                    (H_2018_POST, "제30조제1항제2호"),
                    (H_2018_POST, "제30조제1항제2호나목"),
                ),
                allowed_citations=(
                    (H_2018_POST, "제25조제1항"),
                    (H_2018_POST, "제25조제1항제5호"),
                    (H_2018_POST, "제25조제1항제5호마목#본문"),
                    (H_2018_POST, "제25조제1항제5호마목#3)"),
                    (H_2018_POST, "부칙(대통령령 제29059호)"),
                ),
                note="금액 상한과 사회적기업 요건을 함께 인용",
            ),
            VariantBlueprint(
                Variant.CURRENT_DECOY,
                "2018-07-23",
                (
                    *SINGLE_QUOTE_SOCIAL_FACTS,
                    "현행 업무편람은 사회적기업의 1인 견적을 허용한다고 설명한다.",
                ),
                AnswerStatus.NO_APPLICABLE_PROVISION,
                note="현행 편람 decoy",
            ),
            VariantBlueprint(
                Variant.INSUFFICIENT_FACTS,
                "2018-07-24",
                tuple(
                    fact for fact in SINGLE_QUOTE_SOCIAL_FACTS if "취약계층 고용비율" not in fact
                ),
                AnswerStatus.INSUFFICIENT_FACTS,
                required_missing_fact_keys=("vulnerable_employment_ratio_compliance",),
                note="1인 견적 예외의 고용비율 요건 누락",
            ),
        ),
    ),
    FamilyBlueprint(
        family_id="statutory-digital-service",
        boundary_id="digital-service-2020-11-03",
        effective_at="2020-11-03",
        question=(
            "이 클라우드 기반 서비스를 수의계약할 수 있는 시행령상 근거가 기준일 당시 "
            "존재하는지 "
            "판단하고, 존재한다면 해당 조문 원문을 정확히 인용하라."
        ),
        event_kind="contract_method_decision",
        variants=(
            VariantBlueprint(
                Variant.PRE_EFFECTIVE,
                "2020-11-02",
                (
                    "대상은 클라우드 기반 서비스이다.",
                    "다른 수의계약 사유에는 해당하지 않는다.",
                    DIGITAL_SERVICE_SELECTION_FACT,
                ),
                AnswerStatus.NO_APPLICABLE_PROVISION,
                note="디지털서비스 수의계약 조항 시행 전",
            ),
            VariantBlueprint(
                Variant.POST_EFFECTIVE,
                "2020-11-03",
                (
                    "대상은 클라우드 기반 서비스이다.",
                    "다른 수의계약 사유에는 해당하지 않는다.",
                    DIGITAL_SERVICE_SELECTION_FACT,
                ),
                AnswerStatus.ANSWERED,
                (
                    (H_2020_POST, "제25조제1항"),
                    (H_2020_POST, "제25조제1항제8호차목"),
                ),
                allowed_citations=(
                    (H_2020_POST, "제25조제1항제8호"),
                    (H_2020_POST, "부칙(대통령령 제31136호)제1조"),
                ),
                note="디지털서비스 조항 신설 후",
            ),
            VariantBlueprint(
                Variant.CURRENT_DECOY,
                "2020-11-02",
                (
                    "대상은 클라우드 기반 서비스이다.",
                    "다른 수의계약 사유에는 해당하지 않는다.",
                    DIGITAL_SERVICE_SELECTION_FACT,
                    "현재 운영 중인 조달 화면에는 디지털서비스 수의계약 메뉴가 보인다.",
                ),
                AnswerStatus.NO_APPLICABLE_PROVISION,
                note="현재 조달 화면 decoy",
            ),
            VariantBlueprint(
                Variant.INSUFFICIENT_FACTS,
                "2020-11-03",
                (
                    "대상은 클라우드 기반 서비스이다.",
                    "다른 수의계약 사유에는 해당하지 않는다.",
                ),
                AnswerStatus.INSUFFICIENT_FACTS,
                required_missing_fact_keys=("statutory_digital_service_selection",),
                note="법정 디지털서비스 선정 여부 누락",
            ),
        ),
    ),
    FamilyBlueprint(
        family_id="legacy-innovation-transition",
        boundary_id="digital-service-2020-11-03",
        effective_at="2020-11-03",
        question=(
            "이 혁신제품과 수의계약을 할 수 있는지 기준일과 견적서 제출일을 반영해 "
            "판단하라. 적용되는 본문을 인용하고, 적용되는 부칙이 있으면 함께 정확히 "
            "인용하라."
        ),
        event_kind="contract_method_decision",
        variants=(
            VariantBlueprint(
                Variant.PRE_EFFECTIVE,
                "2020-11-02",
                (
                    "제품은 2020년 11월 2일 당시 제25조제1항제6호라목8)의 요건을 충족한다.",
                    "수의계약 견적서를 2020년 11월 2일 제출받았다.",
                    "다른 수의계약 사유에는 해당하지 않는다.",
                ),
                AnswerStatus.ANSWERED,
                (
                    (H_2020_PRE, "제25조제1항"),
                    (H_2020_PRE, "제25조제1항제6호라목#본문"),
                    (H_2020_PRE, "제25조제1항제6호라목#8)"),
                ),
                allowed_citations=((H_2020_PRE, "제25조제1항제6호"),),
                note="개정 시행 전 종전 본문 직접 적용",
            ),
            VariantBlueprint(
                Variant.POST_EFFECTIVE,
                "2020-11-04",
                (
                    "제품은 2020년 11월 2일 당시 제25조제1항제6호라목8)의 요건을 충족한다.",
                    "수의계약 견적서를 2020년 11월 2일 제출받았다.",
                    "다른 수의계약 사유에는 해당하지 않는다.",
                ),
                AnswerStatus.ANSWERED,
                (
                    (H_2020_POST, "제25조제1항"),
                    (H_2020_PRE, "제25조제1항제6호라목#본문"),
                    (H_2020_PRE, "제25조제1항제6호라목#8)"),
                    (H_2020_POST, "부칙(대통령령 제31136호)제2조"),
                ),
                allowed_citations=(
                    (H_2020_PRE, "제25조제1항"),
                    (H_2020_PRE, "제25조제1항제6호"),
                    (H_2020_POST, "제25조제1항제6호"),
                    (H_2020_POST, "제25조제1항제6호라목#본문"),
                    (H_2020_POST, "제25조제1항제6호라목#8)"),
                    (H_2020_POST, "부칙(대통령령 제31136호)제1조"),
                ),
                note="종전 본문과 신법 부칙 경과조치를 함께 적용",
            ),
            VariantBlueprint(
                Variant.CURRENT_DECOY,
                "2020-11-04",
                (
                    "제품은 2020년 11월 2일 당시 제25조제1항제6호라목8)의 요건을 충족한다.",
                    "수의계약 견적서를 2020년 11월 2일 제출받았다.",
                    "현행 본문 검색 결과에는 종전 8) 항목이 삭제된 것으로 표시된다.",
                    "다른 수의계약 사유에는 해당하지 않는다.",
                ),
                AnswerStatus.ANSWERED,
                (
                    (H_2020_POST, "제25조제1항"),
                    (H_2020_PRE, "제25조제1항제6호라목#본문"),
                    (H_2020_PRE, "제25조제1항제6호라목#8)"),
                    (H_2020_POST, "부칙(대통령령 제31136호)제2조"),
                ),
                allowed_citations=(
                    (H_2020_PRE, "제25조제1항"),
                    (H_2020_PRE, "제25조제1항제6호"),
                    (H_2020_POST, "제25조제1항제6호"),
                    (H_2020_POST, "제25조제1항제6호라목#본문"),
                    (H_2020_POST, "제25조제1항제6호라목#8)"),
                    (H_2020_POST, "부칙(대통령령 제31136호)제1조"),
                ),
                note="현행 본문 삭제 표시보다 경과조치를 우선",
            ),
            VariantBlueprint(
                Variant.INSUFFICIENT_FACTS,
                "2020-11-04",
                (
                    "제품은 2020년 11월 2일 당시 제25조제1항제6호라목8)의 요건을 충족한다.",
                    "다른 수의계약 사유에는 해당하지 않는다.",
                ),
                AnswerStatus.INSUFFICIENT_FACTS,
                required_missing_fact_keys=("pre_effective_estimate_received",),
                note="시행 전 견적서 제출 여부 누락",
            ),
        ),
    ),
    FamilyBlueprint(
        family_id="material-price-adjustment-transition",
        boundary_id="price-adjustment-2025-07-08",
        effective_at="2025-07-08",
        question=(
            "이 공사계약에 대해 기준일 당시 제73조제6항에 따른 자재 가격 변동 계약금액 "
            "조정 요건이 충족되는지 판단하라. 적용되는 본문을 인용하고, 개정규정의 적용 "
            "여부에 부칙이 관여하면 해당 부칙도 함께 정확히 인용하라."
        ),
        event_kind="price_adjustment_request",
        variants=(
            VariantBlueprint(
                Variant.PRE_EFFECTIVE,
                "2025-07-07",
                (
                    "공사계약은 2025년 3월 20일 체결되었다.",
                    "해당 자재는 특정규격의 자재이다.",
                    "입찰일을 기준일로 하여 산정한 해당 자재의 가격증감률은 100분의 12 상승이다.",
                    "해당 자재의 금액은 해당 공사비를 구성하는 재료비ㆍ노무비ㆍ경비 합계액의 "
                    "1000분의 5를 초과한다.",
                    "2025년 7월 7일 기준일 현재 이 공사계약의 이행은 완료되지 않았다.",
                ),
                AnswerStatus.ANSWERED,
                ((H_2025_PRE, "제73조제6항"),),
                expected_conclusion=Conclusion.NEGATIVE,
                note="종전 15퍼센트 기준",
            ),
            VariantBlueprint(
                Variant.POST_EFFECTIVE,
                "2025-07-08",
                (
                    "공사계약은 2025년 3월 20일 체결되었다.",
                    "해당 자재는 특정규격의 자재이다.",
                    "입찰일을 기준일로 하여 산정한 해당 자재의 가격증감률은 100분의 12 상승이다.",
                    "해당 자재의 금액은 해당 공사비를 구성하는 재료비ㆍ노무비ㆍ경비 합계액의 "
                    "1000분의 5를 초과한다.",
                    "2025년 7월 8일 이 영 시행 당시 이 공사계약의 이행은 완료되지 않았다.",
                ),
                AnswerStatus.ANSWERED,
                (
                    (H_2025_POST, "제73조제6항"),
                    (H_2025_POST, "부칙(대통령령 제35644호)제4조"),
                ),
                allowed_citations=((H_2025_POST, "부칙(대통령령 제35644호)제1조"),),
                note="10퍼센트 기준과 미완료 계약 적용례",
            ),
            VariantBlueprint(
                Variant.CURRENT_DECOY,
                "2025-07-07",
                (
                    "공사계약은 2025년 3월 20일 체결되었다.",
                    "해당 자재는 특정규격의 자재이다.",
                    "입찰일을 기준일로 하여 산정한 해당 자재의 가격증감률은 100분의 12 상승이다.",
                    "해당 자재의 금액은 해당 공사비를 구성하는 재료비ㆍ노무비ㆍ경비 합계액의 "
                    "1000분의 5를 초과한다.",
                    "2025년 7월 7일 기준일 현재 이 공사계약의 이행은 완료되지 않았다.",
                    "담당자가 본 현행 화면에는 특정 자재 기준이 10퍼센트라고 적혀 있다.",
                ),
                AnswerStatus.ANSWERED,
                ((H_2025_PRE, "제73조제6항"),),
                expected_conclusion=Conclusion.NEGATIVE,
                note="시행 직전에는 현행 10퍼센트 문구를 사용하면 안 됨",
            ),
            VariantBlueprint(
                Variant.INSUFFICIENT_FACTS,
                "2025-07-08",
                (
                    "공사계약은 2025년 3월 20일 체결되었다.",
                    "해당 자재는 특정규격의 자재이다.",
                    "입찰일을 기준일로 하여 산정한 해당 자재의 가격증감률은 100분의 12 상승이다.",
                    "해당 자재의 금액은 해당 공사비를 구성하는 재료비ㆍ노무비ㆍ경비 합계액의 "
                    "1000분의 5를 초과한다.",
                ),
                AnswerStatus.INSUFFICIENT_FACTS,
                required_missing_fact_keys=("unfinished_at_effective_date",),
                note="부칙 적용에 필요한 시행일 당시 미완료 여부 누락",
            ),
        ),
    ),
)


def _gold_citations(
    store: FixtureStore, references: tuple[tuple[str, str], ...]
) -> list[GoldCitation]:
    result: list[GoldCitation] = []
    for history_id, locator in references:
        provision = store.get_provision(history_id, locator)
        result.append(
            GoldCitation(
                instrument_id=INSTRUMENT_ID,
                history_id=history_id,
                provision_locator=locator,
                exact_passage_hash=provision["exact_passage_hash"],
            )
        )
    return result


def _forbidden_histories(
    references: tuple[tuple[str, str], ...], family: FamilyBlueprint
) -> list[str]:
    selected = {history for history, _ in references}
    boundary_histories = {
        spec.history_id for spec in SNAPSHOTS if spec.boundary_id == family.boundary_id
    }
    candidates = boundary_histories | {H_LATEST}
    return sorted(candidates - selected)


def _expected_conclusion(variant: VariantBlueprint) -> Conclusion:
    if variant.expected_conclusion is not None:
        return variant.expected_conclusion
    if variant.expected_status == AnswerStatus.ANSWERED:
        return Conclusion.AFFIRMATIVE
    if variant.expected_status == AnswerStatus.NO_APPLICABLE_PROVISION:
        return Conclusion.NEGATIVE
    return Conclusion.INDETERMINATE


def _build_records(store: FixtureStore) -> tuple[list[CaseRecord], list[GoldRecord]]:
    cases: list[CaseRecord] = []
    gold_records: list[GoldRecord] = []
    for family in FAMILIES:
        for variant in family.variants:
            event = ControllingEvent(
                kind=family.event_kind,
                date=variant.event_date,
                timezone=TIMEZONE,
            )
            item_id = f"{family.family_id}--{variant.variant.value}"
            case = CaseRecord(
                schema_version=SCHEMA_VERSION,
                item_id=item_id,
                family_id=family.family_id,
                metadata=CaseMetadata(
                    boundary_id=family.boundary_id,
                    variant=variant.variant,
                    effective_at=family.effective_at,
                    candidate_gold=True,
                ),
                input=TaskInput(
                    task="kr-law-temporal-citation",
                    question=family.question,
                    facts=list(variant.facts),
                    instrument_id=INSTRUMENT_ID,
                    controlling_event=event,
                ),
            )
            citations = _gold_citations(store, variant.citations)
            allowed_citations = _gold_citations(store, variant.allowed_citations)
            all_references = variant.citations + variant.allowed_citations
            gold = GoldRecord(
                schema_version=SCHEMA_VERSION,
                item_id=item_id,
                expected_status=variant.expected_status,
                expected_conclusion=_expected_conclusion(variant),
                required_citations=citations,
                allowed_citations=allowed_citations,
                forbidden_history_ids=_forbidden_histories(all_references, family),
                required_missing_fact_keys=list(variant.required_missing_fact_keys),
                allowed_missing_fact_keys=[],
                controlling_event_hash=controlling_event_hash(event.model_dump(mode="json")),
                adjudication_note=(
                    "CANDIDATE GOLD — human legal SME review pending. " + variant.note
                ),
                review=GoldReview(status=ReviewStatus.PENDING),
            )
            cases.append(case)
            gold_records.append(gold)
    return cases, gold_records


def build_dataset(
    store: FixtureStore,
    cases_path: Path = DEFAULT_CASES_PATH,
    gold_path: Path = DEFAULT_GOLD_PATH,
    *,
    force_new_cycle: bool = False,
) -> dict[str, Any]:
    cases_path = cases_path.resolve()
    gold_path = gold_path.resolve()
    if cases_path == gold_path:
        raise DatasetBuildError("cases and gold outputs must be different files")
    if gold_path.exists():
        try:
            existing_gold = load_gold(gold_path)
        except (OSError, ValueError) as exc:
            raise DatasetBuildError(
                "existing gold is unreadable; refusing to replace it without manual recovery"
            ) from exc
        finalized = [
            record.item_id
            for record in existing_gold
            if record.review.status != ReviewStatus.PENDING
        ]
        if finalized and not force_new_cycle:
            raise DatasetBuildError(
                "existing gold contains reviewed records; pass force_new_cycle=True only when "
                "starting an explicit new legal-review cycle"
            )

    cases, gold_records = _build_records(store)
    cases_payload = b"".join(
        canonical_json_bytes(record.model_dump(mode="json")) for record in cases
    )
    gold_payload = b"".join(
        canonical_json_bytes(record.model_dump(mode="json")) for record in gold_records
    )
    cases_path.parent.mkdir(parents=True, exist_ok=True)
    gold_path.parent.mkdir(parents=True, exist_ok=True)

    staged_cases = _write_staged_payload(cases_path, cases_payload)
    staged_gold: Path | None = None
    try:
        staged_gold = _write_staged_payload(gold_path, gold_payload)
        staged_report = dataset_integrity_report(store, staged_cases, staged_gold)
        if not (
            staged_report["ok"]
            and staged_report["case_count"] == len(cases)
            and staged_report["pending_human_review"] == len(gold_records)
            and staged_report["excluded_items"] == 0
            and staged_report["case_blueprint_match"] is True
            and staged_report["candidate_gold_blueprint_match"] is True
        ):
            raise DatasetBuildError(
                "staged dataset pair failed deterministic validation: "
                + "; ".join(staged_report.get("errors", []))
            )
        _replace_dataset_pair(staged_cases, cases_path, staged_gold, gold_path)
    finally:
        staged_cases.unlink(missing_ok=True)
        if staged_gold is not None:
            staged_gold.unlink(missing_ok=True)
    return {
        "case_count": len(cases),
        "gold_count": len(gold_records),
        "case_sha256": content_sha256(cases_payload),
        "gold_sha256": content_sha256(gold_payload),
        "review_status": "pending",
        "force_new_cycle": force_new_cycle,
        "pair_validated_before_replace": True,
    }


def _write_staged_payload(target: Path, payload: bytes) -> Path:
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    staged_path = Path(raw_path)
    try:
        handle = os.fdopen(descriptor, "wb")
    except BaseException:
        os.close(descriptor)
        staged_path.unlink(missing_ok=True)
        raise
    try:
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        staged_path.unlink(missing_ok=True)
        raise
    return staged_path


def _restore_payload(target: Path, previous: bytes | None) -> None:
    if previous is None:
        target.unlink(missing_ok=True)
        return
    staged = _write_staged_payload(target, previous)
    try:
        os.replace(staged, target)
    finally:
        staged.unlink(missing_ok=True)


def _replace_dataset_pair(
    staged_cases: Path,
    cases_path: Path,
    staged_gold: Path,
    gold_path: Path,
) -> None:
    previous_cases = cases_path.read_bytes() if cases_path.exists() else None
    previous_gold = gold_path.read_bytes() if gold_path.exists() else None
    try:
        os.replace(staged_cases, cases_path)
        os.replace(staged_gold, gold_path)
    except OSError as exc:
        rollback_errors: list[str] = []
        for target, previous in (
            (cases_path, previous_cases),
            (gold_path, previous_gold),
        ):
            try:
                _restore_payload(target, previous)
            except OSError as rollback_exc:
                rollback_errors.append(f"{target}: {rollback_exc}")
        if rollback_errors:
            raise DatasetBuildError(
                "dataset replacement failed and rollback was incomplete: "
                + "; ".join(rollback_errors)
            ) from exc
        raise DatasetBuildError("dataset replacement failed; prior pair was restored") from exc


def _load_jsonl(path: Path, model_type: type[CaseRecord] | type[GoldRecord]) -> list[Any]:
    result: list[Any] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                result.append(model_type.model_validate_json(line, strict=True))
            except (ValidationError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid {path}:{line_number}: {exc}") from exc
    return result


def load_cases(path: Path = DEFAULT_CASES_PATH) -> list[CaseRecord]:
    return _load_jsonl(path, CaseRecord)


def load_gold(path: Path = DEFAULT_GOLD_PATH) -> list[GoldRecord]:
    return _load_jsonl(path, GoldRecord)


def dataset_integrity_report(
    store: FixtureStore,
    cases_path: Path = DEFAULT_CASES_PATH,
    gold_path: Path = DEFAULT_GOLD_PATH,
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        cases = load_cases(cases_path)
        gold_records = load_gold(gold_path)
    except (OSError, ValueError) as exc:
        return {"ok": False, "analysis_ready": False, "errors": [str(exc)]}
    try:
        expected_cases, expected_candidate_gold = _build_records(store)
    except FixtureLookupError as exc:
        return {
            "ok": False,
            "analysis_ready": False,
            "errors": [f"cannot reconstruct deterministic dataset blueprint: {exc}"],
        }
    case_blueprint_match = cases == expected_cases
    if not case_blueprint_match:
        errors.append("cases differ from the deterministic dataset blueprint")
    candidate_gold_blueprint_match: bool | None = None
    if gold_records and all(
        record.review.status == ReviewStatus.PENDING for record in gold_records
    ):
        candidate_gold_blueprint_match = gold_records == expected_candidate_gold
        if not candidate_gold_blueprint_match:
            errors.append("pending candidate gold differs from the deterministic blueprint")
    case_by_id = {case.item_id: case for case in cases}
    gold_by_id = {gold.item_id: gold for gold in gold_records}
    if len(case_by_id) != len(cases):
        errors.append("case item_id values are not unique")
    if len(gold_by_id) != len(gold_records):
        errors.append("gold item_id values are not unique")
    if set(case_by_id) != set(gold_by_id):
        errors.append("cases and gold do not have a one-to-one item_id mapping")
    if len(cases) != 20:
        errors.append(f"expected 20 cases, found {len(cases)}")
    family_variants: dict[str, set[Variant]] = defaultdict(set)
    boundary_families: dict[str, set[str]] = defaultdict(set)
    instruments: set[str] = set()
    for case in cases:
        family_variants[case.family_id].add(case.metadata.variant)
        boundary_families[case.metadata.boundary_id].add(case.family_id)
        instruments.add(case.input.instrument_id)
        event_date = date.fromisoformat(case.input.controlling_event.date)
        effective_at = date.fromisoformat(case.metadata.effective_at)
        if case.metadata.variant == Variant.PRE_EFFECTIVE:
            if not event_date < effective_at:
                errors.append(f"{case.item_id}: pre date is not before effective_at")
        elif (
            case.metadata.variant in {Variant.POST_EFFECTIVE, Variant.INSUFFICIENT_FACTS}
            and event_date < effective_at
        ):
            errors.append(f"{case.item_id}: post/insufficient date is before effective_at")
        gold = gold_by_id.get(case.item_id)
        if gold and gold.controlling_event_hash != controlling_event_hash(
            case.input.controlling_event.model_dump(mode="json")
        ):
            errors.append(f"{case.item_id}: controlling_event hash mismatch")
    required_variants = set(Variant)
    if len(family_variants) != 5:
        errors.append(f"expected 5 families, found {len(family_variants)}")
    for family_id, variants in family_variants.items():
        if variants != required_variants:
            errors.append(f"{family_id}: variants are {sorted(variants)}, expected all four")
    family_counts = sorted(len(value) for value in boundary_families.values())
    if family_counts != [1, 2, 2]:
        errors.append(f"boundary family counts are {family_counts}, expected [1, 2, 2]")
    if instruments != {INSTRUMENT_ID}:
        errors.append(f"expected one instrument {INSTRUMENT_ID}, found {sorted(instruments)}")
    for gold in gold_records:
        required = [citation.tuple4() for citation in gold.required_citations]
        allowed = [citation.tuple4() for citation in gold.allowed_citations]
        if len(required) != len(set(required)) or len(allowed) != len(set(allowed)):
            errors.append(f"{gold.item_id}: duplicate gold citation")
        if set(required) & set(allowed):
            errors.append(f"{gold.item_id}: required and allowed citations overlap")
        if any(
            citation.history_id in gold.forbidden_history_ids
            for citation in gold.required_citations + gold.allowed_citations
        ):
            errors.append(f"{gold.item_id}: accepted citation history is forbidden")
        if len(gold.forbidden_history_ids) != len(set(gold.forbidden_history_ids)):
            errors.append(f"{gold.item_id}: duplicate forbidden history ID")
        required_missing = gold.required_missing_fact_keys
        allowed_missing = gold.allowed_missing_fact_keys
        if len(required_missing) != len(set(required_missing)) or len(allowed_missing) != len(
            set(allowed_missing)
        ):
            errors.append(f"{gold.item_id}: duplicate missing-fact key")
        if set(required_missing) & set(allowed_missing):
            errors.append(f"{gold.item_id}: required and allowed missing-fact keys overlap")
        if gold.expected_status == AnswerStatus.ANSWERED and not required:
            errors.append(f"{gold.item_id}: answered gold has no required citations")
        if gold.expected_status != AnswerStatus.ANSWERED and (required or allowed):
            errors.append(f"{gold.item_id}: abstention gold contains citations")
        if gold.expected_status == AnswerStatus.INSUFFICIENT_FACTS:
            if not gold.required_missing_fact_keys:
                errors.append(f"{gold.item_id}: insufficient gold has no required fact key")
            if gold.expected_conclusion != Conclusion.INDETERMINATE:
                errors.append(f"{gold.item_id}: insufficient gold conclusion is not indeterminate")
        elif gold.required_missing_fact_keys or gold.allowed_missing_fact_keys:
            errors.append(f"{gold.item_id}: non-insufficient gold contains missing fact keys")
        if (
            gold.expected_status == AnswerStatus.ANSWERED
            and gold.expected_conclusion == Conclusion.INDETERMINATE
        ):
            errors.append(f"{gold.item_id}: answered gold conclusion is indeterminate")
        if (
            gold.expected_status == AnswerStatus.NO_APPLICABLE_PROVISION
            and gold.expected_conclusion != Conclusion.NEGATIVE
        ):
            errors.append(f"{gold.item_id}: no-applicable gold conclusion is not negative")
        for citation in gold.required_citations + gold.allowed_citations:
            try:
                provision = store.get_provision(citation.history_id, citation.provision_locator)
            except FixtureLookupError as exc:
                errors.append(f"{gold.item_id}: unresolved gold citation: {exc}")
                continue
            if provision["instrument_id"] != citation.instrument_id:
                errors.append(f"{gold.item_id}: fixture instrument mismatch")
            if provision["exact_passage_hash"] != citation.exact_passage_hash:
                errors.append(f"{gold.item_id}: gold passage hash mismatch")
    pending = sum(gold.review.status == ReviewStatus.PENDING for gold in gold_records)
    excluded = sum(gold.review.status == ReviewStatus.EXCLUDED for gold in gold_records)
    analysis_ready = not errors and pending == 0 and excluded == 0
    return {
        "ok": not errors,
        "analysis_ready": analysis_ready,
        "case_count": len(cases),
        "family_count": len(family_variants),
        "boundary_count": len(boundary_families),
        "pending_human_review": pending,
        "excluded_items": excluded,
        "case_blueprint_match": case_blueprint_match,
        "candidate_gold_blueprint_match": candidate_gold_blueprint_match,
        "errors": errors,
    }


__all__ = [
    "FAMILIES",
    "H_LATEST",
    "DatasetBuildError",
    "build_dataset",
    "dataset_integrity_report",
    "load_cases",
    "load_gold",
]
