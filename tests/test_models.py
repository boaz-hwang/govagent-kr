import pytest
from pydantic import ValidationError

from govagent_kr.models import GoldReview, ReviewStatus


def _adjudicated_review(**updates: object) -> GoldReview:
    values: dict[str, object] = {
        "status": ReviewStatus.ADJUDICATED,
        "reviewer_1": "sme-a",
        "reviewer_2": "sme-b",
        "reviewer_1_verdict": "correct",
        "reviewer_2_verdict": "incorrect",
        "resolution": "[adjudicator_id=sme-c] evidence checked",
    }
    values.update(updates)
    return GoldReview.model_validate(values, strict=True)


def test_adjudicated_gold_accepts_only_review_verdict_enum() -> None:
    with pytest.raises(ValidationError):
        _adjudicated_review(reviewer_1_verdict="approve")


@pytest.mark.parametrize(
    ("reviewer_1", "reviewer_2"),
    [("same", "SAME"), (" same ", "same")],
)
def test_adjudicated_gold_requires_distinct_reviewer_ids(reviewer_1: str, reviewer_2: str) -> None:
    with pytest.raises(ValidationError, match="independent people"):
        _adjudicated_review(reviewer_1=reviewer_1, reviewer_2=reviewer_2)


def test_adjudicated_gold_rejects_blank_resolution() -> None:
    with pytest.raises(ValidationError, match="requires two reviewer verdicts"):
        _adjudicated_review(resolution="   ")
