from uuid import uuid4

import pytest

from backend.models import (
    GapHumanReview,
    GapReviewerDecision,
    GapReviewStatus,
    GapStatus,
)
from backend.services.human_review_service import (
    DeterministicHumanReviewService,
    HumanReviewService,
)


def build_pending_review() -> GapHumanReview:
    return GapHumanReview(
        gap_assessment_id=uuid4(),
        requirement_id=uuid4(),
        original_gap_status=GapStatus.PARTIALLY_ADDRESSED,
    )


def test_service_satisfies_protocol():
    service = DeterministicHumanReviewService()

    assert isinstance(service, HumanReviewService)


def test_create_pending_review():
    gap_assessment_id = uuid4()
    requirement_id = uuid4()

    service = DeterministicHumanReviewService()

    review = service.create_pending_review(
        gap_assessment_id=gap_assessment_id,
        requirement_id=requirement_id,
        original_gap_status=GapStatus.NOT_ADDRESSED,
    )

    assert review.gap_assessment_id == gap_assessment_id
    assert review.requirement_id == requirement_id
    assert review.original_gap_status == GapStatus.NOT_ADDRESSED
    assert review.status == GapReviewStatus.PENDING
    assert review.decision is None
    assert review.reviewed_at is None


def test_complete_review_accepts_automated_result():
    review = build_pending_review()
    service = DeterministicHumanReviewService()

    completed = service.complete_review(
        review,
        status=GapReviewStatus.APPROVED,
        decision=GapReviewerDecision.ACCEPT_AUTOMATED_RESULT,
        reviewer_id="analyst-101",
        reviewer_notes="The evidence supports the automated result.",
    )

    assert completed.review_id == review.review_id
    assert completed.status == GapReviewStatus.APPROVED
    assert (
        completed.decision
        == GapReviewerDecision.ACCEPT_AUTOMATED_RESULT
    )
    assert completed.reviewer_id == "analyst-101"
    assert completed.reviewed_at is not None
    assert completed.overridden_gap_status is None

    assert review.status == GapReviewStatus.PENDING
    assert review.reviewed_at is None


def test_complete_review_can_override_gap_status():
    review = build_pending_review()
    service = DeterministicHumanReviewService()

    completed = service.complete_review(
        review,
        status=GapReviewStatus.APPROVED,
        decision=GapReviewerDecision.OVERRIDE_GAP_STATUS,
        reviewer_id="senior-reviewer",
        reviewer_notes="The policy fully addresses the requirement.",
        overridden_gap_status=GapStatus.FULLY_ADDRESSED,
    )

    assert (
        completed.decision
        == GapReviewerDecision.OVERRIDE_GAP_STATUS
    )
    assert (
        completed.original_gap_status
        == GapStatus.PARTIALLY_ADDRESSED
    )
    assert (
        completed.overridden_gap_status
        == GapStatus.FULLY_ADDRESSED
    )


def test_complete_review_rejects_override_without_status():
    review = build_pending_review()
    service = DeterministicHumanReviewService()

    with pytest.raises(
        ValueError,
        match="overridden_gap_status is required",
    ):
        service.complete_review(
            review,
            status=GapReviewStatus.APPROVED,
            decision=GapReviewerDecision.OVERRIDE_GAP_STATUS,
            reviewer_id="analyst-101",
        )


def test_complete_review_rejects_status_for_non_override_decision():
    review = build_pending_review()
    service = DeterministicHumanReviewService()

    with pytest.raises(
        ValueError,
        match="only allowed",
    ):
        service.complete_review(
            review,
            status=GapReviewStatus.APPROVED,
            decision=GapReviewerDecision.ACCEPT_AUTOMATED_RESULT,
            reviewer_id="analyst-101",
            overridden_gap_status=GapStatus.FULLY_ADDRESSED,
        )


def test_complete_review_rejects_pending_as_final_status():
    review = build_pending_review()
    service = DeterministicHumanReviewService()

    with pytest.raises(
        ValueError,
        match="approved, rejected",
    ):
        service.complete_review(
            review,
            status=GapReviewStatus.PENDING,
            decision=GapReviewerDecision.REQUEST_MORE_EVIDENCE,
            reviewer_id="analyst-101",
        )


def test_completed_review_cannot_be_completed_again():
    review = build_pending_review()
    service = DeterministicHumanReviewService()

    completed = service.complete_review(
        review,
        status=GapReviewStatus.APPROVED,
        decision=GapReviewerDecision.ACCEPT_AUTOMATED_RESULT,
        reviewer_id="analyst-101",
    )

    with pytest.raises(
        ValueError,
        match="Only pending human reviews",
    ):
        service.complete_review(
            completed,
            status=GapReviewStatus.REJECTED,
            decision=GapReviewerDecision.ESCALATE,
            reviewer_id="legal-reviewer",
        )


def test_complete_review_rejects_empty_reviewer_id():
    review = build_pending_review()
    service = DeterministicHumanReviewService()

    with pytest.raises(ValueError):
        service.complete_review(
            review,
            status=GapReviewStatus.APPROVED,
            decision=GapReviewerDecision.ACCEPT_AUTOMATED_RESULT,
            reviewer_id="",
        )