from datetime import datetime, timezone
from typing import Protocol, runtime_checkable
from uuid import UUID

from backend.models import (
    GapHumanReview,
    GapReviewerDecision,
    GapReviewStatus,
    GapStatus,
)


@runtime_checkable
class HumanReviewService(Protocol):
    def create_pending_review(
        self,
        *,
        gap_assessment_id: UUID,
        requirement_id: UUID,
        original_gap_status: GapStatus,
    ) -> GapHumanReview:
        ...

    def complete_review(
        self,
        review: GapHumanReview,
        *,
        status: GapReviewStatus,
        decision: GapReviewerDecision,
        reviewer_id: str,
        reviewer_notes: str | None = None,
        overridden_gap_status: GapStatus | None = None,
    ) -> GapHumanReview:
        ...


class DeterministicHumanReviewService:
    """Manages auditable human-review decisions for gap assessments."""

    _COMPLETED_STATUSES = {
        GapReviewStatus.APPROVED,
        GapReviewStatus.REJECTED,
        GapReviewStatus.NEEDS_REVISION,
    }

    def create_pending_review(
        self,
        *,
        gap_assessment_id: UUID,
        requirement_id: UUID,
        original_gap_status: GapStatus,
    ) -> GapHumanReview:
        return GapHumanReview(
            gap_assessment_id=gap_assessment_id,
            requirement_id=requirement_id,
            original_gap_status=original_gap_status,
        )

    def complete_review(
        self,
        review: GapHumanReview,
        *,
        status: GapReviewStatus,
        decision: GapReviewerDecision,
        reviewer_id: str,
        reviewer_notes: str | None = None,
        overridden_gap_status: GapStatus | None = None,
    ) -> GapHumanReview:
        self._validate_transition(review, status)
        self._validate_decision(
            decision=decision,
            overridden_gap_status=overridden_gap_status,
        )

        return GapHumanReview(
    review_id=review.review_id,
    gap_assessment_id=review.gap_assessment_id,
    requirement_id=review.requirement_id,
    status=status,
    decision=decision,
    reviewer_id=reviewer_id,
    reviewer_notes=reviewer_notes,
    original_gap_status=review.original_gap_status,
    overridden_gap_status=overridden_gap_status,
    reviewed_at=datetime.now(timezone.utc),
)

    @classmethod
    def _validate_transition(
        cls,
        review: GapHumanReview,
        new_status: GapReviewStatus,
    ) -> None:
        if review.status != GapReviewStatus.PENDING:
            raise ValueError(
                "Only pending human reviews can be completed."
            )

        if new_status not in cls._COMPLETED_STATUSES:
            raise ValueError(
                "A completed review must be approved, rejected, "
                "or marked as needing revision."
            )

    @staticmethod
    def _validate_decision(
        *,
        decision: GapReviewerDecision,
        overridden_gap_status: GapStatus | None,
    ) -> None:
        if (
            decision == GapReviewerDecision.OVERRIDE_GAP_STATUS
            and overridden_gap_status is None
        ):
            raise ValueError(
                "overridden_gap_status is required for an override decision."
            )

        if (
            decision != GapReviewerDecision.OVERRIDE_GAP_STATUS
            and overridden_gap_status is not None
        ):
            raise ValueError(
                "overridden_gap_status is only allowed for an "
                "override decision."
            )