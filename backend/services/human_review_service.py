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
    """Manage auditable human-review decisions."""

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
        reviewer_id = reviewer_id.strip()

        if not reviewer_id:
            raise ValueError("reviewer_id must not be empty.")

        if reviewer_notes is not None:
            reviewer_notes = reviewer_notes.strip() or None

        self._validate_transition(
            review=review,
            new_status=status,
        )

        self._validate_decision(
            status=status,
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
        *,
        review: GapHumanReview,
        new_status: GapReviewStatus,
    ) -> None:
        if review.status != GapReviewStatus.PENDING:
            raise ValueError(
                "Only pending human reviews can be completed."
            )

        if new_status not in cls._COMPLETED_STATUSES:
            raise ValueError(
                "The completed review status must be approved, "
                "rejected, or needs_revision."
            )

    @staticmethod
    def _validate_decision(
        *,
        status: GapReviewStatus,
        decision: GapReviewerDecision,
        overridden_gap_status: GapStatus | None,
    ) -> None:
        if (
            decision == GapReviewerDecision.OVERRIDE_GAP_STATUS
            and overridden_gap_status is None
        ):
            raise ValueError(
                "overridden_gap_status is required when overriding "
                "the automated gap status."
            )

        if (
            decision != GapReviewerDecision.OVERRIDE_GAP_STATUS
            and overridden_gap_status is not None
        ):
            raise ValueError(
                "overridden_gap_status is only allowed for an "
                "override decision."
            )

        if (
            decision == GapReviewerDecision.ACCEPT_AUTOMATED_RESULT
            and status != GapReviewStatus.APPROVED
        ):
            raise ValueError(
                "Accepting the automated result requires approved status."
            )

        if (
            decision == GapReviewerDecision.REQUEST_MORE_EVIDENCE
            and status != GapReviewStatus.NEEDS_REVISION
        ):
            raise ValueError(
                "Requesting more evidence requires needs_revision status."
            )

        if (
            decision == GapReviewerDecision.ESCALATE
            and status != GapReviewStatus.NEEDS_REVISION
        ):
            raise ValueError(
                "Escalating a finding requires needs_revision status."
            )