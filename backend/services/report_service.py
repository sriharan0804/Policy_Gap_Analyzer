from collections import Counter
from typing import Protocol, runtime_checkable
from uuid import UUID

from backend.models import (
    AnalysisResult,
    ComplianceReport,
    GapExplanation,
    GapHumanReview,
    GapReviewerDecision,
    GapReviewStatus,
    GapStatus,
    ReportSummary,
    RequirementCandidate,
    RequirementReport,
    RiskLevel,
)


@runtime_checkable
class ReportService(Protocol):
    def generate(
        self,
        analysis_result: AnalysisResult,
        *,
        human_reviews: list[GapHumanReview] | None = None,
    ) -> ComplianceReport:
        ...


class DeterministicReportService:
    """Generates a structured compliance report from analysis results."""

    def generate(
        self,
        analysis_result: AnalysisResult,
        *,
        human_reviews: list[GapHumanReview] | None = None,
    ) -> ComplianceReport:
        reviews = human_reviews or []

        requirements_by_id = {
            requirement.requirement_id: requirement
            for requirement in analysis_result.requirements
        }

        confidence_by_requirement_id = {
            assessment.requirement_id: assessment
            for assessment in analysis_result.confidence_assessments
        }

        risk_by_requirement_id = {
            assessment.requirement_id: assessment
            for assessment in analysis_result.risk_assessments
        }

        explanation_by_requirement_id = {
            explanation.requirement_id: explanation
            for explanation in analysis_result.explanations
        }

        review_by_requirement_id = {
            review.requirement_id: review
            for review in reviews
        }

        requirement_reports = []

        for gap_assessment in analysis_result.gap_assessments:
            requirement_id = gap_assessment.requirement_id

            requirement = requirements_by_id[requirement_id]
            confidence = confidence_by_requirement_id[requirement_id]
            risk = risk_by_requirement_id[requirement_id]
            explanation = explanation_by_requirement_id[requirement_id]
            review = review_by_requirement_id.get(requirement_id)

            requirement_report = self._build_requirement_report(
                requirement=requirement,
                gap_assessment=gap_assessment,
                confidence=confidence,
                risk=risk,
                explanation=explanation,
                review=review,
            )

            requirement_reports.append(requirement_report)

        summary = self._build_summary(requirement_reports)

        return ComplianceReport(
            analysis_id=analysis_result.analysis_id,
            regulatory_document_id=analysis_result.regulatory_document_id,
            policy_document_id=analysis_result.policy_document_id,
            summary=summary,
            requirement_reports=requirement_reports,
        )

    @staticmethod
    def _build_requirement_report(
        *,
        requirement,
        gap_assessment,
        confidence,
        risk,
        explanation: GapExplanation,
        review: GapHumanReview | None,
    ) -> RequirementReport:
        effective_gap_status = gap_assessment.status
        review_status = GapReviewStatus.PENDING
        reviewer_decision = None

        if review is not None:
            review_status = review.status
            reviewer_decision = review.decision

            if (
                review.decision
                == GapReviewerDecision.OVERRIDE_GAP_STATUS
                and review.overridden_gap_status is not None
            ):
                effective_gap_status = review.overridden_gap_status

        return RequirementReport(
            requirement_id=requirement.requirement_id,
            gap_assessment_id=gap_assessment.assessment_id,
            requirement_summary=explanation.requirement_summary,
            policy_summary=explanation.policy_summary,
            gap_status=gap_assessment.status,
            gap_reason=explanation.gap_reason,
            confidence_score=confidence.confidence_score,
            confidence_level=confidence.confidence_level,
            confidence_reason=explanation.confidence_reason,
            risk_score=risk.risk_score,
            risk_level=risk.risk_level,
            risk_reason=explanation.risk_reason,
            recommended_action=explanation.recommended_action,
            requires_human_review=(
                gap_assessment.requires_human_review
                or confidence.requires_human_review
                or risk.requires_human_review
                or explanation.requires_human_review
            ),
            review_status=review_status,
            reviewer_decision=reviewer_decision,
            effective_gap_status=effective_gap_status,
        )

    @staticmethod
    def _build_summary(
        requirement_reports: list[RequirementReport],
    ) -> ReportSummary:
        status_counts = Counter(
            report.effective_gap_status
            for report in requirement_reports
        )

        risk_counts = Counter(
            report.risk_level
            for report in requirement_reports
        )

        total_requirements = len(requirement_reports)

        compliance_score = (
            DeterministicReportService._calculate_compliance_score(
                requirement_reports
            )
        )

        return ReportSummary(
            total_requirements=total_requirements,
            fully_addressed_count=status_counts[
                GapStatus.FULLY_ADDRESSED
            ],
            partially_addressed_count=status_counts[
                GapStatus.PARTIALLY_ADDRESSED
            ],
            not_addressed_count=status_counts[
                GapStatus.NOT_ADDRESSED
            ],
            contradicted_count=status_counts[
                GapStatus.CONTRADICTED
            ],
            insufficient_evidence_count=status_counts[
                GapStatus.INSUFFICIENT_EVIDENCE
            ],
            high_risk_count=risk_counts[RiskLevel.HIGH],
            critical_risk_count=risk_counts[RiskLevel.CRITICAL],
            compliance_score=compliance_score,
            human_review_required_count=sum(
                report.requires_human_review
                for report in requirement_reports
            ),
        )

    @staticmethod
    def _calculate_compliance_score(
        requirement_reports: list[RequirementReport],
    ) -> float:
        if not requirement_reports:
            return 0.0

        status_weights = {
            GapStatus.FULLY_ADDRESSED: 1.0,
            GapStatus.PARTIALLY_ADDRESSED: 0.5,
            GapStatus.NOT_ADDRESSED: 0.0,
            GapStatus.CONTRADICTED: 0.0,
            GapStatus.INSUFFICIENT_EVIDENCE: 0.0,
        }

        total_score = sum(
            status_weights[report.effective_gap_status]
            for report in requirement_reports
        )

        return round(total_score / len(requirement_reports), 4)