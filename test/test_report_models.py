from uuid import uuid4

import pytest
from pydantic import ValidationError

from backend.models import (
    ComplianceReport,
    GapConfidenceLevel,
    GapReviewerDecision,
    GapReviewStatus,
    GapStatus,
    ReportSummary,
    RequirementReport,
    RiskLevel,
)


def build_report_summary(
    *,
    total_requirements: int = 1,
) -> ReportSummary:
    return ReportSummary(
        total_requirements=total_requirements,
        fully_addressed_count=total_requirements,
        partially_addressed_count=0,
        not_addressed_count=0,
        contradicted_count=0,
        insufficient_evidence_count=0,
        high_risk_count=0,
        critical_risk_count=0,
        compliance_score=1.0,
        human_review_required_count=0,
    )


def build_requirement_report() -> RequirementReport:
    return RequirementReport(
        requirement_id=uuid4(),
        gap_assessment_id=uuid4(),
        requirement_summary=(
            "The organization must retain transaction records."
        ),
        policy_summary=(
            "The internal policy requires transaction record retention."
        ),
        gap_status=GapStatus.FULLY_ADDRESSED,
        gap_reason="The policy fully covers the required action.",
        confidence_score=0.91,
        confidence_level=GapConfidenceLevel.HIGH,
        confidence_reason="The requirement and policy evidence are clear.",
        risk_score=0.15,
        risk_level=RiskLevel.LOW,
        risk_reason="The requirement is fully addressed.",
        recommended_action="Continue monitoring policy implementation.",
        requires_human_review=False,
        review_status=GapReviewStatus.PENDING,
        effective_gap_status=GapStatus.FULLY_ADDRESSED,
    )


def test_report_summary_accepts_valid_counts():
    summary = ReportSummary(
        total_requirements=5,
        fully_addressed_count=2,
        partially_addressed_count=1,
        not_addressed_count=1,
        contradicted_count=1,
        insufficient_evidence_count=0,
        high_risk_count=1,
        critical_risk_count=1,
        compliance_score=0.5,
        human_review_required_count=3,
    )

    assert summary.total_requirements == 5
    assert summary.fully_addressed_count == 2
    assert summary.compliance_score == 0.5


def test_report_summary_rejects_inconsistent_gap_counts():
    with pytest.raises(
        ValidationError,
        match="Gap classification counts",
    ):
        ReportSummary(
            total_requirements=4,
            fully_addressed_count=2,
            partially_addressed_count=1,
            not_addressed_count=0,
            contradicted_count=0,
            insufficient_evidence_count=0,
            high_risk_count=0,
            critical_risk_count=0,
            compliance_score=0.75,
            human_review_required_count=1,
        )


def test_report_summary_rejects_excessive_risk_counts():
    with pytest.raises(
        ValidationError,
        match="cannot exceed total_requirements",
    ):
        ReportSummary(
            total_requirements=2,
            fully_addressed_count=2,
            partially_addressed_count=0,
            not_addressed_count=0,
            contradicted_count=0,
            insufficient_evidence_count=0,
            high_risk_count=2,
            critical_risk_count=1,
            compliance_score=1.0,
            human_review_required_count=0,
        )


def test_requirement_report_defaults_to_pending_review():
    report = build_requirement_report()

    assert report.review_status == GapReviewStatus.PENDING
    assert report.reviewer_decision is None
    assert report.effective_gap_status == GapStatus.FULLY_ADDRESSED


def test_requirement_report_accepts_completed_review():
    report = RequirementReport(
        requirement_id=uuid4(),
        gap_assessment_id=uuid4(),
        requirement_summary="The organization must encrypt personal data.",
        policy_summary="The current policy does not mandate encryption.",
        gap_status=GapStatus.NOT_ADDRESSED,
        gap_reason="No matching encryption control was found.",
        confidence_score=0.88,
        confidence_level=GapConfidenceLevel.HIGH,
        confidence_reason="The requirement is explicit and evidence is clear.",
        risk_score=0.92,
        risk_level=RiskLevel.CRITICAL,
        risk_reason="Sensitive data is not protected by a documented control.",
        recommended_action="Add a mandatory encryption policy.",
        requires_human_review=True,
        review_status=GapReviewStatus.APPROVED,
        reviewer_decision=GapReviewerDecision.OVERRIDE_GAP_STATUS,
        effective_gap_status=GapStatus.PARTIALLY_ADDRESSED,
    )

    assert report.review_status == GapReviewStatus.APPROVED
    assert (
        report.reviewer_decision
        == GapReviewerDecision.OVERRIDE_GAP_STATUS
    )
    assert report.gap_status == GapStatus.NOT_ADDRESSED
    assert report.effective_gap_status == GapStatus.PARTIALLY_ADDRESSED


def test_requirement_report_rejects_decision_for_pending_review():
    with pytest.raises(
        ValidationError,
        match="must be empty while review is pending",
    ):
        RequirementReport(
            requirement_id=uuid4(),
            gap_assessment_id=uuid4(),
            requirement_summary="The company must retain audit logs.",
            policy_summary="Audit logs are retained for seven years.",
            gap_status=GapStatus.FULLY_ADDRESSED,
            gap_reason="The retention requirement is fully covered.",
            confidence_score=0.9,
            confidence_level=GapConfidenceLevel.HIGH,
            confidence_reason="Both statements are clear.",
            risk_score=0.1,
            risk_level=RiskLevel.LOW,
            risk_reason="No material compliance gap was identified.",
            recommended_action="Continue the existing retention process.",
            requires_human_review=False,
            review_status=GapReviewStatus.PENDING,
            reviewer_decision=(
                GapReviewerDecision.ACCEPT_AUTOMATED_RESULT
            ),
            effective_gap_status=GapStatus.FULLY_ADDRESSED,
        )


def test_compliance_report_accepts_matching_totals():
    requirement_report = build_requirement_report()

    report = ComplianceReport(
        analysis_id=uuid4(),
        regulatory_document_id=uuid4(),
        policy_document_id=uuid4(),
        summary=build_report_summary(),
        requirement_reports=[requirement_report],
    )

    assert report.summary.total_requirements == 1
    assert len(report.requirement_reports) == 1
    assert report.report_version == "1.0"


def test_compliance_report_rejects_mismatched_totals():
    with pytest.raises(
        ValidationError,
        match="must match requirement_reports",
    ):
        ComplianceReport(
            analysis_id=uuid4(),
            regulatory_document_id=uuid4(),
            policy_document_id=uuid4(),
            summary=build_report_summary(total_requirements=2),
            requirement_reports=[build_requirement_report()],
        )


def test_compliance_report_supports_empty_analysis():
    report = ComplianceReport(
        analysis_id=uuid4(),
        regulatory_document_id=uuid4(),
        policy_document_id=uuid4(),
        summary=ReportSummary(
            total_requirements=0,
            fully_addressed_count=0,
            partially_addressed_count=0,
            not_addressed_count=0,
            contradicted_count=0,
            insufficient_evidence_count=0,
            high_risk_count=0,
            critical_risk_count=0,
            compliance_score=0.0,
            human_review_required_count=0,
        ),
        requirement_reports=[],
    )

    assert report.summary.total_requirements == 0
    assert report.requirement_reports == []