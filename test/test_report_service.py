from datetime import datetime, timezone
from uuid import uuid4

import pytest

from backend.models import (
    AnalysisResult,
    ComparisonComponents,
    GapAssessment,
    GapConfidenceAssessment,
    GapConfidenceComponents,
    GapConfidenceLevel,
    GapExplanation,
    GapHumanReview,
    GapReviewerDecision,
    GapReviewStatus,
    GapRiskAssessment,
    GapRiskComponents,
    GapStatus,
    PolicyMatch,
    RequirementCandidate,
    RequirementModality,
    RiskLevel,
    RegulatoryImpact,
    DataSensitivity,
)
from backend.services.report_service import (
    DeterministicReportService,
    ReportService,
)


def build_requirement(
    *,
    action: str,
) -> RequirementCandidate:
    document_id = uuid4()
    chunk_id = uuid4()

    return RequirementCandidate(
        document_id=document_id,
        chunk_id=chunk_id,
        page_number=1,
        chunk_index=0,
        source_text=f"The organization must {action}.",
        subject="organization",
        action=action,
        object=None,
        condition=None,
        timing=None,
        modality=RequirementModality.MANDATORY,
        matched_trigger="must",
        extraction_confidence=0.95,
    )


def build_gap_assessment(
    requirement: RequirementCandidate,
    *,
    status: GapStatus,
    deterministic_score: float,
) -> GapAssessment:
    return GapAssessment(
        requirement_id=requirement.requirement_id,
        regulatory_document_id=requirement.document_id,
        regulatory_chunk_id=requirement.chunk_id,
        status=status,
        best_match=None,
        evaluated_policy_count=0,
        deterministic_score=deterministic_score,
        rationale=["Deterministic comparison completed."],
        requires_human_review=status != GapStatus.FULLY_ADDRESSED,
    )


def build_confidence_assessment(
    gap_assessment: GapAssessment,
) -> GapConfidenceAssessment:
    return GapConfidenceAssessment(
        gap_assessment_id=gap_assessment.assessment_id,
        requirement_id=gap_assessment.requirement_id,
        confidence_score=0.85,
        confidence_level=GapConfidenceLevel.HIGH,
        components=GapConfidenceComponents(
            requirement_extraction_score=0.95,
            policy_extraction_score=0.80,
            retrieval_score=0.80,
            comparison_score=gap_assessment.deterministic_score,
            evidence_completeness_score=0.80,
            evidence_quantity_score=0.70,
        ),
        supporting_evidence_count=1,
        positive_factors=["The requirement is clearly written."],
        limiting_factors=[],
        requires_human_review=gap_assessment.requires_human_review,
    )


def build_risk_assessment(
    gap_assessment: GapAssessment,
) -> GapRiskAssessment:
    if gap_assessment.status == GapStatus.FULLY_ADDRESSED:
        risk_score = 0.15
        risk_level = RiskLevel.LOW
    else:
        risk_score = 0.85
        risk_level = RiskLevel.HIGH

    return GapRiskAssessment(
        gap_assessment_id=gap_assessment.assessment_id,
        requirement_id=gap_assessment.requirement_id,
        risk_score=risk_score,
        risk_level=risk_level,
        components=GapRiskComponents(
            gap_severity_score=risk_score,
            regulatory_impact_score=0.75,
            requirement_criticality_score=0.80,
            data_sensitivity_score=0.50,
            confidence_reliability_score=0.85,
            contradiction_score=0.0,
        ),
        regulatory_impact=RegulatoryImpact.HIGH,
        data_sensitivity=DataSensitivity.CONFIDENTIAL,
        risk_factors=["The requirement has regulatory significance."],
        mitigating_factors=[],
        remediation_priority="high" if risk_level == RiskLevel.HIGH else "low",
        requires_human_review=gap_assessment.requires_human_review,
    )


def build_explanation(
    requirement: RequirementCandidate,
    gap_assessment: GapAssessment,
) -> GapExplanation:
    return GapExplanation(
        requirement_id=requirement.requirement_id,
        requirement_summary=requirement.source_text,
        policy_summary="No directly matching policy statement was identified.",
        gap_reason=f"The gap was classified as {gap_assessment.status.value}.",
        confidence_reason="The assessment is supported by clear evidence.",
        risk_reason="Risk was calculated using deterministic factors.",
        recommended_action="Review and update the internal policy as needed.",
        requires_human_review=gap_assessment.requires_human_review,
    )


def build_analysis_result(
    statuses: list[GapStatus],
) -> AnalysisResult:
    regulatory_document_id = uuid4()
    policy_document_id = uuid4()

    requirements = []
    gap_assessments = []
    confidence_assessments = []
    risk_assessments = []
    explanations = []

    for index, status in enumerate(statuses):
        requirement = build_requirement(action=f"retain compliance record {index + 1}")

        requirement = requirement.model_copy(
            update={
                "document_id": regulatory_document_id,
            }
        )

        deterministic_score = 1.0 if status == GapStatus.FULLY_ADDRESSED else 0.0

        gap_assessment = build_gap_assessment(
            requirement,
            status=status,
            deterministic_score=deterministic_score,
        )

        requirements.append(requirement)
        gap_assessments.append(gap_assessment)
        confidence_assessments.append(build_confidence_assessment(gap_assessment))
        risk_assessments.append(build_risk_assessment(gap_assessment))
        explanations.append(build_explanation(requirement, gap_assessment))

    return AnalysisResult(
        regulatory_document_id=regulatory_document_id,
        policy_document_id=policy_document_id,
        requirements=requirements,
        policy_statements=[],
        gap_assessments=gap_assessments,
        confidence_assessments=confidence_assessments,
        risk_assessments=risk_assessments,
        explanations=explanations,
    )


@pytest.fixture
def empty_analysis_result() -> AnalysisResult:
    return AnalysisResult(
        regulatory_document_id=uuid4(),
        policy_document_id=uuid4(),
        requirements=[],
        policy_statements=[],
        gap_assessments=[],
        confidence_assessments=[],
        risk_assessments=[],
        explanations=[],
    )


@pytest.fixture
def complete_analysis_result() -> AnalysisResult:
    return build_analysis_result([GapStatus.NOT_ADDRESSED])


@pytest.fixture
def multiple_requirement_analysis_result() -> AnalysisResult:
    return build_analysis_result(
        [
            GapStatus.FULLY_ADDRESSED,
            GapStatus.NOT_ADDRESSED,
        ]
    )


# Helper builders and pytest fixtures go here.


def test_service_satisfies_protocol():
    service = DeterministicReportService()

    assert isinstance(service, ReportService)


def test_generate_empty_report(empty_analysis_result):
    service = DeterministicReportService()

    report = service.generate(empty_analysis_result)

    assert report.analysis_id == empty_analysis_result.analysis_id
    assert report.summary.total_requirements == 0
    assert report.summary.compliance_score == 0.0
    assert report.requirement_reports == []


def test_generate_report_uses_analysis_data(
    complete_analysis_result,
):
    service = DeterministicReportService()

    report = service.generate(complete_analysis_result)

    assert report.analysis_id == complete_analysis_result.analysis_id
    assert report.summary.total_requirements == 1
    assert len(report.requirement_reports) == 1

    requirement_report = report.requirement_reports[0]

    assert (
        requirement_report.requirement_id
        == complete_analysis_result.requirements[0].requirement_id
    )


def test_report_uses_human_override(
    complete_analysis_result,
):
    gap_assessment = complete_analysis_result.gap_assessments[0]

    review = GapHumanReview(
        gap_assessment_id=gap_assessment.assessment_id,
        requirement_id=gap_assessment.requirement_id,
        status=GapReviewStatus.APPROVED,
        decision=GapReviewerDecision.OVERRIDE_GAP_STATUS,
        reviewer_id="senior-reviewer",
        reviewer_notes="The policy provides partial coverage.",
        original_gap_status=gap_assessment.status,
        overridden_gap_status=GapStatus.PARTIALLY_ADDRESSED,
        reviewed_at=datetime.now(timezone.utc),
    )

    service = DeterministicReportService()

    report = service.generate(
        complete_analysis_result,
        human_reviews=[review],
    )

    requirement_report = report.requirement_reports[0]

    assert requirement_report.gap_status == gap_assessment.status
    assert requirement_report.effective_gap_status == GapStatus.PARTIALLY_ADDRESSED
    assert (
        requirement_report.reviewer_decision == GapReviewerDecision.OVERRIDE_GAP_STATUS
    )


def test_compliance_score_uses_effective_statuses(
    multiple_requirement_analysis_result,
):
    service = DeterministicReportService()

    report = service.generate(multiple_requirement_analysis_result)

    assert report.summary.compliance_score == 0.5
