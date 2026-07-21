from uuid import uuid4

import pytest

from backend.models import (
    DataSensitivity,
    GapAssessment,
    GapConfidenceAssessment,
    GapConfidenceComponents,
    GapConfidenceLevel,
    GapStatus,
    RegulatoryImpact,
    RequirementCandidate,
    RequirementModality,
    RiskLevel,
)
from backend.services.risk_scoring_service import (
    DeterministicRiskScoringService,
    RiskScorer,
    RiskThresholds,
    RiskWeights,
)


def build_requirement(
    *,
    modality: RequirementModality = RequirementModality.MANDATORY,
    timing: str | None = None,
    condition: str | None = None,
) -> RequirementCandidate:
    return RequirementCandidate(
        requirement_id=uuid4(),
        chunk_id=uuid4(),
        document_id=uuid4(),
        page_number=1,
        chunk_index=0,
        source_text="The organization must retain records.",
        modality=modality,
        matched_trigger="must",
        subject="the organization",
        action="retain",
        object="records",
        timing=timing,
        condition=condition,
        extraction_confidence=0.92,
    )


def build_gap(
    *,
    requirement,
    status: GapStatus,
    score: float,
) -> GapAssessment:
    return GapAssessment(
        assessment_id=uuid4(),
        requirement_id=requirement.requirement_id,
        regulatory_document_id=requirement.document_id,
        regulatory_chunk_id=requirement.chunk_id,
        status=status,
        deterministic_score=score,
        best_match=None,
        evaluated_policy_count=0,
        rationale=["Deterministic comparison result."],
        requires_human_review=False,
    )


def build_confidence(
    *,
    requirement_id,
    gap_assessment_id,
    score: float = 0.90,
    requires_human_review: bool = False,
) -> GapConfidenceAssessment:
    level = (
        GapConfidenceLevel.HIGH
        if score >= 0.80
        else GapConfidenceLevel.MEDIUM if score >= 0.55 else GapConfidenceLevel.LOW
    )

    return GapConfidenceAssessment(
        gap_assessment_id=gap_assessment_id,
        requirement_id=requirement_id,
        confidence_score=score,
        confidence_level=level,
        components=GapConfidenceComponents(
            requirement_extraction_score=0.90,
            policy_extraction_score=0.90,
            retrieval_score=0.90,
            comparison_score=0.90,
            evidence_completeness_score=0.90,
            evidence_quantity_score=0.90,
        ),
        supporting_evidence_count=1,
        positive_factors=[],
        limiting_factors=[],
        requires_human_review=requires_human_review,
    )


def test_service_satisfies_risk_protocol():
    service = DeterministicRiskScoringService()

    assert isinstance(service, RiskScorer)


def test_fully_addressed_requirement_produces_low_risk():
    requirement = build_requirement()

    gap = build_gap(
        requirement=requirement,
        status=GapStatus.FULLY_ADDRESSED,
        score=0.95,
    )

    confidence = build_confidence(
        requirement_id=requirement.requirement_id,
        gap_assessment_id=gap.assessment_id,
    )

    result = DeterministicRiskScoringService().score(
        requirement,
        gap,
        confidence,
        regulatory_impact=RegulatoryImpact.LOW,
        data_sensitivity=DataSensitivity.NONE,
    )

    assert result.risk_level == RiskLevel.LOW
    assert result.risk_score < 0.40


def test_missing_mandatory_requirement_produces_high_risk():
    requirement = build_requirement()

    gap = build_gap(
        requirement=requirement,
        status=GapStatus.NOT_ADDRESSED,
        score=0.10,
    )

    confidence = build_confidence(
        requirement_id=requirement.requirement_id,
        gap_assessment_id=gap.assessment_id,
        score=0.92,
    )

    result = DeterministicRiskScoringService().score(
        requirement,
        gap,
        confidence,
        regulatory_impact=RegulatoryImpact.HIGH,
        data_sensitivity=DataSensitivity.PERSONAL,
    )

    assert result.risk_level in {
        RiskLevel.HIGH,
        RiskLevel.CRITICAL,
    }
    assert result.requires_human_review is True
    assert "The policy does not address the requirement." in result.risk_factors


def test_contradiction_increases_risk():
    requirement = build_requirement()

    gap = build_gap(
        requirement=requirement,
        status=GapStatus.CONTRADICTED,
        score=0.0,
    )

    confidence = build_confidence(
        requirement_id=requirement.requirement_id,
        gap_assessment_id=gap.assessment_id,
    )

    result = DeterministicRiskScoringService().score(
        requirement,
        gap,
        confidence,
        regulatory_impact=RegulatoryImpact.HIGH,
        data_sensitivity=DataSensitivity.CONFIDENTIAL,
    )

    assert result.components.contradiction_score == 1.0
    assert result.requires_human_review is True
    assert any("contradicts" in factor.lower() for factor in result.risk_factors)


def test_sensitive_data_increases_risk_score():
    requirement = build_requirement()

    gap = build_gap(
        requirement=requirement,
        status=GapStatus.PARTIALLY_ADDRESSED,
        score=0.60,
    )

    confidence = build_confidence(
        requirement_id=requirement.requirement_id,
        gap_assessment_id=gap.assessment_id,
    )

    service = DeterministicRiskScoringService()

    low_sensitivity = service.score(
        requirement,
        gap,
        confidence,
        regulatory_impact=RegulatoryImpact.MODERATE,
        data_sensitivity=DataSensitivity.NONE,
    )

    high_sensitivity = service.score(
        requirement,
        gap,
        confidence,
        regulatory_impact=RegulatoryImpact.MODERATE,
        data_sensitivity=DataSensitivity.HIGHLY_SENSITIVE,
    )

    assert high_sensitivity.risk_score > low_sensitivity.risk_score


def test_severe_regulatory_impact_increases_risk_score():
    requirement = build_requirement()

    gap = build_gap(
        requirement=requirement,
        status=GapStatus.PARTIALLY_ADDRESSED,
        score=0.60,
    )

    confidence = build_confidence(
        requirement_id=requirement.requirement_id,
        gap_assessment_id=gap.assessment_id,
    )

    service = DeterministicRiskScoringService()

    low_impact = service.score(
        requirement,
        gap,
        confidence,
        regulatory_impact=RegulatoryImpact.LOW,
        data_sensitivity=DataSensitivity.INTERNAL,
    )

    severe_impact = service.score(
        requirement,
        gap,
        confidence,
        regulatory_impact=RegulatoryImpact.SEVERE,
        data_sensitivity=DataSensitivity.INTERNAL,
    )

    assert severe_impact.risk_score > low_impact.risk_score


def test_low_confidence_finding_requires_human_review():
    requirement = build_requirement()

    gap = build_gap(
        requirement=requirement,
        status=GapStatus.PARTIALLY_ADDRESSED,
        score=0.50,
    )

    confidence = build_confidence(
        requirement_id=requirement.requirement_id,
        gap_assessment_id=gap.assessment_id,
        score=0.40,
        requires_human_review=True,
    )

    result = DeterministicRiskScoringService().score(
        requirement,
        gap,
        confidence,
        regulatory_impact=RegulatoryImpact.MODERATE,
        data_sensitivity=DataSensitivity.INTERNAL,
    )

    assert result.requires_human_review is True
    assert any(
        "limited confidence" in factor.lower() for factor in result.mitigating_factors
    )


def test_required_timing_is_reported_as_risk_factor():
    requirement = build_requirement(
        timing="within 30 days",
    )

    gap = build_gap(
        requirement=requirement,
        status=GapStatus.PARTIALLY_ADDRESSED,
        score=0.55,
    )

    confidence = build_confidence(
        requirement_id=requirement.requirement_id,
        gap_assessment_id=gap.assessment_id,
    )

    result = DeterministicRiskScoringService().score(
        requirement,
        gap,
        confidence,
        regulatory_impact=RegulatoryImpact.MODERATE,
        data_sensitivity=DataSensitivity.INTERNAL,
    )

    assert any("timing" in factor.lower() for factor in result.risk_factors)


def test_weights_must_sum_to_one():
    with pytest.raises(
        ValueError,
        match="sum to exactly 1.0",
    ):
        RiskWeights(
            gap_severity=0.50,
            regulatory_impact=0.20,
            requirement_criticality=0.15,
            data_sensitivity=0.15,
            confidence_reliability=0.10,
            contradiction=0.10,
        )


def test_negative_weight_is_rejected():
    with pytest.raises(
        ValueError,
        match="between 0.0 and 1.0",
    ):
        RiskWeights(
            gap_severity=-0.10,
            regulatory_impact=0.30,
            requirement_criticality=0.20,
            data_sensitivity=0.20,
            confidence_reliability=0.20,
            contradiction=0.20,
        )


def test_invalid_threshold_order_is_rejected():
    with pytest.raises(
        ValueError,
        match="medium <= high <= critical",
    ):
        RiskThresholds(
            medium=0.80,
            high=0.60,
            critical=0.90,
        )
