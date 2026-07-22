from unittest.mock import Mock
from uuid import uuid4

import pytest

from backend.models import (
    GapAssessment,
    GapConfidenceAssessment,
    GapExplanation,
    GapRiskAssessment,
    RequirementCandidate,
    RequirementModality,
)
from backend.services.explanation_service import (
    DeterministicExplanationService,
    ExplanationService,
)


def build_requirement() -> RequirementCandidate:
    requirement_id = uuid4()

    return RequirementCandidate(
        requirement_id=requirement_id,
        document_id=uuid4(),
        chunk_id=uuid4(),
        page_number=1,
        chunk_index=0,
        source_text=("The organization must retain customer records for seven years."),
        subject="organization",
        action="retain",
        object="customer records",
        condition=None,
        timing="seven years",
        modality=RequirementModality.MANDATORY,
        matched_trigger="must",
        extraction_confidence=0.95,
    )


def test_service_satisfies_explanation_protocol():
    service = DeterministicExplanationService()

    assert isinstance(service, ExplanationService)


def test_explain_creates_structured_gap_explanation():
    requirement = build_requirement()

    gap_assessment = GapAssessment.model_construct(
        requirement_id=requirement.requirement_id,
        status="partially_covered",
        best_match=None,
        rationale=(
            "The policy addresses record retention but does not specify "
            "the required seven-year duration."
        ),
        requires_human_review=True,
    )

    confidence_assessment = GapConfidenceAssessment.model_construct(
        requirement_id=requirement.requirement_id,
        confidence_score=0.88,
        confidence_level="high",
        positive_factors=["Relevant retention evidence was identified"],
        limiting_factors=["The retention duration is missing"],
        requires_human_review=True,
    )

    risk_assessment = GapRiskAssessment.model_construct(
        requirement_id=requirement.requirement_id,
        risk_score=0.82,
        risk_level="high",
        risk_factors=["The regulatory requirement is mandatory"],
        mitigating_factors=[],
        remediation_priority="high",
        requires_human_review=True,
    )

    service = DeterministicExplanationService()

    explanation = service.explain(
        requirement,
        gap_assessment,
        confidence_assessment,
        risk_assessment,
    )

    assert isinstance(explanation, GapExplanation)
    assert explanation.requirement_id == requirement.requirement_id

    assert explanation.requirement_summary == (
        "The organization must retain customer records for seven years."
    )

    assert explanation.policy_summary == ("No relevant policy evidence was identified.")

    assert explanation.gap_reason == (
        "The policy addresses record retention but does not specify "
        "the required seven-year duration."
    )

    assert "Confidence is high (88%)." in explanation.confidence_reason
    assert "Risk is high (82%)." in explanation.risk_reason
    assert "high remediation priority" in explanation.recommended_action
    assert explanation.requires_human_review is True


def test_explain_uses_policy_match_source_text():
    requirement = build_requirement()

    policy_match = Mock()
    policy_match.source_text = (
        "The company retains customer records according to business needs."
    )

    gap_assessment = GapAssessment.model_construct(
        requirement_id=requirement.requirement_id,
        status="partially_covered",
        best_match=policy_match,
        rationale="The required retention duration is not defined.",
        requires_human_review=True,
    )

    confidence_assessment = GapConfidenceAssessment.model_construct(
        requirement_id=requirement.requirement_id,
        confidence_score=0.80,
        confidence_level="high",
        positive_factors=[],
        limiting_factors=[],
        requires_human_review=True,
    )

    risk_assessment = GapRiskAssessment.model_construct(
        requirement_id=requirement.requirement_id,
        risk_score=0.75,
        risk_level="high",
        risk_factors=[],
        mitigating_factors=[],
        remediation_priority="high",
        requires_human_review=True,
    )

    service = DeterministicExplanationService()

    explanation = service.explain(
        requirement,
        gap_assessment,
        confidence_assessment,
        risk_assessment,
    )

    assert explanation.policy_summary == (
        "The company retains customer records according to business needs."
    )


def test_explain_rejects_mismatched_requirement_ids():
    requirement = build_requirement()

    gap_assessment = GapAssessment.model_construct(
        requirement_id=uuid4(),
    )

    confidence_assessment = GapConfidenceAssessment.model_construct(
        requirement_id=requirement.requirement_id,
    )

    risk_assessment = GapRiskAssessment.model_construct(
        requirement_id=requirement.requirement_id,
    )

    service = DeterministicExplanationService()

    with pytest.raises(
        ValueError,
        match="Requirement and assessment IDs",
    ):
        service.explain(
            requirement,
            gap_assessment,
            confidence_assessment,
            risk_assessment,
        )


def test_human_review_is_false_only_when_all_assessments_allow_it():
    requirement = build_requirement()

    gap_assessment = GapAssessment.model_construct(
        requirement_id=requirement.requirement_id,
        status="covered",
        best_match=None,
        rationale="The policy fully addresses the requirement.",
        requires_human_review=False,
    )

    confidence_assessment = GapConfidenceAssessment.model_construct(
        requirement_id=requirement.requirement_id,
        confidence_score=0.98,
        confidence_level="high",
        positive_factors=[],
        limiting_factors=[],
        requires_human_review=False,
    )

    risk_assessment = GapRiskAssessment.model_construct(
        requirement_id=requirement.requirement_id,
        risk_score=0.10,
        risk_level="low",
        risk_factors=[],
        mitigating_factors=["The policy fully covers the requirement"],
        remediation_priority="low",
        requires_human_review=False,
    )

    service = DeterministicExplanationService()

    explanation = service.explain(
        requirement,
        gap_assessment,
        confidence_assessment,
        risk_assessment,
    )

    assert explanation.requires_human_review is False
