"""Tests for deterministic confidence scoring."""

from uuid import uuid4

import pytest

from backend.models import (
    GapConfidenceLevel,
    PolicyStatement,
    PolicyStatementType,
    RequirementCandidate,
    RequirementModality,
)
from backend.services.confidence_scoring_service import (
    ConfidenceScorer,
    ConfidenceThresholds,
    ConfidenceWeights,
    DeterministicConfidenceScoringService,
)
from backend.services.gap_comparison_service import (
    DeterministicGapComparisonService,
)


def make_requirement(
    *,
    action: str = "retain",
    object_text: str | None = "customer account records",
    timing: str | None = None,
    condition: str | None = None,
    extraction_confidence: float = 0.95,
) -> RequirementCandidate:
    return RequirementCandidate(
        document_id=uuid4(),
        chunk_id=uuid4(),
        page_number=2,
        chunk_index=3,
        source_text=("The firm must retain customer account records."),
        subject="The firm",
        action=action,
        object=object_text,
        timing=timing,
        condition=condition,
        modality=RequirementModality.MANDATORY,
        matched_trigger="must",
        extraction_confidence=extraction_confidence,
    )


def make_policy(
    *,
    action: str = "retain",
    object_text: str | None = "customer account records",
    timing: str | None = None,
    condition: str | None = None,
    extraction_confidence: float = 0.94,
) -> PolicyStatement:
    return PolicyStatement(
        document_id=uuid4(),
        chunk_id=uuid4(),
        page_number=6,
        chunk_index=9,
        source_text=("Records staff must retain " "customer account records."),
        subject="Records staff",
        action=action,
        object=object_text,
        timing=timing,
        condition=condition,
        responsible_party="Records staff",
        statement_type=(PolicyStatementType.RECORD_RETENTION),
        matched_trigger="must",
        extraction_confidence=extraction_confidence,
    )


def create_gap_assessment(
    requirement: RequirementCandidate,
    policies: list[PolicyStatement],
):
    service = DeterministicGapComparisonService()

    return service.compare(
        requirement=requirement,
        policy_statements=policies,
    )


def test_service_satisfies_confidence_protocol():
    service = DeterministicConfidenceScoringService()

    assert isinstance(
        service,
        ConfidenceScorer,
    )


def test_strong_complete_evidence_produces_high_confidence():
    requirement = make_requirement(
        timing="for at least six years",
    )

    policy = make_policy(
        timing="for at least six years",
    )

    gap = create_gap_assessment(
        requirement,
        [policy],
    )

    service = DeterministicConfidenceScoringService()

    result = service.score(
        requirement=requirement,
        gap_assessment=gap,
        policy_statements=[policy],
        retrieval_scores={
            policy.chunk_id: 0.94,
        },
    )

    assert result.confidence_level == (GapConfidenceLevel.HIGH)

    assert result.confidence_score >= 0.80

    assert result.components.retrieval_score == 0.94


def test_missing_timing_reduces_confidence():
    requirement = make_requirement(
        timing="for at least six years",
    )

    policy = make_policy(
        timing=None,
    )

    gap = create_gap_assessment(
        requirement,
        [policy],
    )

    service = DeterministicConfidenceScoringService()

    result = service.score(
        requirement=requirement,
        gap_assessment=gap,
        policy_statements=[policy],
        retrieval_scores={
            policy.chunk_id: 0.90,
        },
    )

    assert result.components.evidence_completeness_score < 1.0

    assert result.requires_human_review is True

    assert any("timing" in factor.lower() for factor in result.limiting_factors)


def test_low_extraction_confidence_is_reported():
    requirement = make_requirement(
        extraction_confidence=0.35,
    )

    policy = make_policy(
        extraction_confidence=0.40,
    )

    gap = create_gap_assessment(
        requirement,
        [policy],
    )

    service = DeterministicConfidenceScoringService()

    result = service.score(
        requirement=requirement,
        gap_assessment=gap,
        policy_statements=[policy],
        retrieval_scores={
            policy.chunk_id: 0.90,
        },
    )

    assert result.components.requirement_extraction_score == 0.35

    assert any(
        "requirement extraction" in factor.lower() for factor in result.limiting_factors
    )

    assert any(
        "policy statement extraction" in factor.lower()
        for factor in result.limiting_factors
    )


def test_missing_retrieval_scores_uses_neutral_score():
    requirement = make_requirement()
    policy = make_policy()

    gap = create_gap_assessment(
        requirement,
        [policy],
    )

    service = DeterministicConfidenceScoringService()

    result = service.score(
        requirement=requirement,
        gap_assessment=gap,
        policy_statements=[policy],
    )

    assert result.components.retrieval_score == 0.5

    assert any(
        "no retrieval similarity" in factor.lower()
        for factor in result.limiting_factors
    )


def test_no_policy_evidence_requires_human_review():
    requirement = make_requirement()

    gap = create_gap_assessment(
        requirement,
        [],
    )

    service = DeterministicConfidenceScoringService()

    result = service.score(
        requirement=requirement,
        gap_assessment=gap,
        policy_statements=[],
    )

    assert result.supporting_evidence_count == 0
    assert result.requires_human_review is True

    assert any(
        "no policy statements" in factor.lower() for factor in result.limiting_factors
    )


def test_selected_policy_extraction_confidence_is_used():
    requirement = make_requirement()

    unrelated = make_policy(
        action="notify",
        object_text="employees",
        extraction_confidence=0.20,
    )

    related = make_policy(
        action="retain",
        object_text="customer account records",
        extraction_confidence=0.91,
    )

    gap = create_gap_assessment(
        requirement,
        [
            unrelated,
            related,
        ],
    )

    service = DeterministicConfidenceScoringService()

    result = service.score(
        requirement=requirement,
        gap_assessment=gap,
        policy_statements=[
            unrelated,
            related,
        ],
        retrieval_scores={
            unrelated.chunk_id: 0.30,
            related.chunk_id: 0.95,
        },
    )

    assert result.components.policy_extraction_score == 0.91

    assert result.components.retrieval_score == 0.95


def test_weights_must_sum_to_one():
    with pytest.raises(ValueError):
        ConfidenceWeights(
            requirement_extraction=0.20,
            policy_extraction=0.20,
            retrieval=0.20,
            comparison=0.20,
            evidence_completeness=0.20,
            evidence_quantity=0.20,
        )


def test_negative_weight_is_rejected():
    with pytest.raises(ValueError):
        ConfidenceWeights(
            requirement_extraction=-0.10,
            policy_extraction=0.20,
            retrieval=0.20,
            comparison=0.30,
            evidence_completeness=0.30,
            evidence_quantity=0.10,
        )


def test_invalid_threshold_order_is_rejected():
    with pytest.raises(ValueError):
        ConfidenceThresholds(
            high=0.50,
            medium=0.80,
        )
