"""Tests for the deterministic policy gap comparison engine."""

from uuid import uuid4

import pytest

from backend.models import (
    GapStatus,
    PolicyStatement,
    PolicyStatementType,
    RequirementCandidate,
    RequirementModality,
)
from backend.services.gap_comparison_service import (
    ComparisonThresholds,
    DeterministicGapComparisonService,
    GapComparator,
)


def make_requirement(
    *,
    action: str = "retain",
    object_text: str | None = "customer account records",
    timing: str | None = None,
    condition: str | None = None,
    modality: RequirementModality = (RequirementModality.MANDATORY),
) -> RequirementCandidate:
    """Create a valid regulatory requirement candidate."""

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
        modality=modality,
        matched_trigger="must",
        extraction_confidence=0.95,
    )


def make_policy(
    *,
    action: str = "retain",
    object_text: str | None = "customer account records",
    timing: str | None = None,
    condition: str | None = None,
    statement_type: PolicyStatementType = (PolicyStatementType.RECORD_RETENTION),
    source_text: str = ("Records staff must retain customer account records."),
) -> PolicyStatement:
    """Create a valid internal policy statement."""

    return PolicyStatement(
        document_id=uuid4(),
        chunk_id=uuid4(),
        page_number=6,
        chunk_index=9,
        source_text=source_text,
        subject="Records staff",
        action=action,
        object=object_text,
        timing=timing,
        condition=condition,
        responsible_party="Records staff",
        statement_type=statement_type,
        matched_trigger="must",
        extraction_confidence=0.94,
    )


def test_service_satisfies_comparator_protocol():
    service = DeterministicGapComparisonService()

    assert isinstance(
        service,
        GapComparator,
    )


def test_identical_requirement_is_fully_addressed():
    requirement = make_requirement(
        timing="for at least six years",
    )

    policy = make_policy(
        timing="for at least six years",
    )

    service = DeterministicGapComparisonService()

    assessment = service.compare(
        requirement,
        [policy],
    )

    assert assessment.status == (GapStatus.FULLY_ADDRESSED)
    assert assessment.best_match is not None
    assert assessment.deterministic_score >= 0.82


def test_action_synonym_can_match():
    requirement = make_requirement(
        action="preserve",
    )

    policy = make_policy(
        action="retain",
    )

    service = DeterministicGapComparisonService()

    assessment = service.compare(
        requirement,
        [policy],
    )

    assert assessment.best_match is not None
    assert assessment.best_match.components.action_score >= 0.9


def test_missing_timing_creates_partial_gap():
    requirement = make_requirement(
        timing="for at least six years",
    )

    policy = make_policy(
        timing=None,
    )

    service = DeterministicGapComparisonService()

    assessment = service.compare(
        requirement,
        [policy],
    )

    assert assessment.status == (GapStatus.PARTIALLY_ADDRESSED)
    assert assessment.best_match is not None
    assert assessment.best_match.components.timing_score == 0.0


def test_unrelated_policy_is_not_addressed():
    requirement = make_requirement(
        action="retain",
        object_text="customer account records",
    )

    policy = make_policy(
        action="notify",
        object_text="employees about office closures",
        statement_type=PolicyStatementType.CONTROL,
    )

    service = DeterministicGapComparisonService()

    assessment = service.compare(
        requirement,
        [policy],
    )

    assert assessment.status in {
        GapStatus.NOT_ADDRESSED,
        GapStatus.INSUFFICIENT_EVIDENCE,
    }

    assert assessment.deterministic_score < 0.45


def test_no_policy_evidence_is_not_addressed():
    requirement = make_requirement()

    service = DeterministicGapComparisonService()

    assessment = service.compare(
        requirement,
        [],
    )

    assert assessment.status == (GapStatus.NOT_ADDRESSED)
    assert assessment.best_match is None
    assert assessment.evaluated_policy_count == 0
    assert assessment.deterministic_score == 0.0


def test_mandatory_requirement_conflicts_with_prohibition():
    requirement = make_requirement(
        action="retain",
        object_text="customer account records",
        modality=RequirementModality.MANDATORY,
    )

    policy = make_policy(
        action="retain",
        object_text="customer account records",
        statement_type=PolicyStatementType.PROHIBITION,
        source_text=("Employees must not retain customer account records."),
    )

    service = DeterministicGapComparisonService()

    assessment = service.compare(
        requirement,
        [policy],
    )

    assert assessment.status == (GapStatus.CONTRADICTED)
    assert assessment.best_match is not None
    assert assessment.best_match.is_contradiction is True


def test_prohibition_conflicts_with_permission():
    requirement = make_requirement(
        action="disclose",
        object_text="confidential customer information",
        modality=RequirementModality.PROHIBITED,
    )

    policy = make_policy(
        action="disclose",
        object_text="confidential customer information",
        statement_type=PolicyStatementType.PERMISSION,
        source_text=("Employees may disclose confidential " "customer information."),
    )

    service = DeterministicGapComparisonService()

    assessment = service.compare(
        requirement,
        [policy],
    )

    assert assessment.status == (GapStatus.CONTRADICTED)


def test_best_policy_match_is_selected():
    requirement = make_requirement(
        action="review",
        object_text="customer complaints",
    )

    unrelated = make_policy(
        action="retain",
        object_text="account records",
    )

    related = make_policy(
        action="review",
        object_text="customer complaints",
        statement_type=PolicyStatementType.REVIEW,
    )

    service = DeterministicGapComparisonService()

    assessment = service.compare(
        requirement,
        [
            unrelated,
            related,
        ],
    )

    assert assessment.best_match is not None
    assert assessment.best_match.policy_statement_id == related.statement_id

    assert assessment.evaluated_policy_count == 2


def test_missing_condition_reduces_score():
    requirement = make_requirement(
        action="notify",
        object_text="the customer",
        condition="if account information changes",
    )

    policy = make_policy(
        action="notify",
        object_text="the customer",
        condition=None,
        statement_type=PolicyStatementType.CONTROL,
    )

    service = DeterministicGapComparisonService()

    assessment = service.compare(
        requirement,
        [policy],
    )

    assert assessment.best_match is not None
    assert assessment.best_match.components.condition_score == 0.0
    assert assessment.requires_human_review is True


def test_comparison_preserves_policy_provenance():
    requirement = make_requirement()
    policy = make_policy()

    service = DeterministicGapComparisonService()

    assessment = service.compare(
        requirement,
        [policy],
    )

    match = assessment.best_match

    assert match is not None
    assert match.policy_statement_id == policy.statement_id
    assert match.policy_document_id == policy.document_id
    assert match.policy_chunk_id == policy.chunk_id
    assert match.page_number == policy.page_number
    assert match.chunk_index == policy.chunk_index


def test_compare_many_preserves_requirement_order():
    first = make_requirement(
        action="retain",
        object_text="account records",
    )

    second = make_requirement(
        action="review",
        object_text="customer complaints",
    )

    policies = [
        make_policy(
            action="retain",
            object_text="account records",
        ),
        make_policy(
            action="review",
            object_text="customer complaints",
            statement_type=PolicyStatementType.REVIEW,
        ),
    ]

    service = DeterministicGapComparisonService()

    assessments = service.compare_many(
        [first, second],
        policies,
    )

    assert len(assessments) == 2
    assert assessments[0].requirement_id == first.requirement_id
    assert assessments[1].requirement_id == second.requirement_id


@pytest.mark.parametrize(
    "fully,partial,minimum",
    [
        (1.1, 0.5, 0.2),
        (0.8, -0.1, 0.0),
        (0.4, 0.8, 0.2),
        (0.8, 0.4, 0.6),
    ],
)
def test_invalid_thresholds_are_rejected(
    fully,
    partial,
    minimum,
):
    with pytest.raises(ValueError):
        ComparisonThresholds(
            fully_addressed=fully,
            partially_addressed=partial,
            minimum_evidence=minimum,
        )


def test_missing_required_timing_cannot_be_fully_addressed():
    """A missing regulatory timing component must prevent full coverage."""

    requirement = make_requirement(
        timing="for at least six years",
    )

    policy = make_policy(
        timing=None,
    )

    service = DeterministicGapComparisonService()

    assessment = service.compare(
        requirement,
        [policy],
    )

    assert assessment.status != (GapStatus.FULLY_ADDRESSED)

    assert assessment.requires_human_review is True


def test_missing_required_condition_cannot_be_fully_addressed():
    """A missing regulatory condition must prevent full coverage."""

    requirement = make_requirement(
        action="notify",
        object_text="the customer",
        condition="if account information changes",
    )

    policy = make_policy(
        action="notify",
        object_text="the customer",
        condition=None,
        statement_type=PolicyStatementType.CONTROL,
    )

    service = DeterministicGapComparisonService()

    assessment = service.compare(
        requirement,
        [policy],
    )

    assert assessment.status == (GapStatus.PARTIALLY_ADDRESSED)

    assert assessment.requires_human_review is True
