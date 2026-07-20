

from datetime import date, datetime, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError

from backend.models import (
    ConfidenceAssessment,
    ConfidenceComponents,
    DocumentMetadata,
    DocumentType,
    Evidence,
    Finding,
    GapClassification,
    HumanReview,
    IssuingAuthority,
    ProcessingStatus,
    RegulatoryRequirement,
    RequirementModality,
    RequirementType,
    ReviewDecision,
    RiskAssessment,
    RiskComponents,
    RiskLevel,
    SourceLocation,
    ValidationResult,
    ValidationStatus,
)


VALID_CHECKSUM = "a" * 64


def build_regulatory_evidence(document_id):
 
    return Evidence(
        location=SourceLocation(
            document_id=document_id,
            page_number=4,
            section_title="Books and Records",
        ),
        text="The firm must retain the required records.",
        is_primary_evidence=True,
    )


def build_policy_evidence(document_id):
 
    return Evidence(
        location=SourceLocation(
            document_id=document_id,
            page_number=9,
            section_title="Retention Requirements",
        ),
        text="Records must be retained according to the approved schedule.",
        retrieval_score=0.84,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        is_primary_evidence=True,
    )


def build_requirement():
 
    regulation_id = uuid4()
    evidence = build_regulatory_evidence(regulation_id)

    return RegulatoryRequirement(
        document_id=regulation_id,
        requirement_text="The firm must retain required records.",
        normalized_requirement=(
            "Retain required records for the prescribed duration."
        ),
        requirement_type=RequirementType.RECORDKEEPING,
        modality=RequirementModality.MANDATORY,
        source_evidence=evidence,
        extraction_confidence=0.91,
        extraction_model="test-model",
        prompt_version="extract-v1",
    )


def build_confidence():
  
    return ConfidenceAssessment(
        score=0.80,
        components=ConfidenceComponents(
            retrieval_quality=0.84,
            regulatory_evidence_quality=0.95,
            policy_evidence_quality=0.75,
            validation_success=1.0,
            requirement_clarity=0.88,
        ),
        calculation_version="confidence-v1",
        explanation="Strong source evidence and relevant policy retrieval.",
    )


def build_risk():
  
    return RiskAssessment(
        score=65,
        level=RiskLevel.HIGH,
        components=RiskComponents(
            mandatory_requirement=True,
            missing_requirement=False,
            contradictory_policy=False,
            partial_coverage=True,
            regulatory_evidence_available=True,
            policy_evidence_available=True,
            business_impact_weight=3,
        ),
        calculation_version="risk-v1",
        explanation="A mandatory requirement appears partially covered.",
    )


def test_regulation_requires_issuing_authority():
 
    with pytest.raises(
        ValidationError,
        match="issuing_authority is required",
    ):
        DocumentMetadata(
            document_type=DocumentType.REGULATION,
            original_filename="rule.pdf",
            stored_filename="stored-rule.pdf",
            checksum_sha256=VALID_CHECKSUM,
            file_size_bytes=1000,
        )


def test_effective_date_cannot_precede_publication_date():
  
    with pytest.raises(
        ValidationError,
        match="effective_date cannot be earlier",
    ):
        DocumentMetadata(
            document_type=DocumentType.REGULATION,
            issuing_authority=IssuingAuthority.SEC,
            original_filename="rule.pdf",
            stored_filename="stored-rule.pdf",
            checksum_sha256=VALID_CHECKSUM,
            publication_date=date(2026, 4, 1),
            effective_date=date(2026, 3, 1),
            file_size_bytes=1000,
        )


def test_processed_document_requires_processed_timestamp():
  
    with pytest.raises(
        ValidationError,
        match="processed_at is required",
    ):
        DocumentMetadata(
            document_type=DocumentType.POLICY,
            original_filename="policy.pdf",
            stored_filename="stored-policy.pdf",
            checksum_sha256=VALID_CHECKSUM,
            file_size_bytes=1000,
            processing_status=ProcessingStatus.PROCESSED,
        )


def test_requirement_evidence_must_reference_same_document():
 
    requirement_document_id = uuid4()
    unrelated_document_id = uuid4()

    with pytest.raises(
        ValidationError,
        match="must match source evidence",
    ):
        RegulatoryRequirement(
            document_id=requirement_document_id,
            requirement_text="The firm must retain records.",
            normalized_requirement="Retain required records.",
            requirement_type=RequirementType.RECORDKEEPING,
            modality=RequirementModality.MANDATORY,
            source_evidence=build_regulatory_evidence(
                unrelated_document_id
            ),
            extraction_confidence=0.90,
            extraction_model="test-model",
            prompt_version="extract-v1",
        )


def test_covered_finding_requires_policy_evidence():
  
    with pytest.raises(
        ValidationError,
        match="covered requires policy evidence",
    ):
        Finding(
            analysis_id=uuid4(),
            requirement=build_requirement(),
            policy_evidence=[],
            classification=GapClassification.COVERED,
            rationale="The policy addresses the requirement.",
            confidence=build_confidence(),
            risk=build_risk(),
            validation=ValidationResult(
                status=ValidationStatus.PASSED,
                checks_run=["regulatory_evidence_present"],
                validated_at=datetime.now(timezone.utc),
            ),
            comparison_model="test-model",
            prompt_version="compare-v1",
        )


def test_valid_partially_covered_finding():
  
    policy_id = uuid4()

    finding = Finding(
        analysis_id=uuid4(),
        requirement=build_requirement(),
        policy_evidence=[build_policy_evidence(policy_id)],
        classification=GapClassification.PARTIALLY_COVERED,
        rationale=(
            "The policy requires retention but does not state the "
            "specific regulatory retention duration."
        ),
        confidence=build_confidence(),
        risk=build_risk(),
        validation=ValidationResult(
            status=ValidationStatus.PASSED,
            checks_run=[
                "regulatory_evidence_present",
                "policy_evidence_present",
                "rationale_present",
            ],
            validated_at=datetime.now(timezone.utc),
        ),
        comparison_model="test-model",
        prompt_version="compare-v1",
    )

    assert finding.classification == GapClassification.PARTIALLY_COVERED
    assert len(finding.policy_evidence) == 1
    assert finding.human_review.decision == ReviewDecision.PENDING


def test_completed_review_requires_reviewer_metadata():
 
    with pytest.raises(
        ValidationError,
        match="reviewer_id is required",
    ):
        HumanReview(
            decision=ReviewDecision.ACCEPT,
            reviewed_at=datetime.now(timezone.utc),
        )


def test_failed_validation_requires_failed_check():
  
    with pytest.raises(
        ValidationError,
        match="must contain at least one failed check",
    ):
        ValidationResult(
            status=ValidationStatus.FAILED,
            checks_run=["policy_evidence_present"],
            failed_checks=[],
        )