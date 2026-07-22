from unittest.mock import Mock
from uuid import uuid4

from backend.models import (
    DataSensitivity,
    DocumentChunk,
    GapAssessment,
    GapConfidenceAssessment,
    GapExplanation,
    GapRiskAssessment,
    PolicyStatement,
    PolicyStatementType,
    RegulatoryImpact,
    RequirementCandidate,
    RequirementModality,
)
from backend.services.analysis_orchestration_service import (
    AnalysisOrchestrationService,
    AnalysisOrchestrator,
)


def build_service(
    *,
    requirement_extractor=None,
    policy_extractor=None,
    gap_comparer=None,
    confidence_scorer=None,
    risk_scorer=None,
    explanation_service=None,
) -> AnalysisOrchestrationService:
    return AnalysisOrchestrationService(
        requirement_extractor=requirement_extractor or Mock(),
        policy_extractor=policy_extractor or Mock(),
        gap_comparer=gap_comparer or Mock(),
        confidence_scorer=confidence_scorer or Mock(),
        risk_scorer=risk_scorer or Mock(),
        explanation_service=explanation_service or Mock(),
    )


def build_chunk(
    *,
    document_id=None,
    chunk_index: int = 0,
    text: str = "Sample text",
) -> DocumentChunk:
    return DocumentChunk(
        document_id=document_id or uuid4(),
        chunk_index=chunk_index,
        page_number=1,
        text=text,
        character_count=len(text),
        start_character=0,
        end_character=len(text),
    )


def test_service_satisfies_orchestration_protocol():
    service = build_service()

    assert isinstance(service, AnalysisOrchestrator)


def test_service_stores_injected_dependencies():
    requirement_extractor = Mock()
    policy_extractor = Mock()
    gap_comparer = Mock()
    confidence_scorer = Mock()
    risk_scorer = Mock()
    explanation_service = Mock()

    service = AnalysisOrchestrationService(
        requirement_extractor=requirement_extractor,
        policy_extractor=policy_extractor,
        gap_comparer=gap_comparer,
        confidence_scorer=confidence_scorer,
        risk_scorer=risk_scorer,
        explanation_service=explanation_service,
    )

    assert service._requirement_extractor is requirement_extractor
    assert service._policy_extractor is policy_extractor
    assert service._gap_comparer is gap_comparer
    assert service._confidence_scorer is confidence_scorer
    assert service._risk_scorer is risk_scorer
    assert service._explanation_service is explanation_service


def test_analyze_extracts_requirements_and_policy_statements():
    regulatory_document_id = uuid4()
    policy_document_id = uuid4()

    regulatory_chunk = build_chunk(
        document_id=regulatory_document_id,
        text="The organization must retain records.",
    )

    policy_chunk = build_chunk(
        document_id=policy_document_id,
        text="The company retains records.",
    )

    requirement = RequirementCandidate(
        requirement_id=uuid4(),
        document_id=regulatory_document_id,
        chunk_id=regulatory_chunk.chunk_id,
        page_number=regulatory_chunk.page_number,
        chunk_index=regulatory_chunk.chunk_index,
        source_text=regulatory_chunk.text,
        modality=RequirementModality.MANDATORY,
        matched_trigger="must",
        subject="organization",
        action="retain",
        object="records",
        condition=None,
        timing=None,
        extraction_confidence=0.90,
    )

    policy_statement = PolicyStatement(
        statement_id=uuid4(),
        document_id=policy_document_id,
        chunk_id=policy_chunk.chunk_id,
        page_number=policy_chunk.page_number,
        chunk_index=policy_chunk.chunk_index,
        source_text=policy_chunk.text,
        statement_type=PolicyStatementType.CONTROL,
        matched_trigger="retains",
        subject="company",
        action="retains",
        object="records",
        condition=None,
        timing=None,
        extraction_confidence=0.90,
    )

    requirement_extractor = Mock()
    requirement_extractor.extract.return_value = [requirement]

    policy_extractor = Mock()
    policy_extractor.extract.return_value = [policy_statement]

    gap_comparer = Mock()
    gap_comparer.compare_many.return_value = []

    confidence_scorer = Mock()
    risk_scorer = Mock()
    explanation_service = Mock()

    service = build_service(
        requirement_extractor=requirement_extractor,
        policy_extractor=policy_extractor,
        gap_comparer=gap_comparer,
        confidence_scorer=confidence_scorer,
        risk_scorer=risk_scorer,
        explanation_service=explanation_service,
    )

    result = service.analyze(
        regulatory_document_id=regulatory_document_id,
        policy_document_id=policy_document_id,
        regulatory_chunks=[regulatory_chunk],
        policy_chunks=[policy_chunk],
        regulatory_impact=RegulatoryImpact.HIGH,
        data_sensitivity=DataSensitivity.CONFIDENTIAL,
    )

    assert result.requirements == [requirement]
    assert result.policy_statements == [policy_statement]
    assert result.gap_assessments == []
    assert result.confidence_assessments == []
    assert result.risk_assessments == []
    assert result.explanations == []

    requirement_extractor.extract.assert_called_once_with(regulatory_chunk)
    policy_extractor.extract.assert_called_once_with(policy_chunk)

    gap_comparer.compare_many.assert_called_once_with(
        [requirement],
        [policy_statement],
    )

    confidence_scorer.score.assert_not_called()
    risk_scorer.score.assert_not_called()
    explanation_service.explain.assert_not_called()


def test_analyze_handles_empty_chunk_lists():
    requirement_extractor = Mock()
    policy_extractor = Mock()

    gap_comparer = Mock()
    gap_comparer.compare_many.return_value = []

    confidence_scorer = Mock()
    risk_scorer = Mock()
    explanation_service = Mock()

    service = build_service(
        requirement_extractor=requirement_extractor,
        policy_extractor=policy_extractor,
        gap_comparer=gap_comparer,
        confidence_scorer=confidence_scorer,
        risk_scorer=risk_scorer,
        explanation_service=explanation_service,
    )

    result = service.analyze(
        regulatory_document_id=uuid4(),
        policy_document_id=uuid4(),
        regulatory_chunks=[],
        policy_chunks=[],
        regulatory_impact=RegulatoryImpact.HIGH,
        data_sensitivity=DataSensitivity.CONFIDENTIAL,
    )

    assert result.requirements == []
    assert result.policy_statements == []
    assert result.gap_assessments == []
    assert result.confidence_assessments == []
    assert result.risk_assessments == []
    assert result.explanations == []

    requirement_extractor.extract.assert_not_called()
    policy_extractor.extract.assert_not_called()

    gap_comparer.compare_many.assert_called_once_with([], [])
    confidence_scorer.score.assert_not_called()
    risk_scorer.score.assert_not_called()
    explanation_service.explain.assert_not_called()


def test_analyze_scores_and_explains_each_gap_assessment():
    regulatory_document_id = uuid4()
    policy_document_id = uuid4()

    regulatory_chunk = build_chunk(
        document_id=regulatory_document_id,
        text="The organization must retain records.",
    )

    policy_chunk = build_chunk(
        document_id=policy_document_id,
        text="The company retains records.",
    )

    requirement = RequirementCandidate(
        requirement_id=uuid4(),
        document_id=regulatory_document_id,
        chunk_id=regulatory_chunk.chunk_id,
        page_number=regulatory_chunk.page_number,
        chunk_index=regulatory_chunk.chunk_index,
        source_text=regulatory_chunk.text,
        modality=RequirementModality.MANDATORY,
        matched_trigger="must",
        subject="organization",
        action="retain",
        object="records",
        condition=None,
        timing=None,
        extraction_confidence=0.90,
    )

    policy_statement = PolicyStatement(
        statement_id=uuid4(),
        document_id=policy_document_id,
        chunk_id=policy_chunk.chunk_id,
        page_number=policy_chunk.page_number,
        chunk_index=policy_chunk.chunk_index,
        source_text=policy_chunk.text,
        statement_type=PolicyStatementType.CONTROL,
        matched_trigger="retains",
        subject="company",
        action="retains",
        object="records",
        condition=None,
        timing=None,
        extraction_confidence=0.90,
    )

    gap_assessment = GapAssessment.model_construct(
        requirement_id=requirement.requirement_id,
    )

    confidence_assessment = GapConfidenceAssessment.model_construct(
        requirement_id=requirement.requirement_id,
    )

    risk_assessment = GapRiskAssessment.model_construct(
        requirement_id=requirement.requirement_id,
    )

    explanation = GapExplanation(
        requirement_id=requirement.requirement_id,
        requirement_summary=requirement.source_text,
        policy_summary=policy_statement.source_text,
        gap_reason="The policy does not fully cover the requirement.",
        confidence_reason="Confidence is high.",
        risk_reason="Risk is high.",
        recommended_action="Review and strengthen the policy.",
        requires_human_review=True,
    )

    requirement_extractor = Mock()
    requirement_extractor.extract.return_value = [requirement]

    policy_extractor = Mock()
    policy_extractor.extract.return_value = [policy_statement]

    gap_comparer = Mock()
    gap_comparer.compare_many.return_value = [gap_assessment]

    confidence_scorer = Mock()
    confidence_scorer.score.return_value = confidence_assessment

    risk_scorer = Mock()
    risk_scorer.score.return_value = risk_assessment

    explanation_service = Mock()
    explanation_service.explain.return_value = explanation

    service = build_service(
        requirement_extractor=requirement_extractor,
        policy_extractor=policy_extractor,
        gap_comparer=gap_comparer,
        confidence_scorer=confidence_scorer,
        risk_scorer=risk_scorer,
        explanation_service=explanation_service,
    )

    result = service.analyze(
        regulatory_document_id=regulatory_document_id,
        policy_document_id=policy_document_id,
        regulatory_chunks=[regulatory_chunk],
        policy_chunks=[policy_chunk],
        regulatory_impact=RegulatoryImpact.HIGH,
        data_sensitivity=DataSensitivity.CONFIDENTIAL,
    )

    assert result.gap_assessments == [gap_assessment]
    assert result.confidence_assessments == [confidence_assessment]
    assert result.risk_assessments == [risk_assessment]
    assert result.explanations == [explanation]

    confidence_scorer.score.assert_called_once_with(
        requirement,
        gap_assessment,
        [policy_statement],
    )

    risk_scorer.score.assert_called_once_with(
        requirement,
        gap_assessment,
        confidence_assessment,
        regulatory_impact=RegulatoryImpact.HIGH,
        data_sensitivity=DataSensitivity.CONFIDENTIAL,
    )

    explanation_service.explain.assert_called_once_with(
        requirement,
        gap_assessment,
        confidence_assessment,
        risk_assessment,
    )