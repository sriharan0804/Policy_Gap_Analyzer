from typing import Any, Protocol, runtime_checkable
from uuid import UUID

from backend.models import (
    AnalysisResult,
    DataSensitivity,
    DocumentChunk,
    RegulatoryImpact,
)


@runtime_checkable
class AnalysisOrchestrator(Protocol):
    def analyze(
        self,
        *,
        regulatory_document_id: UUID,
        policy_document_id: UUID,
        regulatory_chunks: list[DocumentChunk],
        policy_chunks: list[DocumentChunk],
        regulatory_impact: RegulatoryImpact,
        data_sensitivity: DataSensitivity,
    ) -> AnalysisResult:
        ...


class AnalysisOrchestrationService:
    """Coordinates the complete deterministic policy gap analysis pipeline."""

    def __init__(
        self,
        *,
        requirement_extractor: Any,
        policy_extractor: Any,
        gap_comparer: Any,
        confidence_scorer: Any,
        risk_scorer: Any,
        explanation_service: Any,
    ) -> None:
        self._requirement_extractor = requirement_extractor
        self._policy_extractor = policy_extractor
        self._gap_comparer = gap_comparer
        self._confidence_scorer = confidence_scorer
        self._risk_scorer = risk_scorer
        self._explanation_service = explanation_service

    def analyze(
        self,
        *,
        regulatory_document_id: UUID,
        policy_document_id: UUID,
        regulatory_chunks: list[DocumentChunk],
        policy_chunks: list[DocumentChunk],
        regulatory_impact: RegulatoryImpact,
        data_sensitivity: DataSensitivity,
    ) -> AnalysisResult:
        requirements = []

        for chunk in regulatory_chunks:
            extracted_requirements = self._requirement_extractor.extract(chunk)
            requirements.extend(extracted_requirements)

        policy_statements = []

        for chunk in policy_chunks:
            extracted_statements = self._policy_extractor.extract(chunk)
            policy_statements.extend(extracted_statements)

        gap_assessments = self._gap_comparer.compare_many(
            requirements,
            policy_statements,
        )

        requirements_by_id = {
            requirement.requirement_id: requirement
            for requirement in requirements
        }

        confidence_assessments = []

        for gap_assessment in gap_assessments:
            requirement = requirements_by_id[
                gap_assessment.requirement_id
            ]

            confidence_assessment = self._confidence_scorer.score(
                requirement,
                gap_assessment,
                policy_statements,
            )

            confidence_assessments.append(confidence_assessment)

        confidence_by_requirement_id = {
            assessment.requirement_id: assessment
            for assessment in confidence_assessments
        }

        risk_assessments = []

        for gap_assessment in gap_assessments:
            requirement = requirements_by_id[
                gap_assessment.requirement_id
            ]

            confidence_assessment = confidence_by_requirement_id[
                gap_assessment.requirement_id
            ]

            risk_assessment = self._risk_scorer.score(
                requirement,
                gap_assessment,
                confidence_assessment,
                regulatory_impact=regulatory_impact,
                data_sensitivity=data_sensitivity,
            )

            risk_assessments.append(risk_assessment)

        risk_by_requirement_id = {
            assessment.requirement_id: assessment
            for assessment in risk_assessments
        }

        explanations = []

        for gap_assessment in gap_assessments:
            requirement = requirements_by_id[
                gap_assessment.requirement_id
            ]

            confidence_assessment = confidence_by_requirement_id[
                gap_assessment.requirement_id
            ]

            risk_assessment = risk_by_requirement_id[
                gap_assessment.requirement_id
            ]

            explanation = self._explanation_service.explain(
                requirement,
                gap_assessment,
                confidence_assessment,
                risk_assessment,
            )

            explanations.append(explanation)

        return AnalysisResult(
            regulatory_document_id=regulatory_document_id,
            policy_document_id=policy_document_id,
            requirements=requirements,
            policy_statements=policy_statements,
            gap_assessments=gap_assessments,
            confidence_assessments=confidence_assessments,
            risk_assessments=risk_assessments,
            explanations=explanations,
        )

