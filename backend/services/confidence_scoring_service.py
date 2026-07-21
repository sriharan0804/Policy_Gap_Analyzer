"""Deterministic confidence scoring for policy gap assessments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable
from uuid import UUID

from backend.exceptions import InvalidRetrievalScoreError
from backend.models import (
    GapConfidenceComponents,
    GapAssessment,
    GapConfidenceAssessment,
    GapConfidenceLevel,
    GapStatus,
    PolicyStatement,
    RequirementCandidate,
)


@dataclass(frozen=True)
class ConfidenceWeights:
    """Weights used to calculate the final confidence score."""

    requirement_extraction: float = 0.15
    policy_extraction: float = 0.15
    retrieval: float = 0.15
    comparison: float = 0.25
    evidence_completeness: float = 0.20
    evidence_quantity: float = 0.10

    def __post_init__(self) -> None:
        values = (
            self.requirement_extraction,
            self.policy_extraction,
            self.retrieval,
            self.comparison,
            self.evidence_completeness,
            self.evidence_quantity,
        )

        if any(value < 0.0 or value > 1.0 for value in values):
            raise ValueError(
                "Confidence weights must be between 0.0 and 1.0."
            )

        total = sum(values)

        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                "Confidence weights must sum to exactly 1.0."
            )


@dataclass(frozen=True)
class ConfidenceThresholds:
    """Thresholds used to classify confidence levels."""

    high: float = 0.80
    medium: float = 0.55

    def __post_init__(self) -> None:
        if not 0.0 <= self.medium <= self.high <= 1.0:
            raise ValueError(
                "Thresholds must satisfy "
                "0.0 <= medium <= high <= 1.0."
            )


@runtime_checkable
class ConfidenceScorer(Protocol):
    """Contract for gap confidence scoring services."""

    def score(
        self,
        requirement: RequirementCandidate,
        gap_assessment: GapAssessment,
        policy_statements: Sequence[PolicyStatement],
        retrieval_scores: Mapping[UUID, float] | None = None,
    ) -> GapConfidenceAssessment:
        """Calculate confidence for one gap assessment."""


class DeterministicConfidenceScoringService:
    """Calculate explainable confidence from deterministic signals."""

    def __init__(
        self,
        *,
        weights: ConfidenceWeights | None = None,
        thresholds: ConfidenceThresholds | None = None,
    ) -> None:
        self._weights = (
            weights
            if weights is not None
            else ConfidenceWeights()
        )

        self._thresholds = (
            thresholds
            if thresholds is not None
            else ConfidenceThresholds()
        )

    def score(
        self,
        requirement: RequirementCandidate,
        gap_assessment: GapAssessment,
        policy_statements: Sequence[PolicyStatement],
        retrieval_scores: Mapping[UUID, float] | None = None,
    ) -> GapConfidenceAssessment:
        """Calculate confidence for a deterministic gap decision."""

        statements = list(policy_statements)
        retrieval_map = dict(retrieval_scores or {})

        self._validate_retrieval_scores(retrieval_map)

        matching_statement = self._find_best_match_statement(
            gap_assessment=gap_assessment,
            policy_statements=statements,
        )

        requirement_extraction_score = self._clamp(
            requirement.extraction_confidence
        )

        policy_extraction_score = (
            self._clamp(
                matching_statement.extraction_confidence
            )
            if matching_statement is not None
            else 0.0
        )

        retrieval_score = self._calculate_retrieval_score(
            gap_assessment=gap_assessment,
            retrieval_scores=retrieval_map,
        )

        comparison_score = self._calculate_comparison_score(
            gap_assessment
        )

        evidence_completeness_score = (
            self._calculate_evidence_completeness(
                requirement=requirement,
                gap_assessment=gap_assessment,
            )
        )

        evidence_quantity_score = (
            self._calculate_evidence_quantity(
                gap_assessment.evaluated_policy_count
            )
        )

        components = GapConfidenceComponents(
            requirement_extraction_score=(
                requirement_extraction_score
            ),
            policy_extraction_score=(
                policy_extraction_score
            ),
            retrieval_score=retrieval_score,
            comparison_score=comparison_score,
            evidence_completeness_score=(
                evidence_completeness_score
            ),
            evidence_quantity_score=(
                evidence_quantity_score
            ),
        )

        confidence_score = self._calculate_weighted_score(
            components
        )

        confidence_level = self._classify_confidence(
            confidence_score
        )

        positive_factors = self._build_positive_factors(
            components=components,
            gap_assessment=gap_assessment,
        )

        limiting_factors = self._build_limiting_factors(
            components=components,
            requirement=requirement,
            gap_assessment=gap_assessment,
            matching_statement=matching_statement,
            retrieval_scores_available=bool(
                retrieval_map
            ),
        )

        requires_human_review = (
            self._requires_human_review(
                confidence_level=confidence_level,
                gap_assessment=gap_assessment,
                components=components,
            )
        )

        return GapConfidenceAssessment(
            gap_assessment_id=(
                gap_assessment.assessment_id
            ),
            requirement_id=requirement.requirement_id,
            confidence_score=confidence_score,
            confidence_level=confidence_level,
            components=components,
            supporting_evidence_count=(
                gap_assessment.evaluated_policy_count
            ),
            positive_factors=positive_factors,
            limiting_factors=limiting_factors,
            requires_human_review=requires_human_review,
        )

    def score_many(
        self,
        requirements: Sequence[RequirementCandidate],
        gap_assessments: Sequence[GapAssessment],
        policy_statements: Sequence[PolicyStatement],
        retrieval_scores: Mapping[UUID, float] | None = None,
    ) -> list[GapConfidenceAssessment]:
        """Score multiple gap assessments by requirement ID."""

        assessment_by_requirement = {
            assessment.requirement_id: assessment
            for assessment in gap_assessments
        }

        results: list[GapConfidenceAssessment] = []

        for requirement in requirements:
            assessment = assessment_by_requirement.get(
                requirement.requirement_id
            )

            if assessment is None:
                continue

            results.append(
                self.score(
                    requirement=requirement,
                    gap_assessment=assessment,
                    policy_statements=policy_statements,
                    retrieval_scores=retrieval_scores,
                )
            )

        return results

    def _calculate_weighted_score(
        self,
        components: GapConfidenceComponents,
    ) -> float:
        """Calculate weighted confidence score."""

        score = (
            components.requirement_extraction_score
            * self._weights.requirement_extraction
            +
            components.policy_extraction_score
            * self._weights.policy_extraction
            +
            components.retrieval_score
            * self._weights.retrieval
            +
            components.comparison_score
            * self._weights.comparison
            +
            components.evidence_completeness_score
            * self._weights.evidence_completeness
            +
            components.evidence_quantity_score
            * self._weights.evidence_quantity
        )

        return round(
            self._clamp(score),
            4,
        )

    def _calculate_retrieval_score(
        self,
        *,
        gap_assessment: GapAssessment,
        retrieval_scores: Mapping[UUID, float],
    ) -> float:
        """Return retrieval relevance for the selected policy evidence."""

        if gap_assessment.best_match is None:
            return 0.0

        chunk_id = (
            gap_assessment.best_match.policy_chunk_id
        )

        score = retrieval_scores.get(chunk_id)

        if score is None:
            return 0.5

        return self._clamp(score)

    @staticmethod
    def _calculate_comparison_score(
        gap_assessment: GapAssessment,
    ) -> float:
        """Convert gap comparison evidence into a confidence signal."""

        if gap_assessment.status == GapStatus.CONTRADICTED:
            if gap_assessment.best_match is None:
                return 0.0

            return gap_assessment.deterministic_score

        if gap_assessment.status == GapStatus.NOT_ADDRESSED:
            if gap_assessment.evaluated_policy_count == 0:
                return 0.85

            return 1.0 - gap_assessment.deterministic_score

        if (
            gap_assessment.status
            == GapStatus.INSUFFICIENT_EVIDENCE
        ):
            return 0.35

        return gap_assessment.deterministic_score

    @staticmethod
    def _calculate_evidence_completeness(
        *,
        requirement: RequirementCandidate,
        gap_assessment: GapAssessment,
    ) -> float:
        """Measure whether required comparison dimensions are present."""

        match = gap_assessment.best_match

        if match is None:
            return (
                0.8
                if gap_assessment.evaluated_policy_count == 0
                else 0.2
            )

        scores = [
            match.components.action_score,
            match.components.object_score,
            match.components.modality_score,
        ]

        if requirement.timing is not None:
            scores.append(
                match.components.timing_score
            )

        if requirement.condition is not None:
            scores.append(
                match.components.condition_score
            )

        if not scores:
            return 0.0

        return round(
            sum(scores) / len(scores),
            4,
        )

    @staticmethod
    def _calculate_evidence_quantity(
        evidence_count: int,
    ) -> float:
        """Normalize policy evidence quantity without rewarding excess."""

        if evidence_count <= 0:
            return 0.0

        if evidence_count == 1:
            return 0.5

        if evidence_count == 2:
            return 0.75

        return 1.0

    @staticmethod
    def _find_best_match_statement(
        *,
        gap_assessment: GapAssessment,
        policy_statements: Sequence[PolicyStatement],
    ) -> PolicyStatement | None:
        """Find the policy statement selected by the gap engine."""

        if gap_assessment.best_match is None:
            return None

        selected_id = (
            gap_assessment.best_match.policy_statement_id
        )

        return next(
            (
                statement
                for statement in policy_statements
                if statement.statement_id == selected_id
            ),
            None,
        )

    def _classify_confidence(
        self,
        score: float,
    ) -> GapConfidenceLevel:
        """Map numerical confidence to a human-readable level."""

        if score >= self._thresholds.high:
            return GapConfidenceLevel.HIGH

        if score >= self._thresholds.medium:
            return GapConfidenceLevel.MEDIUM

        return GapConfidenceLevel.LOW

    @staticmethod
    def _build_positive_factors(
        *,
        components: GapConfidenceComponents,
        gap_assessment: GapAssessment,
    ) -> list[str]:
        """Build explainable positive confidence factors."""

        factors: list[str] = []

        if (
            components.requirement_extraction_score
            >= 0.80
        ):
            factors.append(
                "The regulatory requirement was extracted "
                "with high confidence."
            )

        if components.policy_extraction_score >= 0.80:
            factors.append(
                "The selected policy statement was extracted "
                "with high confidence."
            )

        if components.retrieval_score >= 0.75:
            factors.append(
                "The selected policy evidence had strong "
                "retrieval relevance."
            )

        if components.comparison_score >= 0.80:
            factors.append(
                "The deterministic comparison produced a "
                "strong decision signal."
            )

        if components.evidence_completeness_score >= 0.80:
            factors.append(
                "Most required comparison dimensions were present."
            )

        if gap_assessment.evaluated_policy_count >= 2:
            factors.append(
                "Multiple policy statements were evaluated."
            )

        return factors

    @staticmethod
    def _build_limiting_factors(
        *,
        components: GapConfidenceComponents,
        requirement: RequirementCandidate,
        gap_assessment: GapAssessment,
        matching_statement: PolicyStatement | None,
        retrieval_scores_available: bool,
    ) -> list[str]:
        """Build explainable limitations affecting confidence."""

        factors: list[str] = []

        if (
            components.requirement_extraction_score
            < 0.60
        ):
            factors.append(
                "The regulatory requirement extraction "
                "confidence is low."
            )

        if matching_statement is None:
            factors.append(
                "No selected policy statement was available "
                "for extraction-confidence validation."
            )

        elif components.policy_extraction_score < 0.60:
            factors.append(
                "The selected policy statement extraction "
                "confidence is low."
            )

        if not retrieval_scores_available:
            factors.append(
                "No retrieval similarity scores were supplied; "
                "a neutral retrieval score was used."
            )

        elif components.retrieval_score < 0.50:
            factors.append(
                "The selected policy evidence had weak "
                "retrieval relevance."
            )

        if (
            requirement.timing is not None
            and gap_assessment.best_match is not None
            and (
                gap_assessment
                .best_match
                .components
                .timing_score
                < 1.0
            )
        ):
            factors.append(
                "The regulatory timing requirement was not "
                "fully matched."
            )

        if (
            requirement.condition is not None
            and gap_assessment.best_match is not None
            and (
                gap_assessment
                .best_match
                .components
                .condition_score
                < 1.0
            )
        ):
            factors.append(
                "The regulatory condition was not fully matched."
            )

        if (
            components.evidence_completeness_score
            < 0.50
        ):
            factors.append(
                "The available policy evidence is incomplete."
            )

        if gap_assessment.evaluated_policy_count == 0:
            factors.append(
                "No policy statements were available for comparison."
            )

        if (
            gap_assessment.status
            == GapStatus.INSUFFICIENT_EVIDENCE
        ):
            factors.append(
                "The gap engine classified the available "
                "evidence as insufficient."
            )

        return factors

    @staticmethod
    def _requires_human_review(
        *,
        confidence_level: GapConfidenceLevel,
        gap_assessment: GapAssessment,
        components: GapConfidenceComponents,
    ) -> bool:
        """Determine whether human review remains necessary."""

        if gap_assessment.requires_human_review:
            return True

        if confidence_level != GapConfidenceLevel.HIGH:
            return True

        if components.evidence_completeness_score < 0.90:
            return True

        if (
            gap_assessment.status
            in {
                GapStatus.CONTRADICTED,
                GapStatus.INSUFFICIENT_EVIDENCE,
                GapStatus.PARTIALLY_ADDRESSED,
            }
        ):
            return True

        return False

    @staticmethod
    def _validate_retrieval_scores(
        retrieval_scores: Mapping[UUID, float],
    ) -> None:
        """Validate externally supplied retrieval scores."""

        for chunk_id, score in retrieval_scores.items():
            if isinstance(score, bool) or not isinstance(
                score,
                (int, float),
            ):
                raise InvalidRetrievalScoreError(
                    f"Retrieval score for chunk {chunk_id} "
                    "must be numeric."
                )

            if score < 0.0 or score > 1.0:
                raise InvalidRetrievalScoreError(
                    f"Retrieval score for chunk {chunk_id} "
                    "must be between 0.0 and 1.0."
                )

    @staticmethod
    def _clamp(value: float) -> float:
        """Clamp floating-point values to the supported range."""

        return max(
            0.0,
            min(1.0, float(value)),
        )