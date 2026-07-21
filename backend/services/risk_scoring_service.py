"""Deterministic risk scoring for policy gap findings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from backend.models import (
    DataSensitivity,
    GapAssessment,
    GapConfidenceAssessment,
    GapRiskAssessment,
    GapRiskComponents,
    GapStatus,
    RegulatoryImpact,
    RequirementCandidate,
    RequirementModality,
    RiskLevel,
)


@dataclass(frozen=True)
class RiskWeights:
    gap_severity: float = 0.30
    regulatory_impact: float = 0.20
    requirement_criticality: float = 0.15
    data_sensitivity: float = 0.15
    confidence_reliability: float = 0.10
    contradiction: float = 0.10

    def __post_init__(self) -> None:
        values = (
            self.gap_severity,
            self.regulatory_impact,
            self.requirement_criticality,
            self.data_sensitivity,
            self.confidence_reliability,
            self.contradiction,
        )

        if any(value < 0.0 or value > 1.0 for value in values):
            raise ValueError("Risk weights must be between 0.0 and 1.0.")

        if abs(sum(values) - 1.0) > 1e-9:
            raise ValueError("Risk weights must sum to exactly 1.0.")


@dataclass(frozen=True)
class RiskThresholds:
    critical: float = 0.85
    high: float = 0.65
    medium: float = 0.40

    def __post_init__(self) -> None:
        if not 0.0 <= self.medium <= self.high <= self.critical <= 1.0:
            raise ValueError(
                "Thresholds must satisfy "
                "0.0 <= medium <= high <= critical <= 1.0."
            )


@runtime_checkable
class RiskScorer(Protocol):
    def score(
        self,
        requirement: RequirementCandidate,
        gap_assessment: GapAssessment,
        confidence_assessment: GapConfidenceAssessment,
        *,
        regulatory_impact: RegulatoryImpact,
        data_sensitivity: DataSensitivity,
    ) -> GapRiskAssessment:
        """Calculate deterministic business risk."""


class DeterministicRiskScoringService:
    def __init__(
        self,
        *,
        weights: RiskWeights | None = None,
        thresholds: RiskThresholds | None = None,
    ) -> None:
        self._weights = weights or RiskWeights()
        self._thresholds = thresholds or RiskThresholds()

    def score(
        self,
        requirement: RequirementCandidate,
        gap_assessment: GapAssessment,
        confidence_assessment: GapConfidenceAssessment,
        *,
        regulatory_impact: RegulatoryImpact,
        data_sensitivity: DataSensitivity,
    ) -> GapRiskAssessment:
        components = GapRiskComponents(
            gap_severity_score=self._gap_severity(gap_assessment.status),
            regulatory_impact_score=self._regulatory_impact_score(
                regulatory_impact
            ),
            requirement_criticality_score=self._requirement_criticality(
                requirement.modality
            ),
            data_sensitivity_score=self._data_sensitivity_score(
                data_sensitivity
            ),
            confidence_reliability_score=(
                confidence_assessment.confidence_score
            ),
            contradiction_score=(
                1.0
                if gap_assessment.status == GapStatus.CONTRADICTED
                else 0.0
            ),
        )

        risk_score = self._weighted_score(components)
        risk_level = self._classify(risk_score)

        risk_factors = self._build_risk_factors(
            requirement=requirement,
            gap_assessment=gap_assessment,
            regulatory_impact=regulatory_impact,
            data_sensitivity=data_sensitivity,
            confidence_assessment=confidence_assessment,
        )

        mitigating_factors = self._build_mitigating_factors(
            gap_assessment=gap_assessment,
            confidence_assessment=confidence_assessment,
        )

        return GapRiskAssessment(
            gap_assessment_id=gap_assessment.assessment_id,
            requirement_id=requirement.requirement_id,
            risk_score=risk_score,
            risk_level=risk_level,
            components=components,
            regulatory_impact=regulatory_impact,
            data_sensitivity=data_sensitivity,
            risk_factors=risk_factors,
            mitigating_factors=mitigating_factors,
            remediation_priority=self._priority(risk_level),
            requires_human_review=self._requires_review(
                risk_level=risk_level,
                gap_assessment=gap_assessment,
                confidence_assessment=confidence_assessment,
            ),
        )

    def _weighted_score(self, components: GapRiskComponents) -> float:
        score = (
            components.gap_severity_score * self._weights.gap_severity
            + components.regulatory_impact_score
            * self._weights.regulatory_impact
            + components.requirement_criticality_score
            * self._weights.requirement_criticality
            + components.data_sensitivity_score
            * self._weights.data_sensitivity
            + components.confidence_reliability_score
            * self._weights.confidence_reliability
            + components.contradiction_score
            * self._weights.contradiction
        )

        return round(max(0.0, min(1.0, score)), 4)

    @staticmethod
    def _gap_severity(status: GapStatus) -> float:
        values = {
            GapStatus.FULLY_ADDRESSED: 0.05,
            GapStatus.PARTIALLY_ADDRESSED: 0.55,
            GapStatus.NOT_ADDRESSED: 0.90,
            GapStatus.CONTRADICTED: 1.00,
            GapStatus.INSUFFICIENT_EVIDENCE: 0.65,
        }

        return values[status]

    @staticmethod
    def _regulatory_impact_score(
        impact: RegulatoryImpact,
    ) -> float:
        return {
            RegulatoryImpact.LOW: 0.20,
            RegulatoryImpact.MODERATE: 0.50,
            RegulatoryImpact.HIGH: 0.80,
            RegulatoryImpact.SEVERE: 1.00,
        }[impact]

    @staticmethod
    def _data_sensitivity_score(
        sensitivity: DataSensitivity,
    ) -> float:
        return {
            DataSensitivity.NONE: 0.00,
            DataSensitivity.INTERNAL: 0.25,
            DataSensitivity.CONFIDENTIAL: 0.55,
            DataSensitivity.PERSONAL: 0.80,
            DataSensitivity.HIGHLY_SENSITIVE: 1.00,
        }[sensitivity]

    @staticmethod
    def _requirement_criticality(
        modality: RequirementModality,
    ) -> float:
        value = str(modality.value).lower()

        if "prohibit" in value or "must not" in value or "shall not" in value:
            return 1.0

        if "mandatory" in value or "must" in value or "shall" in value:
            return 0.90

        if "recommended" in value or "should" in value:
            return 0.50

        return 0.25

    def _classify(self, score: float) -> RiskLevel:
        if score >= self._thresholds.critical:
            return RiskLevel.CRITICAL

        if score >= self._thresholds.high:
            return RiskLevel.HIGH

        if score >= self._thresholds.medium:
            return RiskLevel.MEDIUM

        return RiskLevel.LOW

    @staticmethod
    def _priority(level: RiskLevel) -> str:
        return {
            RiskLevel.CRITICAL: "Immediate remediation",
            RiskLevel.HIGH: "Remediate within 30 days",
            RiskLevel.MEDIUM: "Plan remediation",
            RiskLevel.LOW: "Monitor",
        }[level]

    @staticmethod
    def _build_risk_factors(
        *,
        requirement: RequirementCandidate,
        gap_assessment: GapAssessment,
        regulatory_impact: RegulatoryImpact,
        data_sensitivity: DataSensitivity,
        confidence_assessment: GapConfidenceAssessment,
    ) -> list[str]:
        factors: list[str] = []

        if gap_assessment.status == GapStatus.NOT_ADDRESSED:
            factors.append("The policy does not address the requirement.")

        if gap_assessment.status == GapStatus.CONTRADICTED:
            factors.append("The policy contradicts the regulatory requirement.")

        if gap_assessment.status == GapStatus.PARTIALLY_ADDRESSED:
            factors.append("The requirement is only partially addressed.")

        if regulatory_impact in {
            RegulatoryImpact.HIGH,
            RegulatoryImpact.SEVERE,
        }:
            factors.append("The regulatory impact is high.")

        if data_sensitivity in {
            DataSensitivity.PERSONAL,
            DataSensitivity.HIGHLY_SENSITIVE,
        }:
            factors.append("Sensitive data is involved.")

        if confidence_assessment.confidence_score >= 0.80:
            factors.append("The gap determination has high confidence.")

        if requirement.timing is not None:
            factors.append("The requirement contains a mandatory timing element.")

        if requirement.condition is not None:
            factors.append("The requirement applies under a specific condition.")

        return factors

    @staticmethod
    def _build_mitigating_factors(
        *,
        gap_assessment: GapAssessment,
        confidence_assessment: GapConfidenceAssessment,
    ) -> list[str]:
        factors: list[str] = []

        if gap_assessment.status == GapStatus.FULLY_ADDRESSED:
            factors.append("The policy fully addresses the requirement.")

        if confidence_assessment.confidence_score < 0.55:
            factors.append(
                "The finding has limited confidence and requires validation."
            )

        return factors

    @staticmethod
    def _requires_review(
        *,
        risk_level: RiskLevel,
        gap_assessment: GapAssessment,
        confidence_assessment: GapConfidenceAssessment,
    ) -> bool:
        if risk_level in {RiskLevel.HIGH, RiskLevel.CRITICAL}:
            return True

        if gap_assessment.status in {
            GapStatus.CONTRADICTED,
            GapStatus.INSUFFICIENT_EVIDENCE,
        }:
            return True

        if confidence_assessment.requires_human_review:
            return True

        return False