"""Deterministic comparison of regulatory requirements and policy statements."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from backend.models import (
    ComparisonComponents,
    GapAssessment,
    GapStatus,
    PolicyMatch,
    PolicyStatement,
    PolicyStatementType,
    RequirementCandidate,
    RequirementModality,
)


@dataclass(frozen=True)
class ComparisonThresholds:
    """Thresholds used to classify deterministic comparison scores."""

    fully_addressed: float = 0.82
    partially_addressed: float = 0.45
    minimum_evidence: float = 0.20

    def __post_init__(self) -> None:
        values = (
            self.fully_addressed,
            self.partially_addressed,
            self.minimum_evidence,
        )

        if any(value < 0.0 or value > 1.0 for value in values):
            raise ValueError("Comparison thresholds must be between 0.0 and 1.0.")

        if not (
            self.minimum_evidence <= self.partially_addressed <= self.fully_addressed
        ):
            raise ValueError(
                "Thresholds must satisfy minimum_evidence <= "
                "partially_addressed <= fully_addressed."
            )


@runtime_checkable
class GapComparator(Protocol):
    """Contract implemented by requirement-policy comparison engines."""

    def compare(
        self,
        requirement: RequirementCandidate,
        policy_statements: Sequence[PolicyStatement],
    ) -> GapAssessment:
        """Compare one requirement against available policy evidence."""


class DeterministicGapComparisonService:
    """Compare requirements and policies using explainable scoring rules."""

    _STOP_WORDS = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "of",
        "on",
        "or",
        "the",
        "to",
        "with",
        "must",
        "shall",
        "will",
        "may",
    }

    _ACTION_EQUIVALENTS: dict[str, set[str]] = {
        "retain": {
            "retain",
            "preserve",
            "keep",
            "maintain",
            "store",
        },
        "preserve": {
            "retain",
            "preserve",
            "keep",
            "maintain",
            "store",
        },
        "review": {
            "review",
            "inspect",
            "assess",
            "examine",
            "audit",
        },
        "notify": {
            "notify",
            "inform",
            "report",
            "communicate",
            "advise",
        },
        "verify": {
            "verify",
            "validate",
            "confirm",
            "authenticate",
        },
        "disclose": {
            "disclose",
            "share",
            "release",
            "provide",
        },
        "document": {
            "document",
            "record",
            "capture",
            "log",
        },
        "monitor": {
            "monitor",
            "supervise",
            "observe",
            "track",
        },
    }

    _PROHIBITION_TYPES = {
        PolicyStatementType.PROHIBITION,
    }

    def __init__(
        self,
        *,
        thresholds: ComparisonThresholds | None = None,
    ) -> None:
        self._thresholds = (
            thresholds if thresholds is not None else ComparisonThresholds()
        )

    def compare(
        self,
        requirement: RequirementCandidate,
        policy_statements: Sequence[PolicyStatement],
    ) -> GapAssessment:
        """Compare one regulatory requirement against policy evidence."""

        statements = list(policy_statements)

        if not statements:
            return GapAssessment(
                requirement_id=requirement.requirement_id,
                regulatory_document_id=requirement.document_id,
                regulatory_chunk_id=requirement.chunk_id,
                status=GapStatus.NOT_ADDRESSED,
                best_match=None,
                evaluated_policy_count=0,
                deterministic_score=0.0,
                rationale=[
                    "No internal policy statements were available " "for comparison."
                ],
                requires_human_review=True,
            )

        matches = [
            self._compare_statement(
                requirement=requirement,
                statement=statement,
            )
            for statement in statements
        ]

        contradiction_matches = [match for match in matches if match.is_contradiction]

        if contradiction_matches:
            best_contradiction = max(
                contradiction_matches,
                key=lambda match: match.components.overall_score,
            )

            return GapAssessment(
                requirement_id=requirement.requirement_id,
                regulatory_document_id=requirement.document_id,
                regulatory_chunk_id=requirement.chunk_id,
                status=GapStatus.CONTRADICTED,
                best_match=best_contradiction,
                evaluated_policy_count=len(statements),
                deterministic_score=(best_contradiction.components.overall_score),
                rationale=[
                    "The strongest related policy statement has an "
                    "opposing obligation or permission.",
                    *best_contradiction.reasons,
                ],
                requires_human_review=True,
            )

        best_match = max(
            matches,
            key=lambda match: match.components.overall_score,
        )

        status = self._classify_match(
            requirement=requirement,
            match=best_match,
        )

        rationale = [
            f"Best deterministic policy-match score: "
            f"{best_match.components.overall_score:.2f}.",
            *best_match.reasons,
        ]

        return GapAssessment(
            requirement_id=requirement.requirement_id,
            regulatory_document_id=requirement.document_id,
            regulatory_chunk_id=requirement.chunk_id,
            status=status,
            best_match=best_match,
            evaluated_policy_count=len(statements),
            deterministic_score=(best_match.components.overall_score),
            rationale=rationale,
            requires_human_review=self._requires_review(
                requirement=requirement,
                match=best_match,
                status=status,
            ),
        )

    def compare_many(
        self,
        requirements: Sequence[RequirementCandidate],
        policy_statements: Sequence[PolicyStatement],
    ) -> list[GapAssessment]:
        """Compare multiple requirements while preserving requirement order."""

        statements = list(policy_statements)

        return [
            self.compare(
                requirement=requirement,
                policy_statements=statements,
            )
            for requirement in requirements
        ]

    def _compare_statement(
        self,
        *,
        requirement: RequirementCandidate,
        statement: PolicyStatement,
    ) -> PolicyMatch:
        """Calculate deterministic compatibility with one policy statement."""

        action_score = self._calculate_action_score(
            requirement.action,
            statement.action,
        )

        object_score = self._calculate_text_overlap(
            requirement.object,
            statement.object,
        )

        timing_score = self._calculate_optional_field_score(
            requirement.timing,
            statement.timing,
        )

        condition_score = self._calculate_optional_field_score(
            requirement.condition,
            statement.condition,
        )

        modality_score = self._calculate_modality_score(
            requirement=requirement,
            statement=statement,
        )

        is_contradiction = self._is_contradiction(
            requirement=requirement,
            statement=statement,
            action_score=action_score,
            object_score=object_score,
        )

        overall_score = self._calculate_overall_score(
            action_score=action_score,
            object_score=object_score,
            timing_score=timing_score,
            condition_score=condition_score,
            modality_score=modality_score,
            requirement=requirement,
        )

        reasons = self._build_reasons(
            action_score=action_score,
            object_score=object_score,
            timing_score=timing_score,
            condition_score=condition_score,
            modality_score=modality_score,
            requirement=requirement,
            statement=statement,
            is_contradiction=is_contradiction,
        )

        return PolicyMatch(
            policy_statement_id=statement.statement_id,
            policy_document_id=statement.document_id,
            policy_chunk_id=statement.chunk_id,
            page_number=statement.page_number,
            chunk_index=statement.chunk_index,
            source_text=statement.source_text,
            components=ComparisonComponents(
                action_score=action_score,
                object_score=object_score,
                timing_score=timing_score,
                condition_score=condition_score,
                modality_score=modality_score,
                overall_score=overall_score,
            ),
            is_contradiction=is_contradiction,
            reasons=reasons,
        )

    def _classify_score(
        self,
        score: float,
    ) -> GapStatus:
        """Map an overall deterministic score to a gap status."""

        if score >= self._thresholds.fully_addressed:
            return GapStatus.FULLY_ADDRESSED

        if score >= self._thresholds.partially_addressed:
            return GapStatus.PARTIALLY_ADDRESSED

        if score >= self._thresholds.minimum_evidence:
            return GapStatus.INSUFFICIENT_EVIDENCE

        return GapStatus.NOT_ADDRESSED

    def _classify_match(
        self,
        *,
        requirement: RequirementCandidate,
        match: PolicyMatch,
    ) -> GapStatus:
        """Classify a match using both score and required-field coverage."""

        components = match.components

        missing_required_component = (
            requirement.timing is not None and components.timing_score == 0.0
        ) or (requirement.condition is not None and components.condition_score == 0.0)

        if missing_required_component:
            if components.action_score >= 0.85 and components.object_score >= 0.25:
                return GapStatus.PARTIALLY_ADDRESSED

            return GapStatus.INSUFFICIENT_EVIDENCE

        return self._classify_score(components.overall_score)

    def _calculate_action_score(
        self,
        requirement_action: str,
        policy_action: str,
    ) -> float:
        """Compare action verbs using exact and configured synonym matches."""

        left = self._normalize_token(requirement_action)
        right = self._normalize_token(policy_action)

        if not left or not right:
            return 0.0

        if left == right:
            return 1.0

        left_equivalents = self._ACTION_EQUIVALENTS.get(
            left,
            {left},
        )

        right_equivalents = self._ACTION_EQUIVALENTS.get(
            right,
            {right},
        )

        if right in left_equivalents or left in right_equivalents:
            return 0.9

        if left_equivalents.intersection(right_equivalents):
            return 0.85

        return 0.0

    def _calculate_text_overlap(
        self,
        left_text: str | None,
        right_text: str | None,
    ) -> float:
        """Calculate Jaccard token overlap between two text fields."""

        if not left_text or not right_text:
            return 0.0

        left_tokens = self._tokenize(left_text)
        right_tokens = self._tokenize(right_text)

        if not left_tokens or not right_tokens:
            return 0.0

        intersection = left_tokens.intersection(right_tokens)

        union = left_tokens.union(right_tokens)

        if not union:
            return 0.0

        return round(
            len(intersection) / len(union),
            4,
        )

    def _calculate_optional_field_score(
        self,
        requirement_value: str | None,
        policy_value: str | None,
    ) -> float:
        """Compare optional requirement fields fairly."""

        if requirement_value is None:
            return 1.0

        if policy_value is None:
            return 0.0

        normalized_requirement = self._normalize_phrase(requirement_value)

        normalized_policy = self._normalize_phrase(policy_value)

        if normalized_requirement == normalized_policy:
            return 1.0

        return self._calculate_text_overlap(
            requirement_value,
            policy_value,
        )

    @staticmethod
    def _calculate_modality_score(
        *,
        requirement: RequirementCandidate,
        statement: PolicyStatement,
    ) -> float:
        """Measure whether policy strength aligns with regulatory modality."""

        if requirement.modality == RequirementModality.PROHIBITED:
            if statement.statement_type == PolicyStatementType.PROHIBITION:
                return 1.0

            if statement.statement_type == PolicyStatementType.PERMISSION:
                return 0.0

            return 0.35

        if requirement.modality == RequirementModality.MANDATORY:
            if statement.statement_type in {
                PolicyStatementType.CONTROL,
                PolicyStatementType.RESPONSIBILITY,
                PolicyStatementType.REVIEW,
                PolicyStatementType.RECORD_RETENTION,
            }:
                return 1.0

            if statement.statement_type == PolicyStatementType.PERMISSION:
                return 0.25

            if statement.statement_type == PolicyStatementType.PROHIBITION:
                return 0.0

            return 0.4

        if requirement.modality == RequirementModality.PERMISSIVE:
            if statement.statement_type == PolicyStatementType.PERMISSION:
                return 1.0

            return 0.6

        if requirement.modality == RequirementModality.ADVISORY:
            return 0.8

        return 0.5

    def _is_contradiction(
        self,
        *,
        requirement: RequirementCandidate,
        statement: PolicyStatement,
        action_score: float,
        object_score: float,
    ) -> bool:
        """Detect explicit opposing treatment of a related activity."""

        is_related = action_score >= 0.85 and object_score >= 0.25

        if not is_related:
            return False

        if requirement.modality == RequirementModality.PROHIBITED:
            return statement.statement_type == PolicyStatementType.PERMISSION

        if requirement.modality == RequirementModality.MANDATORY:
            return statement.statement_type == PolicyStatementType.PROHIBITION

        return False

    @staticmethod
    def _calculate_overall_score(
        *,
        action_score: float,
        object_score: float,
        timing_score: float,
        condition_score: float,
        modality_score: float,
        requirement: RequirementCandidate,
    ) -> float:
        """Calculate a weighted requirement-policy match score."""

        weighted_items: list[tuple[float, float]] = [
            (action_score, 0.30),
            (object_score, 0.30),
            (modality_score, 0.20),
        ]

        if requirement.timing is not None:
            weighted_items.append((timing_score, 0.12))

        if requirement.condition is not None:
            weighted_items.append((condition_score, 0.08))

        total_weight = sum(weight for _, weight in weighted_items)

        score = (
            sum(component * weight for component, weight in weighted_items)
            / total_weight
        )

        return round(
            max(0.0, min(1.0, score)),
            4,
        )

    @staticmethod
    def _build_reasons(
        *,
        action_score: float,
        object_score: float,
        timing_score: float,
        condition_score: float,
        modality_score: float,
        requirement: RequirementCandidate,
        statement: PolicyStatement,
        is_contradiction: bool,
    ) -> list[str]:
        """Create human-readable explanations for deterministic scores."""

        reasons: list[str] = []

        if action_score >= 0.9:
            reasons.append("The policy action closely matches the regulatory action.")
        elif action_score == 0.0:
            reasons.append("The policy action does not match the regulatory action.")
        else:
            reasons.append("The policy action is only partially aligned.")

        if object_score >= 0.7:
            reasons.append("The policy covers substantially similar subject matter.")
        elif object_score >= 0.25:
            reasons.append(
                "The policy covers some, but not all, relevant subject matter."
            )
        else:
            reasons.append(
                "The policy provides little matching subject-matter evidence."
            )

        if requirement.timing is not None:
            if timing_score >= 0.9:
                reasons.append("The policy timing matches the regulatory timing.")
            elif statement.timing is None:
                reasons.append(
                    "The regulatory timing requirement is absent "
                    "from the policy statement."
                )
            else:
                reasons.append("The policy timing differs from the regulatory timing.")

        if requirement.condition is not None:
            if condition_score >= 0.9:
                reasons.append("The regulatory condition is represented in the policy.")
            elif statement.condition is None:
                reasons.append("The regulatory condition is absent from the policy.")
            else:
                reasons.append("The policy condition only partially matches.")

        if modality_score < 0.5:
            reasons.append(
                "The strength or direction of the policy statement "
                "does not align with the regulation."
            )

        if is_contradiction:
            reasons.append(
                "The policy explicitly conflicts with the regulatory modality."
            )

        return reasons

    @staticmethod
    def _requires_review(
        *,
        requirement: RequirementCandidate,
        match: PolicyMatch,
        status: GapStatus,
    ) -> bool:
        """Determine whether human validation is required."""

        if status != GapStatus.FULLY_ADDRESSED:
            return True

        components = match.components

        if requirement.timing is not None and components.timing_score < 1.0:
            return True

        if requirement.condition is not None and components.condition_score < 1.0:
            return True

        if components.overall_score < 0.90:
            return True

        return False

    @classmethod
    def _tokenize(
        cls,
        text: str,
    ) -> set[str]:
        """Normalize text into meaningful comparison tokens."""

        tokens = re.findall(
            r"[a-z0-9]+",
            text.lower(),
        )

        return {
            cls._singularize(token) for token in tokens if token not in cls._STOP_WORDS
        }

    @staticmethod
    def _singularize(
        token: str,
    ) -> str:
        """Apply conservative normalization for common plural forms."""

        if len(token) > 4 and token.endswith("ies"):
            return token[:-3] + "y"

        if len(token) > 4 and token.endswith("s"):
            return token[:-1]

        return token

    @staticmethod
    def _normalize_token(
        value: str,
    ) -> str:
        """Normalize an individual action token."""

        match = re.search(
            r"[a-z0-9]+",
            value.lower(),
        )

        return match.group(0) if match else ""

    @staticmethod
    def _normalize_phrase(
        value: str,
    ) -> str:
        """Normalize spaces and case for exact phrase comparisons."""

        return (
            re.sub(
                r"\s+",
                " ",
                value,
            )
            .strip()
            .lower()
        )
