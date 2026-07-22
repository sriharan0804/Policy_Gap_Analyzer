import re
from typing import Protocol, runtime_checkable

from backend.models import (
    GapAssessment,
    GapConfidenceAssessment,
    GapExplanation,
    GapRiskAssessment,
    GapStatus,
    RequirementCandidate,
)


@runtime_checkable
class ExplanationService(Protocol):
    def explain(
        self,
        requirement: RequirementCandidate,
        gap_assessment: GapAssessment,
        confidence_assessment: GapConfidenceAssessment,
        risk_assessment: GapRiskAssessment,
    ) -> GapExplanation:
        ...


class DeterministicExplanationService:
    """
    Create auditable, human-readable explanations without using an LLM.

    The service preserves deterministic behaviour while recognizing common
    regulatory-policy differences such as durations, review frequency,
    encryption scope, MFA scope and incident-reporting deadlines.
    """

    NUMBER_WORDS: dict[str, int] = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
        "twenty-four": 24,
        "seventy-two": 72,
    }

    def explain(
        self,
        requirement: RequirementCandidate,
        gap_assessment: GapAssessment,
        confidence_assessment: GapConfidenceAssessment,
        risk_assessment: GapRiskAssessment,
    ) -> GapExplanation:
        self._validate_requirement_ids(
            requirement=requirement,
            gap_assessment=gap_assessment,
            confidence_assessment=confidence_assessment,
            risk_assessment=risk_assessment,
        )

        requirement_summary = self._build_requirement_summary(
            requirement
        )

        policy_summary = self._build_policy_summary(
            gap_assessment
        )

        gap_reason, recommended_action = (
            self._build_specific_gap_explanation(
                requirement_text=requirement_summary,
                policy_text=policy_summary,
                gap_assessment=gap_assessment,
                risk_assessment=risk_assessment,
            )
        )

        return GapExplanation(
            requirement_id=requirement.requirement_id,
            requirement_summary=requirement_summary,
            policy_summary=policy_summary,
            gap_reason=gap_reason,
            confidence_reason=self._build_confidence_reason(
                confidence_assessment
            ),
            risk_reason=self._build_risk_reason(
                risk_assessment
            ),
            recommended_action=recommended_action,
            requires_human_review=self._requires_human_review(
                gap_assessment=gap_assessment,
                confidence_assessment=confidence_assessment,
                risk_assessment=risk_assessment,
            ),
        )

    @staticmethod
    def _validate_requirement_ids(
        *,
        requirement: RequirementCandidate,
        gap_assessment: GapAssessment,
        confidence_assessment: GapConfidenceAssessment,
        risk_assessment: GapRiskAssessment,
    ) -> None:
        expected_requirement_id = requirement.requirement_id

        assessment_requirement_ids = (
            gap_assessment.requirement_id,
            confidence_assessment.requirement_id,
            risk_assessment.requirement_id,
        )

        if any(
            requirement_id != expected_requirement_id
            for requirement_id in assessment_requirement_ids
        ):
            raise ValueError(
                "Requirement and assessment IDs must refer to the "
                "same requirement."
            )

    @staticmethod
    def _build_requirement_summary(
        requirement: RequirementCandidate,
    ) -> str:
        return requirement.source_text.strip()

    @staticmethod
    def _build_policy_summary(
        gap_assessment: GapAssessment,
    ) -> str:
        best_match = gap_assessment.best_match

        if best_match is None:
            return "No relevant policy evidence was identified."

        for attribute_name in (
            "source_text",
            "policy_text",
        ):
            value = getattr(
                best_match,
                attribute_name,
                None,
            )

            if isinstance(value, str) and value.strip():
                return value.strip()

        policy_statement = getattr(
            best_match,
            "policy_statement",
            None,
        )

        if policy_statement is not None:
            statement_text = getattr(
                policy_statement,
                "source_text",
                None,
            )

            if (
                isinstance(statement_text, str)
                and statement_text.strip()
            ):
                return statement_text.strip()

        return "Relevant policy evidence was identified."

    def _build_specific_gap_explanation(
        self,
        *,
        requirement_text: str,
        policy_text: str,
        gap_assessment: GapAssessment,
        risk_assessment: GapRiskAssessment,
    ) -> tuple[str, str]:
        requirement = self._normalize(requirement_text)
        policy = self._normalize(policy_text)

        status = gap_assessment.status

        if (
            gap_assessment.best_match is None
            or policy.startswith("no relevant policy evidence")
        ):
            return (
                (
                    "No relevant internal policy statement was found for "
                    "this regulatory requirement."
                ),
                (
                    "Create and approve a policy control that explicitly "
                    "addresses the required action, scope and timing."
                ),
            )

        retention_difference = self._compare_duration(
            requirement=requirement,
            policy=policy,
            unit="year",
        )

        if (
            retention_difference is not None
            and self._contains_any(
                requirement,
                {"retain", "retention", "records"},
            )
        ):
            required_years, policy_years = retention_difference

            return (
                (
                    "The regulation requires customer records to be "
                    f"retained for at least {required_years} years, while "
                    f"the policy specifies {policy_years} years."
                ),
                (
                    f"Increase the policy retention period from "
                    f"{policy_years} years to at least "
                    f"{required_years} years."
                ),
            )

        hour_difference = self._compare_duration(
            requirement=requirement,
            policy=policy,
            unit="hour",
        )

        if (
            hour_difference is not None
            and self._contains_any(
                requirement,
                {"incident", "report", "notify", "notification"},
            )
        ):
            required_hours, policy_hours = hour_difference

            return (
                (
                    "The regulation requires security incidents to be "
                    f"reported within {required_hours} hours, while the "
                    f"policy allows {policy_hours} hours."
                ),
                (
                    "Revise the incident-reporting deadline from "
                    f"{policy_hours} hours to no more than "
                    f"{required_hours} hours."
                ),
            )

        if self._is_six_month_vs_annual(
            requirement=requirement,
            policy=policy,
        ):
            return (
                (
                    "The regulation requires user access reviews every "
                    "six months, while the policy requires reviews only "
                    "annually."
                ),
                (
                    "Change the access-review frequency from annual to "
                    "at least once every six months."
                ),
            )

        if self._is_missing_encryption_at_rest(
            requirement=requirement,
            policy=policy,
        ):
            return (
                (
                    "The regulation requires sensitive customer data to "
                    "be encrypted both at rest and in transit, while the "
                    "policy explicitly covers only data in transit."
                ),
                (
                    "Expand the policy to explicitly require encryption "
                    "of sensitive customer data at rest as well as in "
                    "transit."
                ),
            )

        if self._is_restricted_mfa_scope(
            requirement=requirement,
            policy=policy,
        ):
            return (
                (
                    "The regulation requires multi-factor "
                    "authentication for all administrative accounts, "
                    "while the policy limits MFA to remote "
                    "administrative access."
                ),
                (
                    "Expand the MFA control to cover every "
                    "administrative account, including local and "
                    "non-remote access."
                ),
            )

        return (
            self._build_default_gap_reason(
                gap_assessment
            ),
            self._build_default_recommended_action(
                gap_assessment=gap_assessment,
                risk_assessment=risk_assessment,
            ),
        )

    @staticmethod
    def _is_six_month_vs_annual(
        *,
        requirement: str,
        policy: str,
    ) -> bool:
        requirement_has_six_months = bool(
            re.search(
                r"\b(?:six|6)\s*(?:\(\s*6\s*\))?\s*months?\b",
                requirement,
            )
        )

        policy_is_annual = bool(
            re.search(
                r"\b(?:annual|annually|yearly|once per year)\b",
                policy,
            )
        )

        return (
            requirement_has_six_months
            and policy_is_annual
            and "review" in requirement
            and "review" in policy
        )

    @staticmethod
    def _is_missing_encryption_at_rest(
        *,
        requirement: str,
        policy: str,
    ) -> bool:
        return (
            "at rest" in requirement
            and "in transit" in requirement
            and "in transit" in policy
            and "at rest" not in policy
        )

    @staticmethod
    def _is_restricted_mfa_scope(
        *,
        requirement: str,
        policy: str,
    ) -> bool:
        requirement_has_all_admins = bool(
            re.search(
                r"\ball\s+administrative\s+accounts?\b",
                requirement,
            )
        )

        policy_has_remote_scope = bool(
            re.search(
                r"\bremote\s+administrative\s+"
                r"(?:access|accounts?)\b",
                policy,
            )
        )

        return (
            requirement_has_all_admins
            and policy_has_remote_scope
        )

    def _compare_duration(
        self,
        *,
        requirement: str,
        policy: str,
        unit: str,
    ) -> tuple[int, int] | None:
        required_value = self._extract_number_near_unit(
            text=requirement,
            unit=unit,
        )

        policy_value = self._extract_number_near_unit(
            text=policy,
            unit=unit,
        )

        if required_value is None or policy_value is None:
            return None

        if required_value == policy_value:
            return None

        return required_value, policy_value

    def _extract_number_near_unit(
        self,
        *,
        text: str,
        unit: str,
    ) -> int | None:
        digit_before_word = re.search(
            rf"\b(\d+)\s*"
            rf"(?:\(\s*\d+\s*\))?\s*"
            rf"{re.escape(unit)}s?\b",
            text,
        )

        if digit_before_word:
            return int(digit_before_word.group(1))

        word_with_parenthesized_digit = re.search(
            rf"\b[a-z-]+\s*"
            rf"\(\s*(\d+)\s*\)\s*"
            rf"{re.escape(unit)}s?\b",
            text,
        )

        if word_with_parenthesized_digit:
            return int(
                word_with_parenthesized_digit.group(1)
            )

        for number_word, value in self.NUMBER_WORDS.items():
            if re.search(
                rf"\b{re.escape(number_word)}\s+"
                rf"{re.escape(unit)}s?\b",
                text,
            ):
                return value

        return None

    @staticmethod
    def _build_default_gap_reason(
        gap_assessment: GapAssessment,
    ) -> str:
        rationale = gap_assessment.rationale

        if isinstance(rationale, str) and rationale.strip():
            return rationale.strip()

        if isinstance(rationale, (list, tuple)):
            cleaned_reasons = [
                reason.strip()
                for reason in rationale
                if isinstance(reason, str) and reason.strip()
            ]

            if cleaned_reasons:
                return " ".join(cleaned_reasons)

        status = DeterministicExplanationService._enum_text(
            gap_assessment.status
        )

        return f"The requirement was assessed as {status}."

    @staticmethod
    def _build_confidence_reason(
        confidence_assessment: GapConfidenceAssessment,
    ) -> str:
        confidence_level = (
            DeterministicExplanationService._enum_text(
                confidence_assessment.confidence_level
            )
        )

        confidence_percentage = round(
            confidence_assessment.confidence_score * 100
        )

        reasons = [
            f"Confidence is {confidence_level} "
            f"({confidence_percentage}%)."
        ]

        if confidence_assessment.positive_factors:
            positive_factors = (
                DeterministicExplanationService._join_factors(
                    confidence_assessment.positive_factors
                )
            )

            if positive_factors:
                reasons.append(
                    f"Positive factors: {positive_factors}."
                )

        if confidence_assessment.limiting_factors:
            limiting_factors = (
                DeterministicExplanationService._join_factors(
                    confidence_assessment.limiting_factors
                )
            )

            if limiting_factors:
                reasons.append(
                    f"Limiting factors: {limiting_factors}."
                )

        return " ".join(reasons)

    @staticmethod
    def _build_risk_reason(
        risk_assessment: GapRiskAssessment,
    ) -> str:
        risk_level = DeterministicExplanationService._enum_text(
            risk_assessment.risk_level
        )

        risk_percentage = round(
            risk_assessment.risk_score * 100
        )

        reasons = [
            f"Risk is {risk_level} ({risk_percentage}%)."
        ]

        if risk_assessment.risk_factors:
            risk_factors = (
                DeterministicExplanationService._join_factors(
                    risk_assessment.risk_factors
                )
            )

            if risk_factors:
                reasons.append(
                    f"Risk factors: {risk_factors}."
                )

        if risk_assessment.mitigating_factors:
            mitigating_factors = (
                DeterministicExplanationService._join_factors(
                    risk_assessment.mitigating_factors
                )
            )

            if mitigating_factors:
                reasons.append(
                    f"Mitigating factors: {mitigating_factors}."
                )

        return " ".join(reasons)

    @staticmethod
    def _build_default_recommended_action(
        *,
        gap_assessment: GapAssessment,
        risk_assessment: GapRiskAssessment,
    ) -> str:
        priority = (
            DeterministicExplanationService._format_priority(
                risk_assessment.remediation_priority
            )
        )

        if gap_assessment.status == GapStatus.FULLY_ADDRESSED:
            return (
                "Retain the current policy evidence and validate the "
                "control during the next scheduled compliance review."
            )

        if gap_assessment.status == GapStatus.NOT_ADDRESSED:
            return (
                "Create a policy control that directly addresses the "
                f"requirement. {priority}"
            )

        if gap_assessment.status == GapStatus.CONTRADICTED:
            return (
                "Revise the conflicting policy language and obtain "
                f"compliance approval. {priority}"
            )

        if gap_assessment.status == GapStatus.INSUFFICIENT_EVIDENCE:
            return (
                "Collect additional policy evidence and complete a "
                f"human review. {priority}"
            )

        return (
            "Strengthen the relevant policy language so that the "
            f"requirement is fully addressed. {priority}"
        )

    @staticmethod
    def _format_priority(value: object) -> str:
        priority = DeterministicExplanationService._enum_text(
            value
        )

        if not priority:
            return "Plan and track remediation."

        if priority.startswith("remediate"):
            return priority[:1].upper() + priority[1:] + "."

        if priority.startswith("plan"):
            return priority[:1].upper() + priority[1:] + "."

        return (
            "Apply the following remediation priority: "
            f"{priority}."
        )

    @staticmethod
    def _requires_human_review(
        *,
        gap_assessment: GapAssessment,
        confidence_assessment: GapConfidenceAssessment,
        risk_assessment: GapRiskAssessment,
    ) -> bool:
        return any(
            (
                gap_assessment.requires_human_review,
                confidence_assessment.requires_human_review,
                risk_assessment.requires_human_review,
            )
        )

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(text.lower().split())

    @staticmethod
    def _contains_any(
        text: str,
        values: set[str],
    ) -> bool:
        return any(value in text for value in values)

    @staticmethod
    def _join_factors(
        factors: list[str],
    ) -> str:
        cleaned: list[str] = []

        for factor in factors:
            normalized = factor.strip().rstrip(".;")

            if normalized:
                cleaned.append(normalized)

        return "; ".join(cleaned)

    @staticmethod
    def _enum_text(value: object) -> str:
        raw_value = getattr(value, "value", value)

        return (
            str(raw_value)
            .replace("_", " ")
            .strip()
            .lower()
        )