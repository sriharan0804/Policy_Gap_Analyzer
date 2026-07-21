"""Deterministic extraction of structured internal policy statements."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from backend.exceptions import EmptyPolicyTextError
from backend.models import (
    DocumentChunk,
    PolicyStatement,
    PolicyStatementType,
)


@dataclass(frozen=True)
class PolicyTrigger:
    """A phrase that indicates an internal policy statement."""

    phrase: str
    statement_type: PolicyStatementType
    confidence: float


@runtime_checkable
class PolicyExtractor(Protocol):
    """Contract implemented by policy extraction providers."""

    def extract_from_chunk(
        self,
        chunk: DocumentChunk,
    ) -> list[PolicyStatement]:
        """Extract structured statements from one policy chunk."""


class RuleBasedPolicyExtractionService:
    """Extract explicit internal policy statements using deterministic rules."""

    _TRIGGERS: tuple[PolicyTrigger, ...] = (
        PolicyTrigger(
            phrase="must not",
            statement_type=PolicyStatementType.PROHIBITION,
            confidence=0.94,
        ),
        PolicyTrigger(
            phrase="shall not",
            statement_type=PolicyStatementType.PROHIBITION,
            confidence=0.94,
        ),
        PolicyTrigger(
            phrase="is prohibited from",
            statement_type=PolicyStatementType.PROHIBITION,
            confidence=0.93,
        ),
        PolicyTrigger(
            phrase="is responsible for",
            statement_type=PolicyStatementType.RESPONSIBILITY,
            confidence=0.92,
        ),
        PolicyTrigger(
            phrase="are responsible for",
            statement_type=PolicyStatementType.RESPONSIBILITY,
            confidence=0.92,
        ),
        PolicyTrigger(
            phrase="must",
            statement_type=PolicyStatementType.CONTROL,
            confidence=0.92,
        ),
        PolicyTrigger(
            phrase="shall",
            statement_type=PolicyStatementType.CONTROL,
            confidence=0.92,
        ),
        PolicyTrigger(
            phrase="will",
            statement_type=PolicyStatementType.CONTROL,
            confidence=0.84,
        ),
        PolicyTrigger(
            phrase="is required to",
            statement_type=PolicyStatementType.CONTROL,
            confidence=0.91,
        ),
        PolicyTrigger(
            phrase="are required to",
            statement_type=PolicyStatementType.CONTROL,
            confidence=0.91,
        ),
        PolicyTrigger(
            phrase="may",
            statement_type=PolicyStatementType.PERMISSION,
            confidence=0.65,
        ),
    )

    _NUMBER_PATTERN = (
        r"(?:\d+|"
        r"one|two|three|four|five|six|seven|eight|nine|ten|"
        r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|"
        r"seventeen|eighteen|nineteen|twenty|thirty|forty|"
        r"fifty|sixty|seventy|eighty|ninety)"
    )

    _TIMING_PATTERNS: tuple[re.Pattern[str], ...] = (
        re.compile(
            rf"\bfor\s+(?:at\s+least\s+)?"
            rf"{_NUMBER_PATTERN}\s+"
            rf"(?:day|days|week|weeks|month|months|year|years)\b",
            flags=re.IGNORECASE,
        ),
        re.compile(
            rf"\bwithin\s+{_NUMBER_PATTERN}\s+"
            rf"(?:business\s+)?"
            rf"(?:day|days|week|weeks|month|months|year|years)\b",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:daily|weekly|monthly|quarterly|annually|yearly)\b",
            flags=re.IGNORECASE,
        ),
    )

    _CONDITION_PATTERNS: tuple[re.Pattern[str], ...] = (
        re.compile(
            r"\bif\s+[^,.;]+",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"\bwhen\s+[^,.;]+",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"\bunless\s+[^,.;]+",
            flags=re.IGNORECASE,
        ),
    )

    _RESPONSIBLE_PARTY_PATTERNS: tuple[re.Pattern[str], ...] = (
        re.compile(
            r"\bby\s+(?:the\s+)?"
            r"([A-Z][A-Za-z&\-/ ]{2,50})"
            r"(?=[,.;]|\s+(?:daily|weekly|monthly|quarterly|annually)|$)"
        ),
        re.compile(
            r"\b(?:responsibility of|owned by)\s+"
            r"(?:the\s+)?"
            r"([A-Z][A-Za-z&\-/ ]{2,50})",
            flags=re.IGNORECASE,
        ),
    )

    def extract_from_chunk(
        self,
        chunk: DocumentChunk,
    ) -> list[PolicyStatement]:
        """Extract explicit policy statements from a document chunk."""

        text = self._normalize_text(chunk.text)

        if not text:
            raise EmptyPolicyTextError(
                "Cannot extract policy statements from empty text."
            )

        statements: list[PolicyStatement] = []

        for sentence in self._split_sentences(text):
            trigger = self._find_trigger(sentence)

            if trigger is None:
                continue

            statements.append(
                self._build_statement(
                    sentence=sentence,
                    trigger=trigger,
                    chunk=chunk,
                )
            )

        return statements

    def extract_from_chunks(
        self,
        chunks: list[DocumentChunk],
    ) -> list[PolicyStatement]:
        """Extract policy statements while preserving source order."""

        statements: list[PolicyStatement] = []

        for chunk in chunks:
            statements.extend(self.extract_from_chunk(chunk))

        return statements

    def _build_statement(
        self,
        *,
        sentence: str,
        trigger: PolicyTrigger,
        chunk: DocumentChunk,
    ) -> PolicyStatement:
        """Build one structured policy statement."""

        subject, remainder = self._split_around_trigger(
            sentence=sentence,
            trigger_phrase=trigger.phrase,
        )

        action, object_text = self._extract_action_and_object(remainder)

        timing = self._extract_first_match(
            text=sentence,
            patterns=self._TIMING_PATTERNS,
        )

        condition = self._extract_first_match(
            text=sentence,
            patterns=self._CONDITION_PATTERNS,
        )

        responsible_party = self._extract_responsible_party(sentence)

        statement_type = self._refine_statement_type(
            base_type=trigger.statement_type,
            sentence=sentence,
            timing=timing,
        )

        confidence = self._calculate_confidence(
            base_confidence=trigger.confidence,
            subject=subject,
            action=action,
            object_text=object_text,
            responsible_party=responsible_party,
        )

        return PolicyStatement(
            document_id=chunk.document_id,
            chunk_id=chunk.chunk_id,
            page_number=chunk.page_number,
            chunk_index=chunk.chunk_index,
            source_text=sentence,
            subject=subject,
            action=action,
            object=object_text,
            condition=condition,
            timing=timing,
            responsible_party=responsible_party,
            statement_type=statement_type,
            matched_trigger=trigger.phrase,
            extraction_confidence=confidence,
        )

    def _find_trigger(
        self,
        sentence: str,
    ) -> PolicyTrigger | None:
        """Return the strongest trigger found in a sentence."""

        lowered = sentence.lower()

        for trigger in self._TRIGGERS:
            if re.search(
                r"\b" + re.escape(trigger.phrase) + r"\b",
                lowered,
            ):
                return trigger

        return None

    @staticmethod
    def _split_around_trigger(
        *,
        sentence: str,
        trigger_phrase: str,
    ) -> tuple[str | None, str]:
        """Separate the subject from the controlled activity."""

        match = re.search(
            re.escape(trigger_phrase),
            sentence,
            flags=re.IGNORECASE,
        )

        if match is None:
            return None, sentence.strip()

        subject = sentence[: match.start()].strip(" ,;:")
        remainder = sentence[match.end() :].strip(" ,;:")

        return subject or None, remainder

    @staticmethod
    def _extract_action_and_object(
        remainder: str,
    ) -> tuple[str, str | None]:
        """Extract a leading action verb and remaining object text."""

        normalized = remainder.strip().rstrip(".")

        if not normalized:
            return "unspecified", None

        words = normalized.split()

        action = words[0].lower()
        object_text = " ".join(words[1:]).strip()

        return action, object_text or None

    @staticmethod
    def _extract_first_match(
        *,
        text: str,
        patterns: tuple[re.Pattern[str], ...],
    ) -> str | None:
        """Return the first matching timing or condition phrase."""

        for pattern in patterns:
            match = pattern.search(text)

            if match:
                return match.group(0).strip()

        return None

    def _extract_responsible_party(
        self,
        sentence: str,
    ) -> str | None:
        """Extract a named team or function where explicitly stated."""

        for pattern in self._RESPONSIBLE_PARTY_PATTERNS:
            match = pattern.search(sentence)

            if match:
                return match.group(1).strip()

        return None

    @staticmethod
    def _refine_statement_type(
        *,
        base_type: PolicyStatementType,
        sentence: str,
        timing: str | None,
    ) -> PolicyStatementType:
        """Refine the policy classification using deterministic signals."""

        lowered = sentence.lower()

        if "retain" in lowered or "preserve" in lowered or "retention" in lowered:
            return PolicyStatementType.RECORD_RETENTION

        if (
            "review" in lowered
            or "audit" in lowered
            or timing
            in {
                "daily",
                "weekly",
                "monthly",
                "quarterly",
                "annually",
                "yearly",
            }
        ):
            return PolicyStatementType.REVIEW

        return base_type

    @staticmethod
    def _calculate_confidence(
        *,
        base_confidence: float,
        subject: str | None,
        action: str,
        object_text: str | None,
        responsible_party: str | None,
    ) -> float:
        """Calculate bounded deterministic extraction confidence."""

        confidence = base_confidence

        if subject:
            confidence += 0.02
        else:
            confidence -= 0.07

        if action == "unspecified":
            confidence -= 0.15

        if object_text:
            confidence += 0.01
        else:
            confidence -= 0.05

        if responsible_party:
            confidence += 0.02

        return round(
            max(0.0, min(1.0, confidence)),
            4,
        )

    @staticmethod
    def _split_sentences(
        text: str,
    ) -> list[str]:
        """Split normalized policy text into sentence-like units."""

        return [
            part.strip()
            for part in re.split(
                r"(?<=[.!?;])\s+",
                text,
            )
            if part.strip()
        ]

    @staticmethod
    def _normalize_text(
        text: str,
    ) -> str:
        """Normalize whitespace while preserving wording."""

        if not isinstance(text, str):
            return ""

        return re.sub(
            r"\s+",
            " ",
            text,
        ).strip()
