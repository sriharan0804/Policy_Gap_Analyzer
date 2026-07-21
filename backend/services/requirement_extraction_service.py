"""Deterministic extraction of regulatory requirement candidates."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from backend.exceptions import EmptyRequirementTextError
from backend.models import (
    DocumentChunk,
    RequirementCandidate,
    RequirementModality,
)


@dataclass(frozen=True)
class TriggerDefinition:
    """Mapping between regulatory language and requirement modality."""

    phrase: str
    modality: RequirementModality
    confidence: float


@runtime_checkable
class RequirementExtractor(Protocol):
    """Contract for regulatory requirement extraction providers."""

    def extract_from_chunk(
        self,
        chunk: DocumentChunk,
    ) -> list[RequirementCandidate]:
        """Extract structured requirements from one document chunk."""


class RuleBasedRequirementExtractionService:
    """Extract regulatory requirements using deterministic language rules.

    This implementation is intentionally conservative. It identifies explicit
    obligation language but does not attempt deep legal interpretation.
    """

    _TRIGGERS: tuple[TriggerDefinition, ...] = (
        TriggerDefinition(
            phrase="must not",
            modality=RequirementModality.PROHIBITED,
            confidence=0.95,
        ),
        TriggerDefinition(
            phrase="shall not",
            modality=RequirementModality.PROHIBITED,
            confidence=0.95,
        ),
        TriggerDefinition(
            phrase="may not",
            modality=RequirementModality.PROHIBITED,
            confidence=0.88,
        ),
        TriggerDefinition(
            phrase="is prohibited from",
            modality=RequirementModality.PROHIBITED,
            confidence=0.94,
        ),
        TriggerDefinition(
            phrase="must",
            modality=RequirementModality.MANDATORY,
            confidence=0.95,
        ),
        TriggerDefinition(
            phrase="shall",
            modality=RequirementModality.MANDATORY,
            confidence=0.95,
        ),
        TriggerDefinition(
            phrase="is required to",
            modality=RequirementModality.MANDATORY,
            confidence=0.93,
        ),
        TriggerDefinition(
            phrase="are required to",
            modality=RequirementModality.MANDATORY,
            confidence=0.93,
        ),
        TriggerDefinition(
            phrase="should",
            modality=RequirementModality.ADVISORY,
            confidence=0.70,
        ),
        TriggerDefinition(
            phrase="may",
            modality=RequirementModality.PERMISSIVE,
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
            rf"\b(?:within|no later than)\s+"
            rf"{_NUMBER_PATTERN}\s+"
            rf"(?:business\s+)?"
            rf"(?:day|days|week|weeks|month|months|year|years)\b",
            flags=re.IGNORECASE,
        ),
        re.compile(
            rf"\bfor\s+(?:at\s+least\s+)?"
            rf"{_NUMBER_PATTERN}\s+"
            rf"(?:day|days|week|weeks|month|months|year|years)\b",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:daily|weekly|monthly|quarterly|annually|yearly)\b",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:before|after|upon)\s+" r"(?:the\s+)?" r"[a-z][a-z\s-]{2,50}",
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
        re.compile(
            r"\bin the event that\s+[^,.;]+",
            flags=re.IGNORECASE,
        ),
    )

    def extract_from_chunk(
        self,
        chunk: DocumentChunk,
    ) -> list[RequirementCandidate]:
        """Extract explicit regulatory requirements from a document chunk."""

        text = self._normalize_text(chunk.text)

        if not text:
            raise EmptyRequirementTextError(
                "Cannot extract requirements from empty text."
            )

        sentences = self._split_sentences(text)
        candidates: list[RequirementCandidate] = []

        for sentence in sentences:
            trigger = self._find_trigger(sentence)

            if trigger is None:
                continue

            candidate = self._build_candidate(
                sentence=sentence,
                trigger=trigger,
                chunk=chunk,
            )

            candidates.append(candidate)

        return candidates

    def extract_from_chunks(
        self,
        chunks: list[DocumentChunk],
    ) -> list[RequirementCandidate]:
        """Extract requirements from multiple chunks in source order."""

        candidates: list[RequirementCandidate] = []

        for chunk in chunks:
            candidates.extend(self.extract_from_chunk(chunk))

        return candidates

    def _build_candidate(
        self,
        *,
        sentence: str,
        trigger: TriggerDefinition,
        chunk: DocumentChunk,
    ) -> RequirementCandidate:
        """Convert one triggered sentence into a structured candidate."""

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

        confidence = self._calculate_confidence(
            base_confidence=trigger.confidence,
            subject=subject,
            action=action,
            object_text=object_text,
            sentence=sentence,
        )

        return RequirementCandidate(
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
            modality=trigger.modality,
            matched_trigger=trigger.phrase,
            extraction_confidence=confidence,
        )

    def _find_trigger(
        self,
        sentence: str,
    ) -> TriggerDefinition | None:
        """Return the strongest matching trigger."""

        lowered_sentence = sentence.lower()

        for trigger in self._TRIGGERS:
            pattern = r"\b" + re.escape(trigger.phrase) + r"\b"

            if re.search(
                pattern,
                lowered_sentence,
            ):
                return trigger

        return None

    @staticmethod
    def _split_around_trigger(
        *,
        sentence: str,
        trigger_phrase: str,
    ) -> tuple[str | None, str]:
        """Split a requirement into responsible party and obligation text."""

        match = re.search(
            re.escape(trigger_phrase),
            sentence,
            flags=re.IGNORECASE,
        )

        if match is None:
            return None, sentence.strip()

        subject = sentence[: match.start()].strip(" ,;:")

        remainder = sentence[match.end() :].strip(" ,;:")

        return (
            subject or None,
            remainder,
        )

    @staticmethod
    def _extract_action_and_object(
        remainder: str,
    ) -> tuple[str, str | None]:
        """Extract the leading action verb and remaining object text."""

        normalized = remainder.strip().rstrip(".")

        if not normalized:
            return "unspecified", None

        words = normalized.split()

        action = words[0].lower()
        object_text = " ".join(words[1:]).strip()

        return (
            action,
            object_text or None,
        )

    @staticmethod
    def _extract_first_match(
        *,
        text: str,
        patterns: tuple[re.Pattern[str], ...],
    ) -> str | None:
        """Return the first matched phrase from a pattern collection."""

        for pattern in patterns:
            match = pattern.search(text)

            if match:
                return match.group(0).strip()

        return None

    @staticmethod
    def _calculate_confidence(
        *,
        base_confidence: float,
        subject: str | None,
        action: str,
        object_text: str | None,
        sentence: str,
    ) -> float:
        """Calculate deterministic extraction confidence."""

        confidence = base_confidence

        if subject:
            confidence += 0.02
        else:
            confidence -= 0.08

        if action == "unspecified":
            confidence -= 0.15

        if object_text:
            confidence += 0.01
        else:
            confidence -= 0.05

        if len(sentence.split()) < 4:
            confidence -= 0.10

        return round(
            max(
                0.0,
                min(1.0, confidence),
            ),
            4,
        )

    @staticmethod
    def _split_sentences(
        text: str,
    ) -> list[str]:
        """Split normalized text into sentence-like units."""

        parts = re.split(
            r"(?<=[.!?;])\s+",
            text,
        )

        return [part.strip() for part in parts if part.strip()]

    @staticmethod
    def _normalize_text(
        text: str,
    ) -> str:
        """Normalize whitespace without changing semantic content."""

        if not isinstance(text, str):
            return ""

        return re.sub(
            r"\s+",
            " ",
            text,
        ).strip()
