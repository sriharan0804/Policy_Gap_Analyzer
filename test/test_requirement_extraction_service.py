"""Tests for deterministic regulatory requirement extraction."""

from uuid import uuid4

import pytest

from backend.exceptions import EmptyRequirementTextError
from backend.models import (
    DocumentChunk,
    RequirementModality,
)
from backend.services.requirement_extraction_service import (
    RequirementExtractor,
    RuleBasedRequirementExtractionService,
)


def make_chunk(
    *,
    text: str,
    page_number: int = 1,
    chunk_index: int = 0,
) -> DocumentChunk:
    """Create a valid chunk for extraction tests."""

    return DocumentChunk(
        document_id=uuid4(),
        page_number=page_number,
        chunk_index=chunk_index,
        text=text,
        character_count=len(text),
        start_character=0,
        end_character=len(text),
    )


def test_service_satisfies_extractor_protocol():
    """The rule-based service should satisfy the extractor contract."""

    service = RuleBasedRequirementExtractionService()

    assert isinstance(
        service,
        RequirementExtractor,
    )


def test_extracts_mandatory_requirement():
    """Explicit 'must' language should produce a mandatory requirement."""

    chunk = make_chunk(
        text=(
            "Each broker-dealer must preserve customer "
            "account records for at least six years."
        ),
        page_number=4,
        chunk_index=7,
    )

    service = RuleBasedRequirementExtractionService()

    results = service.extract_from_chunk(chunk)

    assert len(results) == 1

    requirement = results[0]

    assert requirement.modality == (
        RequirementModality.MANDATORY
    )

    assert requirement.subject == (
        "Each broker-dealer"
    )

    assert requirement.action == "preserve"

    assert requirement.object == (
        "customer account records for at least six years"
    )

    assert requirement.timing == (
        "for at least six years"
    )

    assert requirement.page_number == 4
    assert requirement.chunk_index == 7
    assert requirement.chunk_id == chunk.chunk_id


def test_extracts_prohibition():
    """'Must not' should be classified as prohibited, not mandatory."""

    chunk = make_chunk(
        text=(
            "A registered representative must not disclose "
            "confidential customer information."
        )
    )

    service = RuleBasedRequirementExtractionService()

    results = service.extract_from_chunk(chunk)

    assert len(results) == 1
    assert results[0].modality == (
        RequirementModality.PROHIBITED
    )
    assert results[0].matched_trigger == "must not"
    assert results[0].action == "disclose"


def test_extracts_required_to_phrase():
    """Multi-word obligation triggers should be recognized."""

    chunk = make_chunk(
        text=(
            "Member firms are required to maintain "
            "written supervisory procedures."
        )
    )

    service = RuleBasedRequirementExtractionService()

    results = service.extract_from_chunk(chunk)

    assert len(results) == 1
    assert results[0].modality == (
        RequirementModality.MANDATORY
    )
    assert results[0].matched_trigger == (
        "are required to"
    )
    assert results[0].action == "maintain"


def test_extracts_condition():
    """Conditional requirement phrases should be preserved."""

    chunk = make_chunk(
        text=(
            "The firm must notify the customer "
            "if account information changes."
        )
    )

    service = RuleBasedRequirementExtractionService()

    results = service.extract_from_chunk(chunk)

    assert len(results) == 1
    assert results[0].condition == (
        "if account information changes"
    )


def test_extracts_timing():
    """Explicit deadlines should be captured."""

    chunk = make_chunk(
        text=(
            "The compliance department shall submit the report "
            "within 30 business days."
        )
    )

    service = RuleBasedRequirementExtractionService()

    results = service.extract_from_chunk(chunk)

    assert len(results) == 1
    assert results[0].timing == (
        "within 30 business days"
    )


def test_non_requirement_sentence_returns_empty_list():
    """Descriptive statements should not become obligations."""

    chunk = make_chunk(
        text=(
            "The organization was founded in 1998 "
            "and operates in several regions."
        )
    )

    service = RuleBasedRequirementExtractionService()

    results = service.extract_from_chunk(chunk)

    assert results == []


def test_extracts_multiple_requirements():
    """Multiple obligation sentences should produce multiple candidates."""

    chunk = make_chunk(
        text=(
            "The firm must retain customer records. "
            "Supervisors shall review the records annually."
        )
    )

    service = RuleBasedRequirementExtractionService()

    results = service.extract_from_chunk(chunk)

    assert len(results) == 2
    assert results[0].action == "retain"
    assert results[1].action == "review"
    assert results[1].timing == "annually"


def test_prohibition_trigger_has_priority():
    """Longer negative triggers must be checked before shorter triggers."""

    chunk = make_chunk(
        text=(
            "Employees shall not alter approved records."
        )
    )

    service = RuleBasedRequirementExtractionService()

    results = service.extract_from_chunk(chunk)

    assert len(results) == 1
    assert results[0].modality == (
        RequirementModality.PROHIBITED
    )
    assert results[0].matched_trigger == (
        "shall not"
    )


def test_normal_text_is_accepted():
    """A valid chunk should be processed without extraction errors."""

    chunk = make_chunk(
        text="The firm must maintain records."
    )

    service = RuleBasedRequirementExtractionService()

    results = service.extract_from_chunk(
        chunk
    )

    assert len(results) == 1


def test_extract_from_chunks_preserves_order():
    """Batch extraction should preserve document and chunk order."""

    chunks = [
        make_chunk(
            text="The firm must maintain records.",
            chunk_index=0,
        ),
        make_chunk(
            text="Supervisors shall conduct annual reviews.",
            chunk_index=1,
        ),
    ]

    service = RuleBasedRequirementExtractionService()

    results = service.extract_from_chunks(
        chunks
    )

    assert len(results) == 2
    assert results[0].chunk_index == 0
    assert results[1].chunk_index == 1


def test_confidence_remains_in_valid_range():
    """Extraction confidence should always remain between zero and one."""

    chunk = make_chunk(
        text="The firm must maintain records."
    )

    service = RuleBasedRequirementExtractionService()

    result = service.extract_from_chunk(
        chunk
    )[0]

    assert 0.0 <= result.extraction_confidence <= 1.0

def test_extracts_word_based_duration():
    """Durations written as words should be captured."""

    chunk = make_chunk(
        text=(
            "The firm must retain account records "
            "for at least six years."
        )
    )

    service = RuleBasedRequirementExtractionService()

    results = service.extract_from_chunk(chunk)

    assert len(results) == 1
    assert results[0].timing == (
        "for at least six years"
    )