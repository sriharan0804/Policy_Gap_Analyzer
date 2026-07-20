"""Tests for deterministic internal policy extraction."""

from uuid import uuid4

from backend.models import (
    DocumentChunk,
    PolicyStatementType,
)
from backend.services.policy_extraction_service import (
    PolicyExtractor,
    RuleBasedPolicyExtractionService,
)


def make_chunk(
    *,
    text: str,
    page_number: int = 1,
    chunk_index: int = 0,
) -> DocumentChunk:
    """Create a valid chunk for policy extraction tests."""

    return DocumentChunk(
        document_id=uuid4(),
        page_number=page_number,
        chunk_index=chunk_index,
        text=text,
        character_count=len(text),
        start_character=0,
        end_character=len(text),
    )


def test_service_satisfies_policy_extractor_protocol():
    service = RuleBasedPolicyExtractionService()

    assert isinstance(service, PolicyExtractor)


def test_extracts_policy_control():
    chunk = make_chunk(
        text=(
            "Operations staff must verify all new "
            "customer account information."
        )
    )

    service = RuleBasedPolicyExtractionService()

    results = service.extract_from_chunk(chunk)

    assert len(results) == 1
    assert results[0].subject == "Operations staff"
    assert results[0].action == "verify"
    assert results[0].statement_type == (
        PolicyStatementType.CONTROL
    )


def test_extracts_policy_prohibition():
    chunk = make_chunk(
        text=(
            "Employees must not disclose confidential "
            "customer information."
        )
    )

    service = RuleBasedPolicyExtractionService()

    result = service.extract_from_chunk(chunk)[0]

    assert result.statement_type == (
        PolicyStatementType.PROHIBITION
    )
    assert result.matched_trigger == "must not"
    assert result.action == "disclose"


def test_classifies_record_retention_statement():
    chunk = make_chunk(
        text=(
            "The Records Department must retain customer "
            "account files for at least three years."
        )
    )

    service = RuleBasedPolicyExtractionService()

    result = service.extract_from_chunk(chunk)[0]

    assert result.statement_type == (
        PolicyStatementType.RECORD_RETENTION
    )
    assert result.timing == "for at least three years"


def test_classifies_review_statement():
    chunk = make_chunk(
        text=(
            "Compliance staff shall review active accounts annually."
        )
    )

    service = RuleBasedPolicyExtractionService()

    result = service.extract_from_chunk(chunk)[0]

    assert result.statement_type == (
        PolicyStatementType.REVIEW
    )
    assert result.timing == "annually"


def test_extracts_condition():
    chunk = make_chunk(
        text=(
            "Supervisors must investigate the account "
            "if suspicious activity is detected."
        )
    )

    service = RuleBasedPolicyExtractionService()

    result = service.extract_from_chunk(chunk)[0]

    assert result.condition == (
        "if suspicious activity is detected"
    )


def test_extracts_responsible_party():
    chunk = make_chunk(
        text=(
            "Customer records must be reviewed annually "
            "by Compliance."
        )
    )

    service = RuleBasedPolicyExtractionService()

    result = service.extract_from_chunk(chunk)[0]

    assert result.responsible_party == "Compliance"


def test_non_policy_statement_is_ignored():
    chunk = make_chunk(
        text=(
            "The company maintains offices in several cities."
        )
    )

    service = RuleBasedPolicyExtractionService()

    assert service.extract_from_chunk(chunk) == []


def test_extracts_multiple_policy_statements():
    chunk = make_chunk(
        text=(
            "Operations must verify customer identities. "
            "Compliance shall review exceptions quarterly."
        )
    )

    service = RuleBasedPolicyExtractionService()

    results = service.extract_from_chunk(chunk)

    assert len(results) == 2
    assert results[0].action == "verify"
    assert results[1].action == "review"
    assert results[1].timing == "quarterly"


def test_preserves_source_provenance():
    chunk = make_chunk(
        text="Compliance must review customer complaints.",
        page_number=9,
        chunk_index=13,
    )

    service = RuleBasedPolicyExtractionService()

    result = service.extract_from_chunk(chunk)[0]

    assert result.document_id == chunk.document_id
    assert result.chunk_id == chunk.chunk_id
    assert result.page_number == 9
    assert result.chunk_index == 13


def test_confidence_is_bounded():
    chunk = make_chunk(
        text="Compliance must review customer complaints."
    )

    service = RuleBasedPolicyExtractionService()

    result = service.extract_from_chunk(chunk)[0]

    assert 0.0 <= result.extraction_confidence <= 1.0