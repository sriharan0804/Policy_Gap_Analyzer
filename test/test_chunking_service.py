"""Tests for deterministic document chunking."""

from uuid import uuid4

import pytest

from backend.models import (
    ParsedDocument,
    ParsedPage,
)
from backend.services.chunking_service import ChunkingService


def make_page(
    *,
    page_number: int,
    text: str,
) -> ParsedPage:
    return ParsedPage(
        page_number=page_number,
        text=text,
        character_count=len(text),
        is_empty=len(text) == 0,
        may_require_ocr=len(text) < 20,
    )


def test_short_page_creates_one_chunk():
    document_id = uuid4()

    parsed = ParsedDocument(
        document_id=document_id,
        page_count=1,
        pages=[
            make_page(
                page_number=1,
                text="The firm must retain customer records.",
            )
        ],
        extracted_character_count=38,
        empty_page_count=0,
        requires_ocr=False,
    )

    service = ChunkingService(
        chunk_size=100,
        chunk_overlap=20,
    )

    chunks = service.chunk_document(parsed)

    assert len(chunks) == 1
    assert chunks[0].document_id == document_id
    assert chunks[0].page_number == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].text == (
        "The firm must retain customer records."
    )


def test_long_page_creates_overlapping_chunks():
    text = " ".join(
        f"requirement-{index}"
        for index in range(100)
    )

    page = make_page(
        page_number=3,
        text=text,
    )

    parsed = ParsedDocument(
        document_id=uuid4(),
        page_count=1,
        pages=[page],
        extracted_character_count=len(text),
        empty_page_count=0,
        requires_ocr=False,
    )

    service = ChunkingService(
        chunk_size=120,
        chunk_overlap=25,
    )

    chunks = service.chunk_document(parsed)

    assert len(chunks) > 1
    assert all(chunk.page_number == 3 for chunk in chunks)

    for previous, current in zip(
        chunks,
        chunks[1:],
    ):
        assert current.start_character < previous.end_character


def test_chunk_indexes_continue_across_pages():
    page_one_text = "A" * 150
    page_two_text = "B" * 150

    parsed = ParsedDocument(
        document_id=uuid4(),
        page_count=2,
        pages=[
            make_page(
                page_number=1,
                text=page_one_text,
            ),
            make_page(
                page_number=2,
                text=page_two_text,
            ),
        ],
        extracted_character_count=300,
        empty_page_count=0,
        requires_ocr=False,
    )

    service = ChunkingService(
        chunk_size=100,
        chunk_overlap=20,
    )

    chunks = service.chunk_document(parsed)

    assert [
        chunk.chunk_index for chunk in chunks
    ] == list(range(len(chunks)))


def test_empty_pages_are_skipped():
    parsed = ParsedDocument(
        document_id=uuid4(),
        page_count=2,
        pages=[
            make_page(
                page_number=1,
                text="",
            ),
            make_page(
                page_number=2,
                text="Policy review is required annually.",
            ),
        ],
        extracted_character_count=35,
        empty_page_count=1,
        requires_ocr=True,
    )

    service = ChunkingService()

    chunks = service.chunk_document(parsed)

    assert len(chunks) == 1
    assert chunks[0].page_number == 2


def test_invalid_chunk_configuration():
    with pytest.raises(
        ValueError,
        match="smaller than",
    ):
        ChunkingService(
            chunk_size=100,
            chunk_overlap=100,
        )