"""Tests for page-level PDF parsing."""

from pathlib import Path
from uuid import uuid4

import fitz
import pytest

from backend.exceptions import (
    CorruptedPDFError,
    PDFNotFoundError,
)
from backend.services.pdf_parser_service import PDFParserService


def create_pdf(
    file_path: Path,
    page_texts: list[str],
) -> None:
    """Create a real PDF for parser tests."""

    document = fitz.open()

    try:
        for page_text in page_texts:
            page = document.new_page()

            if page_text:
                page.insert_text(
                    (72, 72),
                    page_text,
                )

        document.save(file_path)

    finally:
        document.close()


def test_parse_pdf_preserves_page_numbers(tmp_path: Path):
    """Extracted text must remain linked to source pages."""

    pdf_path = tmp_path / "policy.pdf"

    create_pdf(
        pdf_path,
        [
            "Page one policy requirement.",
            "Page two review procedure.",
        ],
    )

    service = PDFParserService(minimum_text_characters=5)

    parsed = service.parse(
        document_id=uuid4(),
        file_path=pdf_path,
    )

    assert parsed.page_count == 2
    assert parsed.pages[0].page_number == 1
    assert parsed.pages[1].page_number == 2

    assert "Page one" in parsed.pages[0].text
    assert "Page two" in parsed.pages[1].text

    assert parsed.empty_page_count == 0
    assert parsed.requires_ocr is False


def test_empty_page_is_flagged_for_ocr(tmp_path: Path):
    """Image-only or blank pages should be flagged."""

    pdf_path = tmp_path / "blank.pdf"

    create_pdf(
        pdf_path,
        [""],
    )

    service = PDFParserService(minimum_text_characters=20)

    parsed = service.parse(
        document_id=uuid4(),
        file_path=pdf_path,
    )

    assert parsed.page_count == 1
    assert parsed.empty_page_count == 1
    assert parsed.pages[0].is_empty is True
    assert parsed.pages[0].may_require_ocr is True
    assert parsed.requires_ocr is True


def test_short_page_is_flagged_for_ocr(tmp_path: Path):
    """Very little extracted text may indicate a scanned page."""

    pdf_path = tmp_path / "short.pdf"

    create_pdf(
        pdf_path,
        ["Hi"],
    )

    service = PDFParserService(minimum_text_characters=20)

    parsed = service.parse(
        document_id=uuid4(),
        file_path=pdf_path,
    )

    assert parsed.pages[0].is_empty is False
    assert parsed.pages[0].may_require_ocr is True


def test_missing_pdf_is_rejected(tmp_path: Path):
    """Missing files must produce a domain-level error."""

    service = PDFParserService()

    with pytest.raises(
        PDFNotFoundError,
        match="not found",
    ):
        service.parse(
            document_id=uuid4(),
            file_path=tmp_path / "missing.pdf",
        )


def test_corrupted_pdf_is_rejected(tmp_path: Path):
    """Invalid PDF data must not reach downstream services."""

    pdf_path = tmp_path / "corrupted.pdf"
    pdf_path.write_bytes(b"%PDF-this is not a complete PDF")

    service = PDFParserService()

    with pytest.raises(
        CorruptedPDFError,
        match="could not be opened",
    ):
        service.parse(
            document_id=uuid4(),
            file_path=pdf_path,
        )


def test_normalize_text():
    """Whitespace should be cleaned consistently."""

    raw_text = "  Policy    title  \n" "\n" "\n" "\n" " Review\tprocedure "

    normalized = PDFParserService.normalize_text(raw_text)

    assert normalized == ("Policy title\n\nReview procedure")
