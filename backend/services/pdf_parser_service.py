from __future__ import annotations

import re
from pathlib import Path
from uuid import UUID

import fitz

from backend.exceptions import (
    CorruptedPDFError,
    EncryptedPDFError,
    PDFNotFoundError,
)
from backend.models import ParsedDocument, ParsedPage

MULTIPLE_SPACES_PATTERN = re.compile(r"[ \t]+")
EXCESSIVE_NEWLINES_PATTERN = re.compile(r"\n{3,}")


class PDFParserService:
    """Extract traceable page-level text from stored PDFs."""

    def __init__(self, minimum_text_characters: int = 20) -> None:
        """Initialize OCR-detection threshold.

        Pages with fewer than this number of extracted characters are
        considered likely candidates for OCR.
        """

        if minimum_text_characters < 0:
            raise ValueError("minimum_text_characters cannot be negative.")

        self._minimum_text_characters = minimum_text_characters

    def parse(
        self,
        *,
        document_id: UUID,
        file_path: Path,
    ) -> ParsedDocument:
        """Open and extract text from one PDF.

        Args:
            document_id:
                Domain identifier of the registered document.

            file_path:
                Location of the stored PDF.

        Returns:
            A structured page-level parsed document.

        Raises:
            PDFNotFoundError:
                If the file does not exist.

            EncryptedPDFError:
                If the PDF requires a password.

            CorruptedPDFError:
                If the file cannot be opened or parsed.
        """

        if not file_path.exists():
            raise PDFNotFoundError(f"PDF not found: {file_path}")

        if not file_path.is_file():
            raise PDFNotFoundError(f"PDF path is not a file: {file_path}")

        try:
            document = fitz.open(file_path)

        except (fitz.FileDataError, RuntimeError, ValueError) as exc:
            raise CorruptedPDFError("The PDF could not be opened.") from exc

        try:
            if document.needs_pass:
                raise EncryptedPDFError("Password-protected PDFs are not supported.")

            pages: list[ParsedPage] = []

            for page_index in range(document.page_count):
                try:
                    page = document.load_page(page_index)
                    raw_text = page.get_text("text")
                except (RuntimeError, ValueError) as exc:
                    raise CorruptedPDFError(
                        f"Page {page_index + 1} could not be parsed."
                    ) from exc

                normalized_text = self.normalize_text(raw_text)
                character_count = len(normalized_text)
                is_empty = character_count == 0

                may_require_ocr = character_count < self._minimum_text_characters

                pages.append(
                    ParsedPage(
                        page_number=page_index + 1,
                        text=normalized_text,
                        character_count=character_count,
                        is_empty=is_empty,
                        may_require_ocr=may_require_ocr,
                    )
                )

            extracted_character_count = sum(page.character_count for page in pages)

            empty_page_count = sum(1 for page in pages if page.is_empty)

            requires_ocr = any(page.may_require_ocr for page in pages)

            return ParsedDocument(
                document_id=document_id,
                page_count=len(pages),
                pages=pages,
                extracted_character_count=extracted_character_count,
                empty_page_count=empty_page_count,
                requires_ocr=requires_ocr,
            )

        finally:
            document.close()

    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize whitespace without destroying page structure."""

        lines = []

        for raw_line in text.splitlines():
            normalized_line = MULTIPLE_SPACES_PATTERN.sub(
                " ",
                raw_line,
            ).strip()

            lines.append(normalized_line)

        normalized = "\n".join(lines)
        normalized = EXCESSIVE_NEWLINES_PATTERN.sub(
            "\n\n",
            normalized,
        )

        return normalized.strip()
