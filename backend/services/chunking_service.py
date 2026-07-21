"""Deterministic document chunking with page provenance."""

from __future__ import annotations

from backend.models import (
    DocumentChunk,
    ParsedDocument,
)


class ChunkingService:
    """Split parsed pages into retrieval-sized overlapping chunks."""

    def __init__(
        self,
        *,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero.")

        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative.")

        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")

        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def chunk_document(
        self,
        parsed_document: ParsedDocument,
    ) -> list[DocumentChunk]:
        """Create ordered chunks while preserving page references."""

        chunks: list[DocumentChunk] = []
        chunk_index = 0

        for page in parsed_document.pages:
            if page.is_empty:
                continue

            page_chunks = self._chunk_page_text(
                document_id=parsed_document.document_id,
                page_number=page.page_number,
                text=page.text,
                starting_chunk_index=chunk_index,
            )

            chunks.extend(page_chunks)
            chunk_index += len(page_chunks)

        return chunks

    def _chunk_page_text(
        self,
        *,
        document_id,
        page_number: int,
        text: str,
        starting_chunk_index: int,
    ) -> list[DocumentChunk]:
        """Split one page using deterministic character windows."""

        chunks: list[DocumentChunk] = []

        start = 0
        local_index = 0
        text_length = len(text)

        while start < text_length:
            proposed_end = min(
                start + self._chunk_size,
                text_length,
            )

            end = self._find_breakpoint(
                text=text,
                start=start,
                proposed_end=proposed_end,
            )

            chunk_text = text[start:end].strip()

            if chunk_text:
                leading_whitespace = len(text[start:end]) - len(
                    text[start:end].lstrip()
                )

                actual_start = start + leading_whitespace
                actual_end = actual_start + len(chunk_text)

                chunks.append(
                    DocumentChunk(
                        document_id=document_id,
                        page_number=page_number,
                        chunk_index=(starting_chunk_index + local_index),
                        text=chunk_text,
                        character_count=len(chunk_text),
                        start_character=actual_start,
                        end_character=actual_end,
                    )
                )

                local_index += 1

            if end >= text_length:
                break

            start = max(
                end - self._chunk_overlap,
                start + 1,
            )

        return chunks

    def _find_breakpoint(
        self,
        *,
        text: str,
        start: int,
        proposed_end: int,
    ) -> int:
        """Prefer paragraph, sentence, or whitespace boundaries."""

        if proposed_end >= len(text):
            return len(text)

        search_region = text[start:proposed_end]

        minimum_breakpoint = int(self._chunk_size * 0.6)

        candidates = [
            search_region.rfind("\n\n"),
            search_region.rfind(". "),
            search_region.rfind("\n"),
            search_region.rfind(" "),
        ]

        for candidate in candidates:
            if candidate >= minimum_breakpoint:
                boundary_adjustment = 1

                if search_region[candidate : candidate + 2] == ". ":
                    boundary_adjustment = 1

                return start + candidate + boundary_adjustment

        return proposed_end
