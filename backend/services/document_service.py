from __future__ import annotations

import hashlib
import re
from pathlib import Path
from uuid import uuid4

from backend.config import Settings
from backend.exceptions import (
    DocumentStorageError,
    DocumentTooLargeError,
    InvalidDocumentError,
    UnsupportedDocumentTypeError,
)
from backend.models import (
    DocumentMetadata,
    DocumentType,
    IssuingAuthority,
)

PDF_MIME_TYPES = {
    "application/pdf",
    "application/x-pdf",
}

PDF_SIGNATURE = b"%PDF-"

SAFE_FILENAME_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


class DocumentService:

    def __init__(self, settings: Settings) -> None:

        self._settings = settings

    def register_document(
        self,
        *,
        file_content: bytes,
        original_filename: str,
        content_type: str | None,
        document_type: DocumentType,
        issuing_authority: IssuingAuthority | None = None,
        title: str | None = None,
    ) -> DocumentMetadata:
        """Validate, save, and register one uploaded PDF.

        Args:
            file_content:
                Complete uploaded file contents.

            original_filename:
                Filename supplied by the upload client.

            content_type:
                MIME type supplied by the client.

            document_type:
                Whether the document is a regulation or internal policy.

            issuing_authority:
                Regulatory issuer. Required for regulatory documents.

            title:
                Optional human-readable document title.

        Returns:
            Validated metadata for the stored document.

        Raises:
            InvalidDocumentError:
                If the file is empty or has an invalid filename.

            UnsupportedDocumentTypeError:
                If the file is not recognized as a PDF.

            DocumentTooLargeError:
                If the file exceeds the configured upload limit.

            DocumentStorageError:
                If the validated file cannot be written safely.
        """

        self._validate_filename(original_filename)
        self._validate_file_size(file_content)
        self._validate_pdf(
            file_content=file_content,
            original_filename=original_filename,
            content_type=content_type,
        )

        checksum = self.calculate_sha256(file_content)
        stored_filename = self._generate_stored_filename(
            original_filename=original_filename,
            checksum=checksum,
        )

        destination_directory = self._directory_for_document_type(document_type)
        destination_directory.mkdir(parents=True, exist_ok=True)

        destination_path = destination_directory / stored_filename

        self._write_file_atomically(
            destination_path=destination_path,
            file_content=file_content,
        )

        try:
            return DocumentMetadata(
                document_type=document_type,
                original_filename=original_filename,
                stored_filename=stored_filename,
                checksum_sha256=checksum,
                title=title,
                issuing_authority=issuing_authority,
                file_size_bytes=len(file_content),
                mime_type="application/pdf",
            )
        except Exception:
            # If metadata validation fails, remove the stored file so that
            # storage and registration state do not become inconsistent.
            destination_path.unlink(missing_ok=True)
            raise

    @staticmethod
    def calculate_sha256(file_content: bytes) -> str:

        return hashlib.sha256(file_content).hexdigest()

    def _validate_filename(self, original_filename: str) -> None:

        if not original_filename or not original_filename.strip():
            raise InvalidDocumentError("A filename is required.")

        candidate = Path(original_filename)

        if candidate.name != original_filename:
            raise InvalidDocumentError(
                "The filename must not contain a directory path."
            )

        if original_filename in {".", ".."}:
            raise InvalidDocumentError("The filename is invalid.")

    def _validate_file_size(self, file_content: bytes) -> None:

        if not file_content:
            raise InvalidDocumentError("The uploaded document is empty.")

        maximum_bytes = self._settings.max_upload_size_mb * 1024 * 1024

        if len(file_content) > maximum_bytes:
            raise DocumentTooLargeError(
                "The uploaded document exceeds the maximum size of "
                f"{self._settings.max_upload_size_mb} MB."
            )

    def _validate_pdf(
        self,
        *,
        file_content: bytes,
        original_filename: str,
        content_type: str | None,
    ) -> None:

        suffix = Path(original_filename).suffix.lower()

        if suffix != ".pdf":
            raise UnsupportedDocumentTypeError("Only PDF documents are supported.")

        if content_type and content_type.lower() not in PDF_MIME_TYPES:
            raise UnsupportedDocumentTypeError(
                f"Unsupported content type: {content_type}."
            )

        if not file_content.startswith(PDF_SIGNATURE):
            raise UnsupportedDocumentTypeError(
                "The uploaded file does not contain a valid PDF signature."
            )

    def _generate_stored_filename(
        self,
        *,
        original_filename: str,
        checksum: str,
    ) -> str:

        original_stem = Path(original_filename).stem
        safe_stem = SAFE_FILENAME_PATTERN.sub("_", original_stem)
        safe_stem = safe_stem.strip("._-") or "document"

        # Keep names manageable on Windows and Linux file systems.
        safe_stem = safe_stem[:80]

        unique_suffix = uuid4().hex[:12]
        checksum_prefix = checksum[:12]

        return f"{safe_stem}_{checksum_prefix}_{unique_suffix}.pdf"

    def _directory_for_document_type(
        self,
        document_type: DocumentType,
    ) -> Path:

        if document_type == DocumentType.REGULATION:
            return self._settings.regulation_directory

        return self._settings.policy_directory

    def _write_file_atomically(
        self,
        *,
        destination_path: Path,
        file_content: bytes,
    ) -> None:

        temporary_path = destination_path.with_suffix(".tmp")

        try:
            with temporary_path.open("wb") as file_handle:
                file_handle.write(file_content)
                file_handle.flush()

            temporary_path.replace(destination_path)

        except OSError as exc:
            temporary_path.unlink(missing_ok=True)
            destination_path.unlink(missing_ok=True)

            raise DocumentStorageError("The document could not be stored.") from exc
