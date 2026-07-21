from pathlib import Path

import pytest

from backend.config import Settings
from backend.exceptions import (
    DocumentTooLargeError,
    InvalidDocumentError,
    UnsupportedDocumentTypeError,
)
from backend.models import (
    DocumentType,
    IssuingAuthority,
)
from backend.services.document_service import DocumentService

MINIMAL_PDF = (
    b"%PDF-1.4\n"
    b"1 0 obj\n"
    b"<< /Type /Catalog >>\n"
    b"endobj\n"
    b"trailer\n"
    b"<< /Root 1 0 R >>\n"
    b"%%EOF"
)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:

    return Settings(
        app_env="test",
        data_directory=tmp_path,
        regulation_directory=tmp_path / "regulations",
        policy_directory=tmp_path / "policies",
        processed_directory=tmp_path / "processed",
        faiss_directory=tmp_path / "faiss",
        max_upload_size_mb=1,
    )


@pytest.fixture
def service(settings: Settings) -> DocumentService:

    return DocumentService(settings)


def test_register_regulation(service: DocumentService, settings: Settings):

    document = service.register_document(
        file_content=MINIMAL_PDF,
        original_filename="SEC Rule Test.pdf",
        content_type="application/pdf",
        document_type=DocumentType.REGULATION,
        issuing_authority=IssuingAuthority.SEC,
        title="SEC Test Rule",
    )

    stored_path = settings.regulation_directory / document.stored_filename

    assert stored_path.exists()
    assert stored_path.read_bytes() == MINIMAL_PDF

    assert document.original_filename == "SEC Rule Test.pdf"
    assert document.document_type == DocumentType.REGULATION
    assert document.issuing_authority == IssuingAuthority.SEC
    assert len(document.checksum_sha256) == 64


def test_register_policy(service: DocumentService, settings: Settings):

    document = service.register_document(
        file_content=MINIMAL_PDF,
        original_filename="Records Policy.pdf",
        content_type="application/pdf",
        document_type=DocumentType.POLICY,
    )

    stored_path = settings.policy_directory / document.stored_filename

    assert stored_path.exists()
    assert document.issuing_authority is None


def test_reject_empty_document(service: DocumentService):

    with pytest.raises(
        InvalidDocumentError,
        match="empty",
    ):
        service.register_document(
            file_content=b"",
            original_filename="empty.pdf",
            content_type="application/pdf",
            document_type=DocumentType.POLICY,
        )


def test_reject_non_pdf_extension(service: DocumentService):

    with pytest.raises(
        UnsupportedDocumentTypeError,
        match="Only PDF",
    ):
        service.register_document(
            file_content=MINIMAL_PDF,
            original_filename="policy.txt",
            content_type="text/plain",
            document_type=DocumentType.POLICY,
        )


def test_reject_invalid_pdf_signature(service: DocumentService):

    with pytest.raises(
        UnsupportedDocumentTypeError,
        match="PDF signature",
    ):
        service.register_document(
            file_content=b"This is not a PDF.",
            original_filename="fake.pdf",
            content_type="application/pdf",
            document_type=DocumentType.POLICY,
        )


def test_reject_directory_traversal_filename(
    service: DocumentService,
):

    with pytest.raises(
        InvalidDocumentError,
        match="directory path",
    ):
        service.register_document(
            file_content=MINIMAL_PDF,
            original_filename="../policy.pdf",
            content_type="application/pdf",
            document_type=DocumentType.POLICY,
        )


def test_reject_oversized_document(service: DocumentService):

    oversized_content = b"%PDF-" + (b"x" * (1024 * 1024))

    with pytest.raises(
        DocumentTooLargeError,
        match="maximum size",
    ):
        service.register_document(
            file_content=oversized_content,
            original_filename="large.pdf",
            content_type="application/pdf",
            document_type=DocumentType.POLICY,
        )


def test_checksum_is_deterministic(service: DocumentService):

    checksum_one = service.calculate_sha256(MINIMAL_PDF)
    checksum_two = service.calculate_sha256(MINIMAL_PDF)

    assert checksum_one == checksum_two
    assert len(checksum_one) == 64
