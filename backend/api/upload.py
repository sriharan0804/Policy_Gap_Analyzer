"""Document upload API routes."""

from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)

from backend.config import get_settings , Settings
from backend.exceptions import (
    DocumentStorageError,
    DocumentTooLargeError,
    InvalidDocumentError,
    UnsupportedDocumentTypeError,
)
from backend.models import DocumentType, IssuingAuthority
from backend.schemas import DocumentUploadResponse
from backend.services.document_service import DocumentService


router = APIRouter(
    prefix="/documents",
    tags=["documents"],
)


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(
    file: Annotated[UploadFile, File(description="PDF document")],
    document_type: Annotated[DocumentType, Form()],
    issuing_authority: Annotated[IssuingAuthority | None, Form()] = None,
    title: Annotated[str | None, Form(max_length=500)] = None,
    settings: Settings = Depends(get_settings),
) -> DocumentUploadResponse:
    """Validate, store, and register an uploaded document."""


    maximum_bytes = settings.max_upload_size_mb * 1024 * 1024

    # Read one extra byte so oversized documents can be rejected without
    # silently truncating the upload.
    file_content = await file.read(maximum_bytes + 1)

    service = DocumentService(settings)

    try:
        document = service.register_document(
            file_content=file_content,
            original_filename=file.filename or "",
            content_type=file.content_type,
            document_type=document_type,
            issuing_authority=issuing_authority,
            title=title,
        )

    except DocumentTooLargeError as exc:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=str(exc),
        ) from exc

    except UnsupportedDocumentTypeError as exc:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=str(exc),
        ) from exc

    except InvalidDocumentError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except DocumentStorageError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The document could not be stored.",
        ) from exc

    finally:
        await file.close()

    return DocumentUploadResponse(
        document=document,
        message="Document uploaded and registered successfully.",
    )