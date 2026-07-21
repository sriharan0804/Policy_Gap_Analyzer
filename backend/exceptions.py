"""Application-specific exceptions.

Service-layer exceptions keep business errors independent from HTTP.
The API layer will later translate them into HTTP responses.
"""


class DocumentServiceError(Exception):
    """Base exception for document-processing failures."""


class InvalidDocumentError(DocumentServiceError):
    """Raised when an uploaded document fails validation."""


class DocumentTooLargeError(InvalidDocumentError):
    """Raised when an uploaded document exceeds the configured limit."""


class UnsupportedDocumentTypeError(InvalidDocumentError):
    """Raised when an unsupported file type is uploaded."""


class DuplicateDocumentError(DocumentServiceError):
    """Raised when the exact document has already been registered."""


class DocumentStorageError(DocumentServiceError):
    """Raised when a document cannot be saved safely."""


class PDFParsingError(DocumentServiceError):
    """Base exception for PDF parsing failures."""


class PDFNotFoundError(PDFParsingError):
    """Raised when a stored PDF cannot be found."""


class EncryptedPDFError(PDFParsingError):
    """Raised when a PDF is password-protected."""


class CorruptedPDFError(PDFParsingError):
    """Raised when a PDF cannot be opened or parsed."""

class EmbeddingError(Exception):
    """Base exception for embedding generation failures."""


class EmptyEmbeddingInputError(EmbeddingError):
    """Raised when no valid text is supplied for embedding."""


class EmbeddingModelError(EmbeddingError):
    """Raised when the embedding model cannot generate vectors."""

class VectorStoreError(Exception):
    """Base exception for vector-index failures."""


class VectorDimensionError(VectorStoreError):
    """Raised when vectors use an unexpected dimension."""


class EmptyVectorStoreError(VectorStoreError):
    """Raised when a search is attempted on an empty index."""


class VectorStorePersistenceError(VectorStoreError):
    """Raised when an index cannot be saved or loaded."""

class RequirementExtractionError(Exception):
    """Base exception for regulatory requirement extraction."""


class EmptyRequirementTextError(RequirementExtractionError):
    """Raised when requirement extraction receives empty text."""


class PolicyExtractionError(Exception):
    """Base exception for internal policy extraction failures."""


class EmptyPolicyTextError(PolicyExtractionError):
    """Raised when policy extraction receives unusable text."""

class GapComparisonError(Exception):
    """Base exception for policy gap comparison failures."""


class MissingComparisonEvidenceError(GapComparisonError):
    """Raised when a comparison cannot be performed without evidence."""

class ConfidenceScoringError(Exception):
    """Base exception for confidence scoring failures."""


class MissingGapEvidenceError(ConfidenceScoringError):
    """Raised when required gap evidence is unavailable."""


class InvalidRetrievalScoreError(ConfidenceScoringError):
    """Raised when a retrieval score is outside the supported range."""

class RiskScoringError(Exception):
    """Base exception for risk scoring failures."""


class InvalidRiskInputError(RiskScoringError):
    """Raised when risk inputs are invalid."""