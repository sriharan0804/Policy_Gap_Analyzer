from __future__ import annotations
from datetime import date , datetime , timezone
from enum import StrEnum , Enum
from typing import Annotated
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field ,  model_validator

def utc_now() -> datetime:
    return datetime.now(timezone.utc)

NonEmptyText = Annotated[str , Field(min_length=1)]
NormalizedScore = Annotated[float , Field(ge=0.0 , le=1.0)]
PositivePageNumber = Annotated[int , Field(ge=1)]

class DomainModel(BaseModel):
    model_config = ConfigDict(
        extra = "forbid",
        validate_assignment = True,
        str_strip_whitespace = True,
        use_enum_values = False,
    )

class DocumentType(StrEnum):
    REGULATION = "regulation"
    POLICY = "policy"

class IssuingAuthority(StrEnum):
    SEC = "sec"
    FINRA = "finra"
    OTHER = "other"

class ProcessingStatus(StrEnum):
    UPLOADED = "uploaded"
    VALIDATING = "validating"
    PARSING = "parsing"
    PROCESSED = "processed"
    FAILED = "failed"

class RequirementType(StrEnum):
    OBLIGATION = "obligation"
    PROHIBITION = "prohibition"
    REPORTING = "reporting"
    RECORDKEEPING ="recordkeeping"
    DISCLOSURE = "disclosure"
    SUPERVISION = "supervision"
    GOVERNANCE = "governance"
    TRAINING = "training"
    PERMISSION = "permission"
    CONDITION = "condition"
    OTHER = "other"

class RequirementModality(StrEnum):
    MANDATORY = "mandatory"
    RECOMMENDED = "recommended"
    PERMISSIVE = "permissive"
    UNCLEAR = "unclear"

class GapClassification(StrEnum):
    COVERED = "covered"
    PARTIALLY_COVERED = "partially_covered"
    NOT_COVERED = "not_covered"
    CONTRADICTORY = "contradictory"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    REQUIRES_LEGAL_REVIEW = "requires_legal_review"

class RiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ReviewDecision(StrEnum):
    PENDING = "pending"
    ACCEPT = "accept"
    REJECT = "reject"
    NEEDS_MORE_EVIDENCE = "needs_more_evidence"
    ESCALATE_TO_LEGAL = "escalate_to_legal"
    NOT_APPLICABLE = "not_applicable"

class ValidationStatus(StrEnum):
    NOT_RUN = "not_run"
    PASSED = "passed"
    FAILED = "failed"

class DocumentMetadata(DomainModel):
    """Registered source-document metadata.

    The checksum identifies the exact uploaded file version. It will later
    be calculated using SHA-256 during upload.
    """

    document_id: UUID = Field(default_factory=uuid4)
    document_type: DocumentType

    original_filename: NonEmptyText
    stored_filename: NonEmptyText
    checksum_sha256: Annotated[str, Field(pattern=r"^[a-fA-F0-9]{64}$")]

    title: NonEmptyText | None = None
    issuing_authority: IssuingAuthority | None = None

    publication_date: date | None = None
    effective_date: date | None = None

    page_count: Annotated[int, Field(ge=1)] | None = None
    mime_type: str = "application/pdf"
    file_size_bytes: Annotated[int, Field(gt=0)]

    processing_status: ProcessingStatus = ProcessingStatus.UPLOADED
    parser_name: str | None = None
    parser_version: str | None = None

    uploaded_at: datetime = Field(default_factory=utc_now)
    processed_at: datetime | None = None

    @model_validator(mode="after")
    def validate_regulatory_metadata(self) -> DocumentMetadata:
        """Ensure regulation-specific metadata is internally consistent."""

        if (
            self.document_type == DocumentType.REGULATION
            and self.issuing_authority is None
        ):
            raise ValueError(
                "issuing_authority is required for regulatory documents"
            )

        if (
            self.publication_date is not None
            and self.effective_date is not None
            and self.effective_date < self.publication_date
        ):
            raise ValueError(
                "effective_date cannot be earlier than publication_date"
            )

        if (
            self.processing_status == ProcessingStatus.PROCESSED
            and self.processed_at is None
        ):
            raise ValueError(
                "processed_at is required when processing_status is processed"
            )

        return self

class ParsedPage(DomainModel):
    """Text extracted from one PDF page."""

    page_number: int = Field(ge=1)
    text: str
    character_count: int = Field(ge=0)
    is_empty: bool
    may_require_ocr: bool

    @model_validator(mode="after")
    def validate_character_count(self) -> "ParsedPage":
        """Ensure metadata matches the extracted text."""

        actual_count = len(self.text)

        if self.character_count != actual_count:
            raise ValueError(
                "character_count must equal the length of text."
            )

        if self.is_empty != (actual_count == 0):
            raise ValueError(
                "is_empty must reflect whether extracted text is empty."
            )

        return self


class ParsedDocument(DomainModel):
    """Structured page-level text extracted from a PDF."""

    document_id: UUID
    page_count: int = Field(ge=0)
    pages: list[ParsedPage]
    extracted_character_count: int = Field(ge=0)
    empty_page_count: int = Field(ge=0)
    requires_ocr: bool

    @model_validator(mode="after")
    def validate_document_totals(self) -> "ParsedDocument":
        """Ensure page and extraction totals remain consistent."""

        if self.page_count != len(self.pages):
            raise ValueError(
                "page_count must equal the number of parsed pages."
            )

        expected_characters = sum(
            page.character_count for page in self.pages
        )

        if self.extracted_character_count != expected_characters:
            raise ValueError(
                "extracted_character_count does not match page totals."
            )

        expected_empty_pages = sum(
            1 for page in self.pages if page.is_empty
        )

        if self.empty_page_count != expected_empty_pages:
            raise ValueError(
                "empty_page_count does not match parsed pages."
            )

        expected_requires_ocr = any(
            page.may_require_ocr for page in self.pages
        )

        if self.requires_ocr != expected_requires_ocr:
            raise ValueError(
                "requires_ocr must reflect page-level OCR flags."
            )

        return self
    
class SourceLocation(DomainModel):

    document_id: UUID
    page_number: PositivePageNumber

    section_title: str | None = None
    paragraph_number: Annotated[int, Field(ge=1)] | None = None

    character_start: Annotated[int, Field(ge=0)] | None = None
    character_end: Annotated[int, Field(gt=0)] | None = None

    @model_validator(mode="after")
    def validate_character_range(self) -> SourceLocation:
        """Ensure character offsets form a valid half-open range."""

        if (
            self.character_start is not None
            and self.character_end is not None
            and self.character_end <= self.character_start
        ):
            raise ValueError(
                "character_end must be greater than character_start"
            )

        return self
    
class Evidence(DomainModel):
    evidence_id: UUID = Field(default_factory=uuid4)
    location: SourceLocation
    text: NonEmptyText

    retrieval_score: NormalizedScore | None = None
    embedding_model: str | None = None

    is_primary_evidence: bool = False

class RegulatoryRequirement(DomainModel):

    requirement_id: UUID = Field(default_factory=uuid4)
    document_id: UUID

    requirement_text: NonEmptyText
    normalized_requirement: NonEmptyText

    requirement_type: RequirementType
    modality: RequirementModality

    source_evidence: Evidence
    extraction_confidence: NormalizedScore

    applies_from: date | None = None
    extraction_model: str
    prompt_version: str

    requires_human_confirmation: bool = True
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_source_document(self) -> RegulatoryRequirement:
        """Require requirement and source evidence to reference one document."""

        if self.source_evidence.location.document_id != self.document_id:
            raise ValueError(
                "requirement document_id must match source evidence document_id"
            )

        return self

class ConfidenceComponents(DomainModel):
   
    retrieval_quality: NormalizedScore
    regulatory_evidence_quality: NormalizedScore
    policy_evidence_quality: NormalizedScore
    validation_success: NormalizedScore
    requirement_clarity: NormalizedScore


class ConfidenceAssessment(DomainModel):
  

    score: NormalizedScore
    components: ConfidenceComponents
    calculation_version: NonEmptyText
    explanation: NonEmptyText

class RiskComponents(DomainModel):
    mandatory_requirement: bool
    missing_requirement: bool
    contradictory_policy: bool
    partial_coverage: bool
    regulatory_evidence_available: bool
    policy_evidence_available: bool

    business_impact_weight: Annotated[int, Field(ge=0, le=5)] = 0


class RiskAssessment(DomainModel):

    score: Annotated[int, Field(ge=0, le=100)]
    level: RiskLevel
    components: RiskComponents
    calculation_version: NonEmptyText

    explanation: NonEmptyText


class ValidationResult(DomainModel):
    status: ValidationStatus = ValidationStatus.NOT_RUN
    checks_run: list[str] = Field(default_factory=list)
    failed_checks: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    validated_at: datetime | None = None

    @model_validator(mode="after")
    def validate_status_consistency(self) -> ValidationResult:
        """Keep validation status consistent with failed checks."""

        if self.status == ValidationStatus.PASSED and self.failed_checks:
            raise ValueError(
                "passed validation cannot contain failed checks"
            )

        if self.status == ValidationStatus.FAILED and not self.failed_checks:
            raise ValueError(
                "failed validation must contain at least one failed check"
            )

        return self


class HumanReview(DomainModel):

    decision: ReviewDecision = ReviewDecision.PENDING

    reviewer_id: str | None = None
    reviewer_comment: str | None = None
    reviewed_at: datetime | None = None

    @model_validator(mode="after")
    def validate_completed_review(self) -> HumanReview:
        """Require reviewer metadata after a final review decision."""

        if self.decision != ReviewDecision.PENDING:
            if not self.reviewer_id:
                raise ValueError(
                    "reviewer_id is required for a completed review"
                )

            if self.reviewed_at is None:
                raise ValueError(
                    "reviewed_at is required for a completed review"
                )

        return self


class Finding(DomainModel):

    finding_id: UUID = Field(default_factory=uuid4)
    analysis_id: UUID

    requirement: RegulatoryRequirement
    policy_evidence: list[Evidence] = Field(default_factory=list)

    classification: GapClassification
    rationale: NonEmptyText

    confidence: ConfidenceAssessment
    risk: RiskAssessment
    validation: ValidationResult
    human_review: HumanReview = Field(default_factory=HumanReview)

    comparison_model: NonEmptyText
    prompt_version: NonEmptyText

    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_evidence_classification(self) -> Finding:

        evidence_required_classifications = {
            GapClassification.COVERED,
            GapClassification.PARTIALLY_COVERED,
            GapClassification.CONTRADICTORY,
        }

        if (
            self.classification in evidence_required_classifications
            and not self.policy_evidence
        ):
            raise ValueError(
                f"{self.classification.value} requires policy evidence"
            )

        return self
    
class DocumentChunk(DomainModel):
    """A retrieval-sized passage with source provenance."""

    chunk_id: UUID = Field(default_factory=uuid4)
    document_id: UUID
    page_number: int = Field(ge=1)
    chunk_index: int = Field(ge=0)

    text: str = Field(min_length=1)
    character_count: int = Field(ge=1)

    start_character: int = Field(ge=0)
    end_character: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_chunk_boundaries(self) -> "DocumentChunk":
        if self.character_count != len(self.text):
            raise ValueError(
                "character_count must equal the length of text."
            )

        if self.end_character <= self.start_character:
            raise ValueError(
                "end_character must be greater than start_character."
            )

        if (
            self.end_character - self.start_character
            != self.character_count
        ):
            raise ValueError(
                "Chunk offsets must match character_count."
            )

        return self

class RetrievedChunk(DomainModel):
    """A source chunk returned by semantic retrieval."""

    chunk: DocumentChunk
    similarity_score: float = Field(ge=-1.0, le=1.0)
    rank: int = Field(ge=1)


class RequirementModality(str, Enum):
    """Strength of a regulatory statement."""

    MANDATORY = "mandatory"
    PROHIBITED = "prohibited"
    PERMISSIVE = "permissive"
    ADVISORY = "advisory"
    UNKNOWN = "unknown"


class RequirementCandidate(DomainModel):
    """A structured regulatory obligation extracted from source text."""

    requirement_id: UUID = Field(default_factory=uuid4)

    document_id: UUID
    chunk_id: UUID
    page_number: int = Field(ge=1)
    chunk_index: int = Field(ge=0)

    source_text: str = Field(min_length=1)
    subject: str | None = None
    action: str = Field(min_length=1)
    object: str | None = None
    condition: str | None = None
    timing: str | None = None

    modality: RequirementModality
    matched_trigger: str

    extraction_confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

class PolicyStatementType(str, Enum):
    """Classification of an internal policy statement."""

    CONTROL = "control"
    PROHIBITION = "prohibition"
    PERMISSION = "permission"
    RESPONSIBILITY = "responsibility"
    REVIEW = "review"
    RECORD_RETENTION = "record_retention"
    UNKNOWN = "unknown"


class PolicyStatement(DomainModel):
    """A structured statement extracted from an internal policy."""

    statement_id: UUID = Field(default_factory=uuid4)

    document_id: UUID
    chunk_id: UUID
    page_number: int = Field(ge=1)
    chunk_index: int = Field(ge=0)

    source_text: str = Field(min_length=1)

    subject: str | None = None
    action: str = Field(min_length=1)
    object: str | None = None

    condition: str | None = None
    timing: str | None = None
    responsible_party: str | None = None

    statement_type: PolicyStatementType
    matched_trigger: str

    extraction_confidence: float = Field(
        ge=0.0,
        le=1.0,
    )


class GapStatus(str, Enum):
    """Deterministic classification of a policy gap."""

    FULLY_ADDRESSED = "fully_addressed"
    PARTIALLY_ADDRESSED = "partially_addressed"
    NOT_ADDRESSED = "not_addressed"
    CONTRADICTED = "contradicted"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class ComparisonComponents(DomainModel):
    """Individual deterministic comparison scores."""

    action_score: float = Field(ge=0.0, le=1.0)
    object_score: float = Field(ge=0.0, le=1.0)
    timing_score: float = Field(ge=0.0, le=1.0)
    condition_score: float = Field(ge=0.0, le=1.0)
    modality_score: float = Field(ge=0.0, le=1.0)

    overall_score: float = Field(ge=0.0, le=1.0)


class PolicyMatch(DomainModel):
    """Comparison between one requirement and one policy statement."""

    policy_statement_id: UUID
    policy_document_id: UUID
    policy_chunk_id: UUID

    page_number: int = Field(ge=1)
    chunk_index: int = Field(ge=0)

    source_text: str = Field(min_length=1)

    components: ComparisonComponents
    is_contradiction: bool = False
    reasons: list[str] = Field(default_factory=list)


class GapAssessment(DomainModel):
    """Final gap decision for one regulatory requirement."""

    assessment_id: UUID = Field(default_factory=uuid4)

    requirement_id: UUID
    regulatory_document_id: UUID
    regulatory_chunk_id: UUID

    status: GapStatus

    best_match: PolicyMatch | None = None
    evaluated_policy_count: int = Field(ge=0)

    deterministic_score: float = Field(
        ge=0.0,
        le=1.0,
    )

    rationale: list[str] = Field(default_factory=list)

    requires_human_review: bool = True