from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from backend.models import (
    DocumentMetadata,
    Finding,
    ReviewDecision,
)


class ApiSchema(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )


class HealthResponse(ApiSchema):

    status: str = "healthy"
    application: str
    version: str
    environment: str


class DocumentUploadResponse(ApiSchema):

    document: DocumentMetadata
    message: str


class StartAnalysisRequest(ApiSchema):

    regulation_document_id: UUID
    policy_document_ids: list[UUID] = Field(min_length=1)

    @property
    def unique_policy_document_ids(self) -> set[UUID]:
        """Return policy identifiers with duplicates removed."""

        return set(self.policy_document_ids)


class StartAnalysisResponse(ApiSchema):

    analysis_id: UUID
    status: str
    regulation_document_id: UUID
    policy_document_ids: list[UUID]
    created_at: datetime


class FindingResponse(ApiSchema):

    finding: Finding


class ReviewFindingRequest(ApiSchema):

    decision: ReviewDecision
    reviewer_id: str = Field(min_length=1, max_length=200)
    comment: str | None = Field(default=None, max_length=5_000)


class ReviewFindingResponse(ApiSchema):

    finding: Finding
    message: str


class AnalysisSummaryResponse(ApiSchema):

    analysis_id: UUID
    regulation_document_id: UUID

    policy_document_count: int = Field(ge=1)
    requirement_count: int = Field(ge=0)
    finding_count: int = Field(ge=0)
    pending_review_count: int = Field(ge=0)

    status: str
