import tempfile
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.models import (
    AnalysisResult,
    DataSensitivity,
    GapHumanReview,
    GapReviewerDecision,
    GapReviewStatus,
    GapStatus,
    RegulatoryImpact,
)
from backend.services.chunking_service import ChunkingService
from backend.services.confidence_scoring_service import (
    DeterministicConfidenceScoringService,
)
from backend.services.embedding_service import (
    SentenceTransformerEmbeddingService,
)
from backend.services.explanation_service import (
    DeterministicExplanationService,
)
from backend.services.gap_comparison_service import (
    DeterministicGapComparisonService,
)
from backend.services.human_review_service import (
    DeterministicHumanReviewService,
)
from backend.services.pdf_parser_service import PDFParserService
from backend.services.policy_extraction_service import (
    RuleBasedPolicyExtractionService,
)
from backend.services.report_service import (
    DeterministicReportService,
)
from backend.services.requirement_extraction_service import (
    RuleBasedRequirementExtractionService,
)
from backend.services.risk_scoring_service import (
    DeterministicRiskScoringService,
)
from backend.services.vector_store import FaissVectorStore


BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"

STATIC_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "application/x-pdf",
}

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


app = FastAPI(
    title="AI-Assisted Regulatory Policy Gap Analyzer",
    description=(
        "Compares regulatory requirements against internal policies "
        "and identifies compliance gaps."
    ),
    version="1.1.0",
)

app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static",
)


# ---------------------------------------------------------------------------
# Temporary application storage
# ---------------------------------------------------------------------------
#
# These dictionaries keep analyses and reviews while FastAPI is running.
# Restarting the application clears them. SQLite persistence can be added later.

analysis_results_store: dict[
    UUID,
    AnalysisResult,
] = {}

human_reviews_store: dict[
    UUID,
    list[GapHumanReview],
] = {}


# ---------------------------------------------------------------------------
# Shared services
# ---------------------------------------------------------------------------

report_service = DeterministicReportService()
human_review_service = DeterministicHumanReviewService()


# ---------------------------------------------------------------------------
# API request models
# ---------------------------------------------------------------------------

class CreateHumanReviewRequest(BaseModel):
    requirement_id: UUID


class CompleteHumanReviewRequest(BaseModel):
    status: GapReviewStatus
    decision: GapReviewerDecision

    reviewer_id: str = Field(
        min_length=1,
        max_length=200,
    )

    reviewer_notes: str | None = Field(
        default=None,
        max_length=5000,
    )

    overridden_gap_status: GapStatus | None = None


# ---------------------------------------------------------------------------
# General helper functions
# ---------------------------------------------------------------------------

def validate_pdf(
    upload: UploadFile,
    field_name: str,
) -> None:
    """Validate an uploaded PDF."""

    filename = upload.filename or ""

    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} must be a PDF file.",
        )

    if (
        upload.content_type
        and upload.content_type not in ALLOWED_CONTENT_TYPES
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                f"{field_name} has an unsupported content type: "
                f"{upload.content_type}"
            ),
        )


def get_analysis_result(
    analysis_id: UUID,
) -> AnalysisResult:
    """Return a stored analysis result."""

    analysis_result = analysis_results_store.get(
        analysis_id
    )

    if analysis_result is None:
        raise HTTPException(
            status_code=404,
            detail="Analysis result was not found.",
        )

    return analysis_result


def get_analysis_reviews(
    analysis_id: UUID,
) -> list[GapHumanReview]:
    """Return all reviews belonging to an analysis."""

    get_analysis_result(analysis_id)

    return human_reviews_store.setdefault(
        analysis_id,
        [],
    )


def get_requirement_gap_assessment(
    analysis_result: AnalysisResult,
    requirement_id: UUID,
):
    """Return the automated gap assessment for a requirement."""

    for gap_assessment in analysis_result.gap_assessments:
        if (
            gap_assessment.requirement_id
            == requirement_id
        ):
            return gap_assessment

    raise HTTPException(
        status_code=404,
        detail=(
            "The requirement does not exist in this analysis."
        ),
    )


def get_review_by_requirement(
    analysis_id: UUID,
    requirement_id: UUID,
) -> GapHumanReview:
    """Return a requirement's human review."""

    reviews = get_analysis_reviews(analysis_id)

    for review in reviews:
        if review.requirement_id == requirement_id:
            return review

    raise HTTPException(
        status_code=404,
        detail=(
            "A human review has not been created for this "
            "requirement."
        ),
    )


def get_latest_review_map(
    analysis_id: UUID,
) -> dict[UUID, GapHumanReview]:
    """Create a requirement-to-review lookup."""

    return {
        review.requirement_id: review
        for review in get_analysis_reviews(analysis_id)
    }


def build_concise_analysis_response(
    compliance_report: object,
    reviews: list[GapHumanReview] | None = None,
) -> dict[str, object]:
    """Build the client-facing response used by Streamlit."""

    report_data = compliance_report.model_dump(
        mode="json"
    )

    review_map = {
        str(review.requirement_id): review
        for review in reviews or []
    }

    findings: list[dict[str, object]] = []

    for report in report_data["requirement_reports"]:
        requirement_id = report["requirement_id"]
        review = review_map.get(requirement_id)

        automated_status = report["gap_status"]
        effective_status = report["effective_gap_status"]
        review_status = report["review_status"]
        reviewer_decision = report["reviewer_decision"]

        if review is not None:
            review_status = review.status.value

            reviewer_decision = (
                review.decision.value
                if review.decision is not None
                else None
            )

            if review.overridden_gap_status is not None:
                effective_status = (
                    review.overridden_gap_status.value
                )
            else:
                effective_status = (
                    review.original_gap_status.value
                )

        findings.append(
            {
                "requirement_id": requirement_id,
                "gap_assessment_id": report[
                    "gap_assessment_id"
                ],
                "requirement_summary": report[
                    "requirement_summary"
                ],
                "policy_summary": report[
                    "policy_summary"
                ],
                "gap_status": automated_status,
                "effective_gap_status": effective_status,
                "gap_reason": report["gap_reason"],
                "confidence_score": report[
                    "confidence_score"
                ],
                "confidence_level": report[
                    "confidence_level"
                ],
                "risk_score": report["risk_score"],
                "risk_level": report["risk_level"],
                "recommended_action": report[
                    "recommended_action"
                ],
                "requires_human_review": report[
                    "requires_human_review"
                ],
                "review_status": review_status,
                "reviewer_decision": reviewer_decision,
                "reviewer_id": (
                    review.reviewer_id
                    if review is not None
                    else None
                ),
                "reviewer_notes": (
                    review.reviewer_notes
                    if review is not None
                    else None
                ),
            }
        )

    return {
        "status": "report_generated",
        "analysis_id": report_data["analysis_id"],
        "report_id": report_data["report_id"],
        "summary": report_data["summary"],
        "findings": findings,
        "generated_at": report_data["generated_at"],
        "report_version": report_data[
            "report_version"
        ],
    }


# ---------------------------------------------------------------------------
# System endpoints
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    tags=["System"],
)
def health_check() -> dict[str, str]:
    return {
        "status": "healthy",
        "service": "policy-gap-analyzer",
    }


@app.get(
    "/health/ai",
    tags=["System"],
)
def ai_health_check() -> dict[str, object]:
    try:
        embedding_service = (
            SentenceTransformerEmbeddingService(
                model_name=EMBEDDING_MODEL_NAME,
            )
        )

        embeddings = embedding_service.embed_texts(
            [
                (
                    "Regulatory policy gap analyzer "
                    "readiness check."
                )
            ]
        )

        return {
            "status": "healthy",
            "ai_model_loaded": True,
            "model_name": EMBEDDING_MODEL_NAME,
            "number_of_vectors": len(embeddings),
            "vector_dimension": len(embeddings[0]),
        }

    except Exception as exc:
        return {
            "status": "unhealthy",
            "ai_model_loaded": False,
            "error": str(exc),
        }


@app.get(
    "/",
    response_class=HTMLResponse,
    tags=["Demo"],
)
def demo_home() -> HTMLResponse:
    return HTMLResponse(
        content="""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta
                name="viewport"
                content="width=device-width, initial-scale=1.0"
            >
            <title>Policy Gap Analyzer</title>

            <style>
                * {
                    box-sizing: border-box;
                }

                body {
                    margin: 0;
                    min-height: 100vh;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    padding: 24px;
                    font-family: Arial, sans-serif;
                    background: #f4f7fb;
                    color: #172033;
                }

                main {
                    width: 100%;
                    max-width: 760px;
                    padding: 40px;
                    background: white;
                    border: 1px solid #dbe3ef;
                    border-radius: 16px;
                    box-shadow:
                        0 12px 35px rgba(23, 32, 51, 0.08);
                }

                h1 {
                    margin-top: 0;
                }

                p {
                    color: #5d687a;
                    line-height: 1.6;
                }

                .upload-section {
                    margin-bottom: 22px;
                }

                label {
                    display: block;
                    margin-bottom: 8px;
                    font-weight: 600;
                }

                input[type="file"] {
                    width: 100%;
                    padding: 14px;
                    border: 1px solid #cbd5e1;
                    border-radius: 10px;
                }

                button {
                    width: 100%;
                    padding: 14px 20px;
                    border: none;
                    border-radius: 10px;
                    font-size: 1rem;
                    font-weight: 600;
                    cursor: pointer;
                    color: white;
                    background: #315efb;
                }
            </style>
        </head>

        <body>
            <main>
                <h1>
                    AI-Assisted Regulatory Policy Gap Analyzer
                </h1>

                <p>
                    Upload a regulatory document and an internal
                    policy document to identify compliance gaps.
                </p>

                <form
                    action="/analyze"
                    method="post"
                    enctype="multipart/form-data"
                >
                    <div class="upload-section">
                        <label for="regulatory_document">
                            Regulatory document
                        </label>

                        <input
                            id="regulatory_document"
                            name="regulatory_document"
                            type="file"
                            accept=".pdf,application/pdf"
                            required
                        >
                    </div>

                    <div class="upload-section">
                        <label for="policy_document">
                            Internal policy document
                        </label>

                        <input
                            id="policy_document"
                            name="policy_document"
                            type="file"
                            accept=".pdf,application/pdf"
                            required
                        >
                    </div>

                    <button type="submit">
                        Analyze documents
                    </button>
                </form>
            </main>
        </body>
        </html>
        """
    )


# ---------------------------------------------------------------------------
# Analysis endpoint
# ---------------------------------------------------------------------------

@app.post(
    "/analyze",
    tags=["Analysis"],
    summary=(
        "Analyze regulatory and policy PDF documents"
    ),
)
async def analyze_documents(
    regulatory_document: UploadFile = File(
        ...,
        description=(
            "Regulatory document in PDF format."
        ),
    ),
    policy_document: UploadFile = File(
        ...,
        description=(
            "Internal policy document in PDF format."
        ),
    ),
) -> dict[str, object]:
    validate_pdf(
        upload=regulatory_document,
        field_name="Regulatory document",
    )

    validate_pdf(
        upload=policy_document,
        field_name="Policy document",
    )

    regulatory_path: Path | None = None
    policy_path: Path | None = None

    try:
        regulatory_content = (
            await regulatory_document.read()
        )

        policy_content = (
            await policy_document.read()
        )

        if not regulatory_content:
            raise HTTPException(
                status_code=400,
                detail=(
                    "The regulatory document is empty."
                ),
            )

        if not policy_content:
            raise HTTPException(
                status_code=400,
                detail="The policy document is empty.",
            )

        with tempfile.NamedTemporaryFile(
            suffix=".pdf",
            delete=False,
        ) as regulatory_temp:
            regulatory_temp.write(
                regulatory_content
            )

            regulatory_path = Path(
                regulatory_temp.name
            )

        with tempfile.NamedTemporaryFile(
            suffix=".pdf",
            delete=False,
        ) as policy_temp:
            policy_temp.write(policy_content)
            policy_path = Path(policy_temp.name)

        regulatory_document_id = uuid4()
        policy_document_id = uuid4()

        parser = PDFParserService()

        regulatory_parsed = parser.parse(
            file_path=regulatory_path,
            document_id=regulatory_document_id,
        )

        policy_parsed = parser.parse(
            file_path=policy_path,
            document_id=policy_document_id,
        )

        chunking_service = ChunkingService(
            chunk_size=1000,
            chunk_overlap=150,
        )

        regulatory_chunks = (
            chunking_service.chunk_document(
                parsed_document=regulatory_parsed,
            )
        )

        policy_chunks = (
            chunking_service.chunk_document(
                parsed_document=policy_parsed,
            )
        )

        if not regulatory_chunks:
            raise HTTPException(
                status_code=422,
                detail=(
                    "No usable text chunks were generated "
                    "from the regulatory document. The PDF "
                    "may require OCR."
                ),
            )

        if not policy_chunks:
            raise HTTPException(
                status_code=422,
                detail=(
                    "No usable text chunks were generated "
                    "from the policy document. The PDF may "
                    "require OCR."
                ),
            )

        requirement_extractor = (
            RuleBasedRequirementExtractionService()
        )

        policy_extractor = (
            RuleBasedPolicyExtractionService()
        )

        gap_comparator = (
            DeterministicGapComparisonService()
        )

        confidence_scorer = (
            DeterministicConfidenceScoringService()
        )

        risk_scorer = (
            DeterministicRiskScoringService()
        )

        explanation_service = (
            DeterministicExplanationService()
        )

        regulatory_requirements = (
            requirement_extractor.extract_from_chunks(
                regulatory_chunks
            )
        )

        if not regulatory_requirements:
            raise HTTPException(
                status_code=422,
                detail=(
                    "No regulatory requirements were "
                    "identified in the uploaded document."
                ),
            )

        embedding_service = (
            SentenceTransformerEmbeddingService(
                model_name=EMBEDDING_MODEL_NAME,
                batch_size=32,
            )
        )

        policy_texts = [
            chunk.text
            for chunk in policy_chunks
        ]

        policy_embeddings = (
            embedding_service.embed_texts(
                policy_texts
            )
        )

        policy_vector_store = FaissVectorStore(
            dimension=embedding_service.dimension,
        )

        policy_vector_store.add(
            chunks=policy_chunks,
            embeddings=policy_embeddings,
        )

        if (
            policy_vector_store.size
            != len(policy_chunks)
        ):
            raise RuntimeError(
                "FAISS index size does not match the "
                "policy chunk count."
            )

        processed_requirements = []
        extracted_policy_statements = []
        gap_assessments = []
        confidence_assessments = []
        risk_assessments = []
        explanations = []

        regulatory_impact = RegulatoryImpact.HIGH

        data_sensitivity = (
            DataSensitivity.CONFIDENTIAL
        )

        for requirement in regulatory_requirements[:20]:
            requirement_embedding = (
                embedding_service.embed_query(
                    requirement.source_text
                )
            )

            matches = policy_vector_store.search(
                query_embedding=(
                    requirement_embedding
                ),
                top_k=5,
            )

            retrieval_scores = {
                match.chunk.chunk_id: (
                    match.similarity_score
                )
                for match in matches
            }

            retrieved_policy_chunks = [
                match.chunk
                for match in matches
            ]

            policy_statements = (
                policy_extractor.extract_from_chunks(
                    retrieved_policy_chunks
                )
            )

            gap_assessment = (
                gap_comparator.compare(
                    requirement=requirement,
                    policy_statements=(
                        policy_statements
                    ),
                )
            )

            confidence_assessment = (
                confidence_scorer.score(
                    requirement=requirement,
                    gap_assessment=(
                        gap_assessment
                    ),
                    policy_statements=(
                        policy_statements
                    ),
                    retrieval_scores=(
                        retrieval_scores
                    ),
                )
            )

            risk_assessment = risk_scorer.score(
                requirement=requirement,
                gap_assessment=gap_assessment,
                confidence_assessment=(
                    confidence_assessment
                ),
                regulatory_impact=(
                    regulatory_impact
                ),
                data_sensitivity=data_sensitivity,
            )

            explanation = (
                explanation_service.explain(
                    requirement=requirement,
                    gap_assessment=(
                        gap_assessment
                    ),
                    confidence_assessment=(
                        confidence_assessment
                    ),
                    risk_assessment=(
                        risk_assessment
                    ),
                )
            )

            processed_requirements.append(
                requirement
            )

            extracted_policy_statements.extend(
                policy_statements
            )

            gap_assessments.append(
                gap_assessment
            )

            confidence_assessments.append(
                confidence_assessment
            )

            risk_assessments.append(
                risk_assessment
            )

            explanations.append(explanation)

        analysis_result = AnalysisResult(
            regulatory_document_id=(
                regulatory_document_id
            ),
            policy_document_id=(
                policy_document_id
            ),
            requirements=processed_requirements,
            policy_statements=(
                extracted_policy_statements
            ),
            gap_assessments=gap_assessments,
            confidence_assessments=(
                confidence_assessments
            ),
            risk_assessments=risk_assessments,
            explanations=explanations,
        )

        # Save the analysis so review endpoints can access it.
        analysis_results_store[
            analysis_result.analysis_id
        ] = analysis_result

        human_reviews_store[
            analysis_result.analysis_id
        ] = []

        compliance_report = report_service.generate(
            analysis_result=analysis_result,
        )

        return build_concise_analysis_response(
            compliance_report=compliance_report,
            reviews=[],
        )

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Document processing failed: {exc}"
            ),
        ) from exc

    finally:
        await regulatory_document.close()
        await policy_document.close()

        if regulatory_path is not None:
            regulatory_path.unlink(
                missing_ok=True
            )

        if policy_path is not None:
            policy_path.unlink(
                missing_ok=True
            )


# ---------------------------------------------------------------------------
# Analysis result endpoint
# ---------------------------------------------------------------------------

@app.get(
    "/analyses/{analysis_id}",
    tags=["Analysis"],
    summary="Get an analysis and its latest reviews",
)
def get_analysis(
    analysis_id: UUID,
) -> dict[str, object]:
    analysis_result = get_analysis_result(
        analysis_id
    )

    reviews = get_analysis_reviews(
        analysis_id
    )

    compliance_report = report_service.generate(
        analysis_result=analysis_result,
    )

    return build_concise_analysis_response(
        compliance_report=compliance_report,
        reviews=reviews,
    )


# ---------------------------------------------------------------------------
# Human-review endpoints
# ---------------------------------------------------------------------------

@app.post(
    "/analyses/{analysis_id}/reviews",
    tags=["Human Review"],
    summary="Create a pending human review",
)
def create_human_review(
    analysis_id: UUID,
    request: CreateHumanReviewRequest,
) -> dict[str, object]:
    analysis_result = get_analysis_result(
        analysis_id
    )

    reviews = get_analysis_reviews(
        analysis_id
    )

    for existing_review in reviews:
        if (
            existing_review.requirement_id
            == request.requirement_id
        ):
            raise HTTPException(
                status_code=409,
                detail=(
                    "A human review already exists for "
                    "this requirement."
                ),
            )

    gap_assessment = (
        get_requirement_gap_assessment(
            analysis_result=analysis_result,
            requirement_id=request.requirement_id,
        )
    )

    review = (
        human_review_service.create_pending_review(
            gap_assessment_id=(
                gap_assessment.assessment_id
            ),
            requirement_id=(
                request.requirement_id
            ),
            original_gap_status=(
                gap_assessment.status
            ),
        )
    )

    reviews.append(review)

    return {
        "status": "review_created",
        "analysis_id": str(analysis_id),
        "review": review.model_dump(
            mode="json"
        ),
    }


@app.put(
    (
        "/analyses/{analysis_id}/reviews/"
        "{requirement_id}"
    ),
    tags=["Human Review"],
    summary="Complete a pending human review",
)
def complete_human_review(
    analysis_id: UUID,
    requirement_id: UUID,
    request: CompleteHumanReviewRequest,
) -> dict[str, object]:
    reviews = get_analysis_reviews(
        analysis_id
    )

    existing_review = (
        get_review_by_requirement(
            analysis_id=analysis_id,
            requirement_id=requirement_id,
        )
    )

    try:
        completed_review = (
            human_review_service.complete_review(
                review=existing_review,
                status=request.status,
                decision=request.decision,
                reviewer_id=request.reviewer_id,
                reviewer_notes=(
                    request.reviewer_notes
                ),
                overridden_gap_status=(
                    request.overridden_gap_status
                ),
            )
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    review_index = reviews.index(
        existing_review
    )

    reviews[review_index] = completed_review

    effective_gap_status = (
        completed_review.overridden_gap_status
        or completed_review.original_gap_status
    )

    return {
        "status": "review_completed",
        "analysis_id": str(analysis_id),
        "review": completed_review.model_dump(
            mode="json"
        ),
        "effective_gap_status": (
            effective_gap_status.value
        ),
    }


@app.get(
    "/analyses/{analysis_id}/reviews",
    tags=["Human Review"],
    summary="List reviews for an analysis",
)
def list_human_reviews(
    analysis_id: UUID,
) -> dict[str, object]:
    reviews = get_analysis_reviews(
        analysis_id
    )

    return {
        "analysis_id": str(analysis_id),
        "review_count": len(reviews),
        "reviews": [
            review.model_dump(mode="json")
            for review in reviews
        ],
    }