import json
import tempfile
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from backend.services.chunking_service import ChunkingService
from backend.services.pdf_parser_service import PDFParserService
from backend.services.embedding_service import (
    SentenceTransformerEmbeddingService,
)
from backend.services.requirement_extraction_service import (
    RuleBasedRequirementExtractionService,
)
from backend.services.policy_extraction_service import (
    RuleBasedPolicyExtractionService,
)
from backend.services.vector_store import FaissVectorStore
from backend.models import (
    AnalysisResult,
    DataSensitivity,
    RegulatoryImpact,
)
from backend.services.gap_comparison_service import (
    DeterministicGapComparisonService,
)
from backend.services.confidence_scoring_service import (
    DeterministicConfidenceScoringService,
)
from backend.services.risk_scoring_service import (
    DeterministicRiskScoringService,
)
from backend.services.explanation_service import (
    DeterministicExplanationService,
)
from backend.services.report_service import DeterministicReportService
from backend.services.report_export_service import (
    DeterministicReportExportService,
)


BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"

ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "application/x-pdf",
}

STATIC_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

app = FastAPI(
    title="AI-Assisted Regulatory Policy Gap Analyzer",
    description=(
        "Compares regulatory requirements against internal policies "
        "and identifies compliance gaps."
    ),
    version="1.0.0",
)


app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static",
)


def validate_pdf(
    upload: UploadFile,
    field_name: str,
) -> None:
    """Validate the uploaded file extension and MIME type."""

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


@app.get(
    "/health",
    tags=["System"],
)
def health_check() -> dict[str, str]:
    """Check whether the FastAPI application is running."""

    return {
        "status": "healthy",
        "service": "policy-gap-analyzer",
    }


@app.get(
    "/health/ai",
    tags=["System"],
    summary="Check whether the AI embedding model can run",
)
def ai_health_check() -> dict[str, object]:
    """Load the embedding model and generate one test vector."""

    from backend.services.embedding_service import (
        SentenceTransformerEmbeddingService,
    )

    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    try:
        embedding_service = SentenceTransformerEmbeddingService(
            model_name=model_name,
        )

        embeddings = embedding_service.embed_texts(
            [
                "Regulatory policy gap analyzer readiness check.",
            ]
        )

        return {
            "status": "healthy",
            "ai_model_loaded": True,
            "model_name": model_name,
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
    """Display a basic PDF upload page."""

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
                    margin-bottom: 16px;
                    font-size: 2rem;
                    line-height: 1.2;
                }

                .description {
                    margin-bottom: 32px;
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
                    background: #f8fafc;
                }

                button {
                    width: 100%;
                    margin-top: 10px;
                    padding: 14px 20px;
                    border: none;
                    border-radius: 10px;
                    font-size: 1rem;
                    font-weight: 600;
                    cursor: pointer;
                    color: white;
                    background: #315efb;
                }

                button:hover {
                    background: #244bd1;
                }

                .note {
                    margin-top: 18px;
                    font-size: 0.9rem;
                    color: #6b7280;
                    text-align: center;
                }
            </style>
        </head>

        <body>
            <main>
                <h1>
                    AI-Assisted Regulatory Policy Gap Analyzer
                </h1>

                <p class="description">
                    Upload a regulatory document and an internal policy
                    document to identify missing, partial, or contradictory
                    compliance controls.
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
                        Parse Documents
                    </button>
                </form>

                <p class="note">
                    PDF documents only. Uploaded files are deleted
                    after processing.
                </p>
            </main>
        </body>
        </html>
        """,
    )

def _build_concise_analysis_response(
    compliance_report: object,
) -> dict[str, object]:
    """Build the concise client-facing response used by Streamlit."""

    report_data = compliance_report.model_dump(mode="json")

    findings: list[dict[str, object]] = []

    for report in report_data["requirement_reports"]:
        findings.append(
            {
                "requirement_id": report["requirement_id"],
                "gap_assessment_id": report["gap_assessment_id"],
                "requirement_summary": report["requirement_summary"],
                "policy_summary": report["policy_summary"],
                "gap_status": report["gap_status"],
                "effective_gap_status": report[
                    "effective_gap_status"
                ],
                "gap_reason": report["gap_reason"],
                "confidence_score": report["confidence_score"],
                "confidence_level": report["confidence_level"],
                "risk_score": report["risk_score"],
                "risk_level": report["risk_level"],
                "recommended_action": report[
                    "recommended_action"
                ],
                "requires_human_review": report[
                    "requires_human_review"
                ],
                "review_status": report["review_status"],
                "reviewer_decision": report["reviewer_decision"],
            }
        )

    return {
        "status": "report_generated",
        "analysis_id": report_data["analysis_id"],
        "report_id": report_data["report_id"],
        "summary": report_data["summary"],
        "findings": findings,
        "generated_at": report_data["generated_at"],
        "report_version": report_data["report_version"],
    }

@app.post(
    "/analyze",
    tags=["Analysis"],
    summary="Parse and chunk regulatory and policy PDF documents",
)
async def analyze_documents(
    regulatory_document: UploadFile = File(
        ...,
        description="Regulatory document in PDF format.",
    ),
    policy_document: UploadFile = File(
        ...,
        description="Internal policy document in PDF format.",
    ),
) -> dict[str, object]:
    """
    Validate, temporarily store, parse and chunk both uploaded PDFs.

    Temporary files are always deleted after processing.
    """

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
        regulatory_content = await regulatory_document.read()
        policy_content = await policy_document.read()

        if not regulatory_content:
            raise HTTPException(
                status_code=400,
                detail="The regulatory document is empty.",
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
            regulatory_temp.write(regulatory_content)
            regulatory_path = Path(regulatory_temp.name)

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

        regulatory_chunks = chunking_service.chunk_document(
            parsed_document=regulatory_parsed,
        )

        policy_chunks = chunking_service.chunk_document(
            parsed_document=policy_parsed,
        )

        if not regulatory_chunks:
            raise HTTPException(
                status_code=422,
                detail=(
                    "No usable text chunks were generated from the "
                    "regulatory document. The PDF may require OCR."
                ),
            )

        if not policy_chunks:
            raise HTTPException(
                status_code=422,
                detail=(
                    "No usable text chunks were generated from the "
                    "policy document. The PDF may require OCR."
                ),
            )
        requirement_extractor = RuleBasedRequirementExtractionService()
        policy_extractor = RuleBasedPolicyExtractionService()
        gap_comparator = DeterministicGapComparisonService()
        confidence_scorer = DeterministicConfidenceScoringService()
        risk_scorer = DeterministicRiskScoringService()
        explanation_service = DeterministicExplanationService()
        report_service = DeterministicReportService()
        report_export_service = DeterministicReportExportService()

        regulatory_requirements = requirement_extractor.extract_from_chunks(
            regulatory_chunks
        )

        embedding_service = SentenceTransformerEmbeddingService(
            model_name=EMBEDDING_MODEL_NAME,
            batch_size=32,
        )

        regulatory_texts = [chunk.text for chunk in regulatory_chunks]
        policy_texts = [chunk.text for chunk in policy_chunks]

        regulatory_embeddings = embedding_service.embed_texts(
            regulatory_texts
        )
        policy_embeddings = embedding_service.embed_texts(
            policy_texts
        )

        policy_vector_store = FaissVectorStore(
            dimension=embedding_service.dimension,
        )
        policy_vector_store.add(
            chunks=policy_chunks,
            embeddings=policy_embeddings,
        )

        if policy_vector_store.size != len(policy_chunks):
            raise RuntimeError(
                "FAISS index size does not match the policy chunk count."
            )

        requirement_retrieval_results: list[dict[str, object]] = []
        processed_requirements = []
        extracted_policy_statements = []
        gap_assessments = []
        confidence_assessments = []
        risk_assessments = []
        explanations = []

        # Temporary demo context. Replace these defaults with user input or
        # document classification when that capability is added.
        regulatory_impact = RegulatoryImpact.HIGH
        data_sensitivity = DataSensitivity.CONFIDENTIAL

        for requirement in regulatory_requirements[:20]:
            requirement_embedding = embedding_service.embed_query(
                requirement.source_text
            )

            matches = policy_vector_store.search(
                query_embedding=requirement_embedding,
                top_k=5,
            )

            retrieval_scores = {
                match.chunk.chunk_id: match.similarity_score
                for match in matches
            }

            retrieved_policy_chunks = [
                match.chunk
                for match in matches
            ]

            policy_statements = policy_extractor.extract_from_chunks(
                retrieved_policy_chunks
            )

            gap_assessment = gap_comparator.compare(
                requirement=requirement,
                policy_statements=policy_statements,
            )

            confidence_assessment = confidence_scorer.score(
                requirement=requirement,
                gap_assessment=gap_assessment,
                policy_statements=policy_statements,
                retrieval_scores=retrieval_scores,
            )

            risk_assessment = risk_scorer.score(
                requirement=requirement,
                gap_assessment=gap_assessment,
                confidence_assessment=confidence_assessment,
                regulatory_impact=regulatory_impact,
                data_sensitivity=data_sensitivity,
            )

            explanation = explanation_service.explain(
                requirement=requirement,
                gap_assessment=gap_assessment,
                confidence_assessment=confidence_assessment,
                risk_assessment=risk_assessment,
            )
            processed_requirements.append(requirement)
            extracted_policy_statements.extend(policy_statements)
            gap_assessments.append(gap_assessment)
            confidence_assessments.append(confidence_assessment)
            risk_assessments.append(risk_assessment)
            explanations.append(explanation)

            requirement_retrieval_results.append(
                {
                    "requirement": {
                        "requirement_id": str(requirement.requirement_id),
                        "source_text": requirement.source_text,
                        "subject": requirement.subject,
                        "action": requirement.action,
                        "object": requirement.object,
                        "condition": requirement.condition,
                        "timing": requirement.timing,
                        "modality": requirement.modality.value,
                        "matched_trigger": requirement.matched_trigger,
                        "extraction_confidence": round(
                            requirement.extraction_confidence,
                            4,
                        ),
                        "source": {
                            "page_number": requirement.page_number,
                            "chunk_index": requirement.chunk_index,
                            "chunk_id": str(requirement.chunk_id),
                        },
                    },
                    "retrieved_policy_chunks": [
                        {
                            "rank": match.rank,
                            "similarity_score": round(
                                match.similarity_score,
                                4,
                            ),
                            "chunk_id": str(match.chunk.chunk_id),
                            "chunk_index": match.chunk.chunk_index,
                            "page_number": match.chunk.page_number,
                            "text_preview": match.chunk.text[:300],
                        }
                        for match in matches
                    ],
                    "policy_statement_count": len(policy_statements),
                    "policy_statements": [
                        {
                            "statement_id": str(statement.statement_id),
                            "source_text": statement.source_text,
                            "subject": statement.subject,
                            "action": statement.action,
                            "object": statement.object,
                            "condition": statement.condition,
                            "timing": statement.timing,
                            "responsible_party": (
                                statement.responsible_party
                            ),
                            "statement_type": (
                                statement.statement_type.value
                            ),
                            "matched_trigger": statement.matched_trigger,
                            "extraction_confidence": round(
                                statement.extraction_confidence,
                                4,
                            ),
                            "source": {
                                "page_number": statement.page_number,
                                "chunk_index": statement.chunk_index,
                                "chunk_id": str(statement.chunk_id),
                            },
                        }
                        for statement in policy_statements
                    ],
                    "gap_assessment": {
                        "assessment_id": str(
                            gap_assessment.assessment_id
                        ),
                        "status": gap_assessment.status.value,
                        "deterministic_score": round(
                            gap_assessment.deterministic_score,
                            4,
                        ),
                        "evaluated_policy_count": (
                            gap_assessment.evaluated_policy_count
                        ),
                        "requires_human_review": (
                            gap_assessment.requires_human_review
                        ),
                        "rationale": gap_assessment.rationale,
                        "best_match": (
                            {
                                "policy_statement_id": str(
                                    gap_assessment
                                    .best_match
                                    .policy_statement_id
                                ),
                                "policy_chunk_id": str(
                                    gap_assessment
                                    .best_match
                                    .policy_chunk_id
                                ),
                                "overall_score": round(
                                    gap_assessment
                                    .best_match
                                    .components
                                    .overall_score,
                                    4,
                                ),
                                "is_contradiction": (
                                    gap_assessment
                                    .best_match
                                    .is_contradiction
                                ),
                                "reasons": (
                                    gap_assessment.best_match.reasons
                                ),
                            }
                            if gap_assessment.best_match is not None
                            else None
                        ),
                    },
                    "confidence_assessment": {
                        "confidence_score": round(
                            confidence_assessment.confidence_score,
                            4,
                        ),
                        "confidence_level": (
                            confidence_assessment
                            .confidence_level
                            .value
                        ),
                        "supporting_evidence_count": (
                            confidence_assessment
                            .supporting_evidence_count
                        ),
                        "requires_human_review": (
                            confidence_assessment
                            .requires_human_review
                        ),
                        "positive_factors": (
                            confidence_assessment.positive_factors
                        ),
                        "limiting_factors": (
                            confidence_assessment.limiting_factors
                        ),
                        "components": {
                            "requirement_extraction_score": round(
                                confidence_assessment
                                .components
                                .requirement_extraction_score,
                                4,
                            ),
                            "policy_extraction_score": round(
                                confidence_assessment
                                .components
                                .policy_extraction_score,
                                4,
                            ),
                            "retrieval_score": round(
                                confidence_assessment
                                .components
                                .retrieval_score,
                                4,
                            ),
                            "comparison_score": round(
                                confidence_assessment
                                .components
                                .comparison_score,
                                4,
                            ),
                            "evidence_completeness_score": round(
                                confidence_assessment
                                .components
                                .evidence_completeness_score,
                                4,
                            ),
                            "evidence_quantity_score": round(
                                confidence_assessment
                                .components
                                .evidence_quantity_score,
                                4,
                            ),
                        },
                    },
                    "risk_assessment": {
                        "risk_score": round(
                            risk_assessment.risk_score,
                            4,
                        ),
                        "risk_level": risk_assessment.risk_level.value,
                        "remediation_priority": (
                            risk_assessment.remediation_priority
                        ),
                        "regulatory_impact": (
                            risk_assessment.regulatory_impact.value
                        ),
                        "data_sensitivity": (
                            risk_assessment.data_sensitivity.value
                        ),
                        "requires_human_review": (
                            risk_assessment.requires_human_review
                        ),
                        "risk_factors": risk_assessment.risk_factors,
                        "mitigating_factors": (
                            risk_assessment.mitigating_factors
                        ),
                        "components": {
                            "gap_severity_score": round(
                                risk_assessment
                                .components
                                .gap_severity_score,
                                4,
                            ),
                            "regulatory_impact_score": round(
                                risk_assessment
                                .components
                                .regulatory_impact_score,
                                4,
                            ),
                            "requirement_criticality_score": round(
                                risk_assessment
                                .components
                                .requirement_criticality_score,
                                4,
                            ),
                            "data_sensitivity_score": round(
                                risk_assessment
                                .components
                                .data_sensitivity_score,
                                4,
                            ),
                            "confidence_reliability_score": round(
                                risk_assessment
                                .components
                                .confidence_reliability_score,
                                4,
                            ),
                            "contradiction_score": round(
                                risk_assessment
                                .components
                                .contradiction_score,
                                4,
                            ),
                        },
                    },
                    "explanation": {
                        "requirement_summary": (
                            explanation.requirement_summary
                        ),
                        "policy_summary": explanation.policy_summary,
                        "gap_reason": explanation.gap_reason,
                        "confidence_reason": (
                            explanation.confidence_reason
                        ),
                        "risk_reason": explanation.risk_reason,
                        "recommended_action": (
                            explanation.recommended_action
                        ),
                        "requires_human_review": (
                            explanation.requires_human_review
                        ),
                    },
                }
            )

        analysis_result = AnalysisResult(
            regulatory_document_id=regulatory_document_id,
            policy_document_id=policy_document_id,
            requirements=processed_requirements,
            policy_statements=extracted_policy_statements,
            gap_assessments=gap_assessments,
            confidence_assessments=confidence_assessments,
            risk_assessments=risk_assessments,
            explanations=explanations,
        )

        compliance_report = report_service.generate(
            analysis_result=analysis_result,
        )

       

        return _build_concise_analysis_response(
            compliance_report=compliance_report,
        )

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Document processing failed: {exc}",
        ) from exc

    finally:
        await regulatory_document.close()
        await policy_document.close()

        if regulatory_path is not None:
            regulatory_path.unlink(missing_ok=True)

        if policy_path is not None:
            policy_path.unlink(missing_ok=True)