```rust
# AI-Assisted Regulatory Policy Gap Analyzer

An end-to-end document AI prototype that compares regulatory requirements against an organisation’s internal policies and identifies potential compliance gaps.

The application extracts requirements from regulatory PDFs, retrieves relevant policy evidence using semantic search, evaluates the level of compliance, assigns confidence and risk scores, generates explainable recommendations, and supports human review of automated findings.

> **Status:** Working prototype / MVP
> **Purpose:** Portfolio, research and demonstration project
> **Important:** This application is intended to assist human reviewers. It should not be treated as legal or regulatory advice.

---

## Demo

### Video demonstration

Watch the project walkthrough:

**Loom Demo:** [https://drive.google.com/file/d/1clFbZiC2lzcDM_SXaPHA2Bm-dhhx_1O0/view?usp=sharing]


```

rather than:

### Human-in-the-loop review

A reviewer can:

* accept the automated assessment;
* override the gap status;
* request additional evidence;
* escalate a finding;
* add reviewer notes;
* record reviewer identity;
* view the effective status after review.

The human review workflow helps prevent automated decisions from being treated as final conclusions in a compliance-sensitive context.

### Streamlit interface

The frontend provides:

* document upload controls;
* analysis summary metrics;
* individual compliance findings;
* confidence and risk information;
* explanations and recommendations;
* human review controls.

### FastAPI backend

The backend provides endpoints for:

* system health checks;
* AI model health checks;
* document analysis;
* retrieving previous in-memory analyses;
* creating human reviews;
* completing reviews;
* listing reviews.

Interactive API documentation is available through Swagger UI.

---

## System Architecture

---

## Technology Stack

### Backend

* Python
* FastAPI
* Pydantic
* Uvicorn

### Frontend

* Streamlit

### Document processing

* PDF parsing service
* Custom text chunking service

### AI and semantic retrieval

* Sentence Transformers
* `all-MiniLM-L6-v2`
* FAISS
* Vector embeddings
* Semantic similarity search

### Analysis

* Rule-based requirement extraction
* Rule-based policy statement extraction
* Deterministic gap comparison
* Deterministic confidence scoring
* Deterministic risk scoring
* Explainable recommendation generation

---

## Why Deterministic Evaluation?

The prototype uses embeddings for semantic retrieval but primarily uses deterministic rules for the final gap, confidence and risk evaluations.

This approach was chosen because it provides:

* repeatable results;
* easier debugging;
* clearer explanations;
* better testability;
* reduced dependence on opaque model outputs;
* more control over compliance-related decisions.

A future version may use a large language model as an optional secondary reasoning layer, while retaining deterministic validation and human review.

---

## Project Structure

The exact structure may differ slightly depending on the current repository version.

Update this section so that it exactly matches your repository.

---

## Installation

### 1. Clone the repository

```sh
git clone https://github.com/sriharan0804/Policy_Gap_Analyzer.git
cd Policy_Gap_Analyzer
```

### 2. Create a virtual environment

#### Windows

```sh
python -m venv .venv
.venv\Scripts\activate
```

#### macOS or Linux

```sh
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```sh
pip install -r requirements.txt
```

The Sentence Transformer model may be downloaded automatically the first time the application runs.

---

## Running the Application

The backend and frontend should be started in separate terminals.

### Start the FastAPI backend

From the project root:

```sh
uvicorn backend.api.app:app --reload
```

The backend should be available at:

Swagger API documentation:

### Start the Streamlit frontend

Open another terminal, activate the same virtual environment and run:

```sh
streamlit run frontend/streamlit_app.py
```

Update the path if your Streamlit file has a different name or location.

The frontend will usually open at:

---

## How to Use

1. Start the FastAPI backend.
2. Start the Streamlit frontend.
3. Open the Streamlit application.
4. Upload a regulatory PDF.
5. Upload an internal policy PDF.
6. Click the analysis button.
7. Review the summary metrics.
8. Open individual findings.
9. Examine the evidence, gap status, confidence, risk and recommendation.
10. Create a human review for findings that require verification.
11. Accept or override the automated assessment.

---

## API Endpoints

### Health check

```http
GET /health
```

Checks whether the FastAPI application is running.

### AI health check

```http
GET /health/ai
```

Attempts to load the embedding model and create a test vector.

### Analyze documents

```http
POST /analyze
```

Accepts:

* `regulatory_document`
* `policy_document`

Both files must be PDFs.

### Get analysis

```http
GET /analyses/{analysis_id}
```

Returns an existing in-memory analysis with its latest review information.

### Create a pending human review

```http
POST /analyses/{analysis_id}/reviews
```

Example request:

```json
{
  "requirement_id": "REQUIREMENT_UUID"
}
```

### Complete a human review

```http
PUT /analyses/{analysis_id}/reviews/{requirement_id}
```

Example: accept the automated result.

```json
{
  "status": "approved",
  "decision": "accept_automated_result",
  "reviewer_id": "reviewer-name",
  "reviewer_notes": "The automated result is supported by the available evidence.",
  "overridden_gap_status": null
}
```

Example: override the automated result.

```json
{
  "status": "approved",
  "decision": "override_gap_status",
  "reviewer_id": "reviewer-name",
  "reviewer_notes": "Additional evidence confirms that the requirement is fully addressed.",
  "overridden_gap_status": "fully_addressed"
}
```

### List human reviews

```http
GET /analyses/{analysis_id}/reviews
```

---

## Example Output

A simplified finding may look like this:

```json
{
  "requirement_summary": "The organisation must retain audit records for at least seven years.",
  "policy_summary": "The internal policy requires audit records to be retained for five years.",
  "gap_status": "partially_addressed",
  "effective_gap_status": "partially_addressed",
  "confidence_score": 0.91,
  "confidence_level": "high",
  "risk_score": 0.82,
  "risk_level": "high",
  "recommended_action": "Increase the audit record retention period from five years to at least seven years.",
  "requires_human_review": true,
  "review_status": "pending"
}
```

---

## Design Decisions

### Embeddings instead of keyword-only search

Keyword matching may miss evidence when two documents describe the same concept using different terminology.

Embeddings allow the application to retrieve policy statements based on semantic similarity.

### Separate confidence and risk scores

Confidence and risk measure different things.

* **Confidence** estimates how reliable the automated conclusion is.
* **Risk** estimates how important or harmful the identified compliance gap may be.

A result may have high risk but low confidence, meaning that it is potentially serious and should be reviewed carefully.

### Human review as a core feature

Automated analysis may miss organisational context, exceptions, evidence stored outside the uploaded document or legal interpretation.

The application therefore treats human review as part of the normal workflow rather than an optional afterthought.

### Explainability

Every automated finding should provide enough reasoning for a reviewer to understand why the system reached its conclusion.

The project therefore returns explanations, evidence and remediation suggestions instead of only returning a classification label.

---

## Current Limitations

This project is a prototype and has several limitations:

* Analyses and reviews are currently stored in memory.
* Data is lost when the FastAPI server restarts.
* Scanned PDFs may require OCR before they can be analysed.
* Rule-based extraction may miss complex or indirectly written requirements.
* Semantic similarity does not guarantee legal equivalence.
* The system evaluates only the content available in the uploaded documents.
* Risk and confidence values are heuristic scores, not legally validated measurements.
* The prototype does not replace professional compliance or legal review.
* Large documents may require additional performance optimisation.
* Authentication and role-based access control have not yet been implemented.

---

## Planned Improvements

* SQLite or PostgreSQL persistence
* Downloadable PDF compliance reports
* Excel and JSON report export
* Search, sorting and filtering
* Improved analytics dashboard
* OCR support for scanned documents
* Automated tests with `pytest`
* Authentication and reviewer roles
* Historical analysis records
* Policy version comparison
* Background processing for large documents
* Docker support
* Cloud deployment
* Optional LLM-assisted analysis
* Citation-level evidence highlighting

---

## Testing

Automated tests are planned for:

* PDF validation;
* text chunking;
* requirement extraction;
* policy extraction;
* vector retrieval;
* gap comparison;
* confidence scoring;
* risk scoring;
* explanation generation;
* human review validation;
* API endpoints.

When the test suite is available, run:

```sh
pytest
```

---

## Responsible Use

This project is designed as an AI-assisted decision-support system.

Users should:

* verify findings against the original documents;
* involve qualified compliance or legal professionals;
* avoid treating automated scores as final legal conclusions;
* protect confidential documents;
* review generated recommendations before implementation.

The developer assumes no responsibility for decisions made solely from the automated output.

---

## What I Learned

Through this project, I explored how to build a complete document AI pipeline rather than only using an individual machine-learning model.

The main areas I worked on include:

* document parsing and chunking;
* structured information extraction;
* vector embeddings;
* semantic retrieval;
* FAISS indexing;
* deterministic reasoning;
* confidence and risk modelling;
* explainable AI output;
* API development with FastAPI;
* frontend development with Streamlit;
* human-in-the-loop system design.

---

## Author

**Sriharan**

* GitHub: [Add your GitHub profile link]
* LinkedIn: [Add your LinkedIn profile link]
* Email: [Add a professional email address]

---

## Contributing

This is currently a personal prototype, but suggestions, issues and constructive feedback are welcome.

To contribute:

1. Fork the repository.
2. Create a new branch.
3. Make your changes.
4. Add or update tests where appropriate.
5. Submit a pull request explaining the change.

---

## License

Add a licence before encouraging external reuse.

For an open-source portfolio project, the MIT License is a common option.

After adding a `LICENSE` file, replace this section with:

---

## Acknowledgements

This project uses open-source technologies including:

* FastAPI
* Streamlit
* Sentence Transformers
* FAISS
* Pydantic

AI-assisted development tools were also used during implementation for brainstorming, boilerplate generation, debugging support and code review. The architecture, integration, testing and project decisions were developed and validated as part of the project-building process.

```