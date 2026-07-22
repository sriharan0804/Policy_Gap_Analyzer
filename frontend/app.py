import streamlit as st
import requests


API_URL = "http://localhost:8000/analyze"


st.set_page_config(
    page_title="Regulatory Policy Gap Analyzer",
    page_icon="📋",
    layout="wide",
)


def format_percentage(value: float) -> str:
    return f"{value * 100:.1f}%"


def risk_badge(risk_level: str) -> str:
    badges = {
        "low": "🟢 Low",
        "medium": "🟡 Medium",
        "high": "🟠 High",
        "critical": "🔴 Critical",
    }

    return badges.get(
        risk_level.lower(),
        risk_level.title(),
    )


def status_badge(status: str) -> str:
    badges = {
        "fully_addressed": "✅ Fully Addressed",
        "partially_addressed": "⚠️ Partially Addressed",
        "not_addressed": "❌ Not Addressed",
        "contradicted": "🚫 Contradicted",
        "insufficient_evidence": "❓ Insufficient Evidence",
    }

    return badges.get(
        status.lower(),
        status.replace("_", " ").title(),
    )


st.title("AI-Assisted Regulatory Policy Gap Analyzer")

st.write(
    "Upload a regulatory document and an internal policy document "
    "to identify compliance gaps."
)

with st.sidebar:
    st.header("Analysis Settings")

    api_url = st.text_input(
        "Backend API URL",
        value=API_URL,
    )

    st.info(
        "Start the FastAPI backend before running the analysis."
    )


regulatory_document = st.file_uploader(
    "Upload regulatory document",
    type=["pdf"],
    key="regulatory_document",
)

policy_document = st.file_uploader(
    "Upload internal policy document",
    type=["pdf"],
    key="policy_document",
)


analyze_clicked = st.button(
    "Analyze Documents",
    type="primary",
    use_container_width=True,
)


if analyze_clicked:
    if regulatory_document is None:
        st.error("Please upload a regulatory PDF.")

    elif policy_document is None:
        st.error("Please upload an internal policy PDF.")

    else:
        files = {
            "regulatory_document": (
                regulatory_document.name,
                regulatory_document.getvalue(),
                "application/pdf",
            ),
            "policy_document": (
                policy_document.name,
                policy_document.getvalue(),
                "application/pdf",
            ),
        }

        try:
            with st.spinner(
                "Analyzing documents. This may take a moment..."
            ):
                response = requests.post(
                    api_url,
                    files=files,
                    timeout=300,
                )

            if response.status_code != 200:
                try:
                    error_detail = response.json().get(
                        "detail",
                        response.text,
                    )
                except ValueError:
                    error_detail = response.text

                st.error(
                    f"Analysis failed: {error_detail}"
                )

            else:
                result = response.json()
                st.session_state["analysis_result"] = result

        except requests.exceptions.ConnectionError:
            st.error(
                "Could not connect to the backend. "
                "Make sure FastAPI is running on port 8000."
            )

        except requests.exceptions.Timeout:
            st.error(
                "The analysis request timed out."
            )

        except requests.exceptions.RequestException as exc:
            st.error(
                f"Request failed: {exc}"
            )


result = st.session_state.get("analysis_result")

if result:
    summary = result["summary"]
    findings = result["findings"]

    st.divider()
    st.subheader("Analysis Summary")

    first_row = st.columns(4)

    first_row[0].metric(
        "Compliance Score",
        format_percentage(
            summary["compliance_score"]
        ),
    )

    first_row[1].metric(
        "Total Requirements",
        summary["total_requirements"],
    )

    first_row[2].metric(
        "Fully Addressed",
        summary["fully_addressed_count"],
    )

    first_row[3].metric(
        "Partially Addressed",
        summary["partially_addressed_count"],
    )

    second_row = st.columns(4)

    second_row[0].metric(
        "Not Addressed",
        summary["not_addressed_count"],
    )

    second_row[1].metric(
        "Contradicted",
        summary["contradicted_count"],
    )

    second_row[2].metric(
        "High Risk",
        summary["high_risk_count"],
    )

    second_row[3].metric(
        "Human Review Required",
        summary["human_review_required_count"],
    )

    st.divider()
    st.subheader("Compliance Findings")

    for index, finding in enumerate(findings, start=1):
        title = (
            f"{index}. "
            f"{status_badge(finding['effective_gap_status'])}"
            f" — {risk_badge(finding['risk_level'])}"
        )

        with st.expander(
            title,
            expanded=index == 1,
        ):
            st.markdown("#### Regulatory Requirement")
            st.write(
                finding["requirement_summary"]
            )

            st.markdown("#### Internal Policy Evidence")
            st.write(
                finding["policy_summary"]
            )

            detail_columns = st.columns(3)

            detail_columns[0].metric(
                "Confidence",
                format_percentage(
                    finding["confidence_score"]
                ),
                finding["confidence_level"].title(),
            )

            detail_columns[1].metric(
                "Risk Score",
                format_percentage(
                    finding["risk_score"]
                ),
                finding["risk_level"].title(),
            )

            detail_columns[2].metric(
                "Review Status",
                finding["review_status"].title(),
            )

            st.markdown("#### Gap Explanation")
            st.warning(
                finding["gap_reason"]
            )

            st.markdown("#### Recommended Action")
            st.success(
                finding["recommended_action"]
            )

            if finding["requires_human_review"]:
                st.info(
                    "This finding requires human review."
                )

            st.caption(
                f"Requirement ID: "
                f"{finding['requirement_id']}"
            )

    st.divider()

    st.caption(
        f"Analysis ID: {result['analysis_id']} | "
        f"Report ID: {result['report_id']} | "
        f"Report Version: {result['report_version']}"
    )