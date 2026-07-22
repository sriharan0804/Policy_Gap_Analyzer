from __future__ import annotations

import json
import os
from typing import Any

import requests
import streamlit as st


DEFAULT_API_URL = os.getenv("POLICY_GAP_API_URL", "http://127.0.0.1:8000")
REQUEST_TIMEOUT_SECONDS = 600

GAP_STATUS_LABELS = {
    "fully_addressed": "Fully addressed",
    "partially_addressed": "Partially addressed",
    "not_addressed": "Not addressed",
    "contradicted": "Contradicted",
    "insufficient_evidence": "Insufficient evidence",
}

RISK_LABELS = {
    "low": "Low",
    "medium": "Medium",
    "high": "High",
    "critical": "Critical",
}

REVIEW_STATUS_OPTIONS = {
    "Approve": "approved",
    "Reject": "rejected",
    "Needs revision": "needs_revision",
}

REVIEW_DECISION_OPTIONS = {
    "Accept automated result": "accept_automated_result",
    "Override gap status": "override_gap_status",
    "Request more evidence": "request_more_evidence",
    "Escalate": "escalate",
}

OVERRIDE_STATUS_OPTIONS = {
    "Fully addressed": "fully_addressed",
    "Partially addressed": "partially_addressed",
    "Not addressed": "not_addressed",
    "Contradicted": "contradicted",
    "Insufficient evidence": "insufficient_evidence",
}


st.set_page_config(
    page_title="Policy Gap Analyzer",
    page_icon="📋",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def check_backend(api_url: str) -> tuple[bool, str]:
    try:
        response = requests.get(
            f"{api_url.rstrip('/')}/health",
            timeout=10,
        )
        response.raise_for_status()
        payload = response.json()
        return payload.get("status") == "healthy", "Backend connected"
    except requests.RequestException as exc:
        return False, f"Backend unavailable: {exc}"


def api_error_message(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text or f"HTTP {response.status_code}"

    detail = payload.get("detail") if isinstance(payload, dict) else None
    if isinstance(detail, list):
        return "; ".join(str(item) for item in detail)
    return str(detail or payload)


def post_analysis(
    api_url: str,
    regulatory_file: Any,
    policy_file: Any,
) -> dict[str, Any]:
    files = {
        "regulatory_document": (
            regulatory_file.name,
            regulatory_file.getvalue(),
            "application/pdf",
        ),
        "policy_document": (
            policy_file.name,
            policy_file.getvalue(),
            "application/pdf",
        ),
    }

    response = requests.post(
        f"{api_url.rstrip('/')}/analyze",
        files=files,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    if not response.ok:
        raise RuntimeError(api_error_message(response))

    return response.json()


def fetch_report(api_url: str, analysis_id: str) -> dict[str, Any]:
    response = requests.get(
        f"{api_url.rstrip('/')}/analyses/{analysis_id}/report",
        timeout=60,
    )
    if not response.ok:
        raise RuntimeError(api_error_message(response))
    return response.json()


def fetch_reviews(api_url: str, analysis_id: str) -> list[dict[str, Any]]:
    response = requests.get(
        f"{api_url.rstrip('/')}/analyses/{analysis_id}/reviews",
        timeout=30,
    )
    if not response.ok:
        raise RuntimeError(api_error_message(response))
    return response.json().get("reviews", [])


def complete_review(
    api_url: str,
    analysis_id: str,
    review_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    response = requests.put(
        f"{api_url.rstrip('/')}/analyses/{analysis_id}/reviews/{review_id}",
        json=payload,
        timeout=60,
    )
    if not response.ok:
        raise RuntimeError(api_error_message(response))
    return response.json()


def normalize_percentage(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"

    if 0 <= number <= 1:
        number *= 100
    return f"{number:.1f}%"


def status_label(value: str | None) -> str:
    if not value:
        return "Unknown"
    return GAP_STATUS_LABELS.get(value, value.replace("_", " ").title())


def risk_label(value: str | None) -> str:
    if not value:
        return "Unknown"
    return RISK_LABELS.get(value, value.replace("_", " ").title())


def report_from_state() -> dict[str, Any]:
    payload = st.session_state.get("report_payload") or {}
    return payload.get("compliance_report") or {}


def exports_from_state() -> dict[str, Any]:
    payload = st.session_state.get("report_payload") or {}
    return payload.get("report_exports") or {}


def refresh_reviewed_data(api_url: str, analysis_id: str) -> None:
    report_payload = fetch_report(api_url, analysis_id)
    reviews = fetch_reviews(api_url, analysis_id)
    st.session_state.report_payload = report_payload
    st.session_state.reviews = reviews


def render_summary(report: dict[str, Any]) -> None:
    summary = report.get("summary") or {}

    first_row = st.columns(4)
    first_row[0].metric("Requirements", summary.get("total_requirements", 0))
    first_row[1].metric(
        "Compliance score",
        normalize_percentage(summary.get("compliance_score")),
    )
    first_row[2].metric(
        "High / critical risk",
        int(summary.get("high_risk_count", 0) or 0)
        + int(summary.get("critical_risk_count", 0) or 0),
    )
    first_row[3].metric(
        "Human reviews",
        summary.get("human_review_required_count", 0),
    )

    second_row = st.columns(5)
    second_row[0].metric(
        "Fully addressed",
        summary.get("fully_addressed_count", 0),
    )
    second_row[1].metric(
        "Partially addressed",
        summary.get("partially_addressed_count", 0),
    )
    second_row[2].metric(
        "Not addressed",
        summary.get("not_addressed_count", 0),
    )
    second_row[3].metric(
        "Contradicted",
        summary.get("contradicted_count", 0),
    )
    second_row[4].metric(
        "Insufficient evidence",
        summary.get("insufficient_evidence_count", 0),
    )


def render_findings(report: dict[str, Any]) -> None:
    findings = report.get("requirement_reports") or []
    if not findings:
        st.info("No requirement findings were returned.")
        return

    status_filter = st.multiselect(
        "Filter by effective status",
        options=sorted(
            {item.get("effective_gap_status") for item in findings if item.get("effective_gap_status")}
        ),
        format_func=status_label,
    )

    risk_filter = st.multiselect(
        "Filter by risk",
        options=sorted(
            {item.get("risk_level") for item in findings if item.get("risk_level")}
        ),
        format_func=risk_label,
    )

    filtered = [
        item
        for item in findings
        if (not status_filter or item.get("effective_gap_status") in status_filter)
        and (not risk_filter or item.get("risk_level") in risk_filter)
    ]

    st.caption(f"Showing {len(filtered)} of {len(findings)} findings")

    for index, finding in enumerate(filtered, start=1):
        requirement_summary = finding.get("requirement_summary") or "Untitled requirement"
        effective_status = finding.get("effective_gap_status")
        risk = finding.get("risk_level")
        title = (
            f"{index}. {status_label(effective_status)} · "
            f"{risk_label(risk)} risk — {requirement_summary[:110]}"
        )

        with st.expander(title, expanded=index == 1):
            left, right = st.columns([2, 1])

            with left:
                st.markdown("**Regulatory requirement**")
                st.write(requirement_summary)

                st.markdown("**Relevant policy evidence**")
                st.write(
                    finding.get("policy_summary")
                    or "No policy evidence identified."
                )

                st.markdown("**Gap explanation**")
                st.write(finding.get("gap_reason") or "No explanation available.")

                st.markdown("**Recommended action**")
                st.write(
                    finding.get("recommended_action")
                    or "No recommendation available."
                )

            with right:
                st.metric(
                    "Confidence",
                    normalize_percentage(finding.get("confidence_score")),
                )
                st.metric(
                    "Risk",
                    normalize_percentage(finding.get("risk_score")),
                )
                st.write(
                    "**Automated status:**",
                    status_label(finding.get("gap_status")),
                )
                st.write(
                    "**Effective status:**",
                    status_label(effective_status),
                )
                st.write(
                    "**Review status:**",
                    str(finding.get("review_status", "pending")).replace("_", " ").title(),
                )
                st.write(
                    "**Human review required:**",
                    "Yes" if finding.get("requires_human_review") else "No",
                )

            with st.popover("Technical details"):
                st.write("**Confidence reason**")
                st.write(finding.get("confidence_reason") or "—")
                st.write("**Risk reason**")
                st.write(finding.get("risk_reason") or "—")
                st.json(finding)


def render_reviews(api_url: str, analysis_id: str) -> None:
    reviews = st.session_state.get("reviews") or []
    if not reviews:
        st.success("No human reviews are currently required.")
        return

    pending = [item for item in reviews if item.get("status") == "pending"]
    completed = [item for item in reviews if item.get("status") != "pending"]

    st.write(f"Pending: **{len(pending)}** · Completed: **{len(completed)}**")

    for review in pending:
        review_id = str(review.get("review_id"))
        requirement_id = str(review.get("requirement_id"))

        with st.expander(
            f"Review requirement {requirement_id[:8]} · "
            f"{status_label(review.get('original_gap_status'))}",
            expanded=True,
        ):
            with st.form(f"review-form-{review_id}"):
                status_display = st.selectbox(
                    "Review outcome",
                    list(REVIEW_STATUS_OPTIONS),
                    key=f"status-{review_id}",
                )
                decision_display = st.selectbox(
                    "Reviewer decision",
                    list(REVIEW_DECISION_OPTIONS),
                    key=f"decision-{review_id}",
                )
                reviewer_id = st.text_input(
                    "Reviewer name or ID",
                    value="prototype-reviewer",
                    key=f"reviewer-{review_id}",
                )
                reviewer_notes = st.text_area(
                    "Review notes",
                    placeholder="Explain why this finding is accepted, overridden, escalated, or needs more evidence.",
                    key=f"notes-{review_id}",
                )

                override_status = None
                if REVIEW_DECISION_OPTIONS[decision_display] == "override_gap_status":
                    override_display = st.selectbox(
                        "New effective gap status",
                        list(OVERRIDE_STATUS_OPTIONS),
                        key=f"override-{review_id}",
                    )
                    override_status = OVERRIDE_STATUS_OPTIONS[override_display]

                submitted = st.form_submit_button(
                    "Submit review",
                    type="primary",
                    use_container_width=True,
                )

            if submitted:
                if not reviewer_id.strip():
                    st.error("Reviewer name or ID is required.")
                else:
                    payload: dict[str, Any] = {
                        "status": REVIEW_STATUS_OPTIONS[status_display],
                        "decision": REVIEW_DECISION_OPTIONS[decision_display],
                        "reviewer_id": reviewer_id.strip(),
                        "reviewer_notes": reviewer_notes.strip() or None,
                        "overridden_gap_status": override_status,
                    }
                    try:
                        with st.spinner("Saving review and rebuilding report..."):
                            response = complete_review(
                                api_url,
                                analysis_id,
                                review_id,
                                payload,
                            )
                            st.session_state.report_payload = response
                            st.session_state.reviews = fetch_reviews(api_url, analysis_id)
                        st.success("Review saved. The report has been updated.")
                        st.rerun()
                    except RuntimeError as exc:
                        st.error(str(exc))

    if completed:
        st.markdown("#### Completed reviews")
        for review in completed:
            label = str(review.get("status", "completed")).replace("_", " ").title()
            with st.expander(
                f"{label} · requirement {str(review.get('requirement_id'))[:8]}"
            ):
                st.json(review)


def render_downloads(report: dict[str, Any], exports: dict[str, Any]) -> None:
    json_export = exports.get("json") or report
    csv_export = exports.get("csv") or ""

    col1, col2 = st.columns(2)
    col1.download_button(
        "Download JSON report",
        data=json.dumps(json_export, indent=2),
        file_name="compliance_report.json",
        mime="application/json",
        use_container_width=True,
    )
    col2.download_button(
        "Download CSV findings",
        data=csv_export,
        file_name="compliance_findings.csv",
        mime="text/csv",
        disabled=not bool(csv_export),
        use_container_width=True,
    )


if "analysis_id" not in st.session_state:
    st.session_state.analysis_id = None
if "analysis_response" not in st.session_state:
    st.session_state.analysis_response = None
if "report_payload" not in st.session_state:
    st.session_state.report_payload = None
if "reviews" not in st.session_state:
    st.session_state.reviews = []


st.title("AI-Assisted Regulatory Policy Gap Analyzer")
st.caption(
    "Upload a regulation and an internal policy to identify coverage gaps, "
    "risk, confidence, recommendations, and review decisions."
)

with st.sidebar:
    st.header("Connection")
    api_url = st.text_input("FastAPI URL", value=DEFAULT_API_URL).rstrip("/")
    connected, connection_message = check_backend(api_url)
    if connected:
        st.success(connection_message)
    else:
        st.error(connection_message)

    if st.button("Clear current analysis", use_container_width=True):
        st.session_state.analysis_id = None
        st.session_state.analysis_response = None
        st.session_state.report_payload = None
        st.session_state.reviews = []
        st.rerun()

st.markdown("### Analyze documents")
with st.form("analysis-form"):
    upload_left, upload_right = st.columns(2)
    with upload_left:
        regulatory_file = st.file_uploader(
            "Regulatory document",
            type=["pdf"],
            help="Upload the regulation, standard, or compliance source document.",
        )
    with upload_right:
        policy_file = st.file_uploader(
            "Internal policy document",
            type=["pdf"],
            help="Upload the policy document to compare against the regulation.",
        )

    analyze_clicked = st.form_submit_button(
        "Analyze documents",
        type="primary",
        use_container_width=True,
        disabled=not connected,
    )

if analyze_clicked:
    if regulatory_file is None or policy_file is None:
        st.warning("Upload both PDF documents before starting the analysis.")
    else:
        try:
            with st.spinner(
                "Parsing PDFs, retrieving evidence, scoring gaps, and building the report..."
            ):
                analysis_response = post_analysis(
                    api_url,
                    regulatory_file,
                    policy_file,
                )
                analysis_id = analysis_response["analysis_id"]
                st.session_state.analysis_id = analysis_id
                st.session_state.analysis_response = analysis_response
                st.session_state.report_payload = analysis_response
                st.session_state.reviews = (
                    analysis_response.get("human_review", {}).get("reviews", [])
                )
            st.success("Analysis completed successfully.")
        except (RuntimeError, KeyError) as exc:
            st.error(f"Analysis failed: {exc}")

analysis_id = st.session_state.get("analysis_id")
if analysis_id:
    st.divider()
    st.caption(f"Analysis ID: `{analysis_id}`")

    if st.button("Refresh reviewed report"):
        try:
            with st.spinner("Refreshing report..."):
                refresh_reviewed_data(api_url, analysis_id)
            st.success("Report refreshed.")
            st.rerun()
        except RuntimeError as exc:
            st.error(str(exc))

    report = report_from_state()
    exports = exports_from_state()

    tabs = st.tabs(["Summary", "Findings", "Human review", "Downloads", "Raw response"])

    with tabs[0]:
        render_summary(report)

    with tabs[1]:
        render_findings(report)

    with tabs[2]:
        render_reviews(api_url, analysis_id)

    with tabs[3]:
        render_downloads(report, exports)

    with tabs[4]:
        st.json(st.session_state.get("report_payload") or {})