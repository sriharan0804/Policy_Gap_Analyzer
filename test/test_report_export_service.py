import csv
import json
from io import StringIO
from uuid import uuid4

from backend.models import (
    ComplianceReport,
    GapConfidenceLevel,
    GapReviewStatus,
    GapStatus,
    ReportSummary,
    RequirementReport,
    RiskLevel,
)
from backend.services.report_export_service import (
    DeterministicReportExportService,
    ReportExportService,
)


def build_compliance_report() -> ComplianceReport:
    requirement_report = RequirementReport(
        requirement_id=uuid4(),
        gap_assessment_id=uuid4(),
        requirement_summary=(
            "The organization must encrypt sensitive customer data."
        ),
        policy_summary=(
            "The policy requires encryption for confidential information."
        ),
        gap_status=GapStatus.PARTIALLY_ADDRESSED,
        gap_reason=(
            "The policy covers confidential information but does not "
            "explicitly mention all customer data."
        ),
        confidence_score=0.86,
        confidence_level=GapConfidenceLevel.HIGH,
        confidence_reason=(
            "The regulatory and policy statements are both clear."
        ),
        risk_score=0.72,
        risk_level=RiskLevel.HIGH,
        risk_reason=(
            "Incomplete encryption coverage may expose sensitive data."
        ),
        recommended_action=(
            "Expand the encryption policy to cover all sensitive "
            "customer data."
        ),
        requires_human_review=True,
        review_status=GapReviewStatus.PENDING,
        effective_gap_status=GapStatus.PARTIALLY_ADDRESSED,
    )

    summary = ReportSummary(
        total_requirements=1,
        fully_addressed_count=0,
        partially_addressed_count=1,
        not_addressed_count=0,
        contradicted_count=0,
        insufficient_evidence_count=0,
        high_risk_count=1,
        critical_risk_count=0,
        compliance_score=0.5,
        human_review_required_count=1,
    )

    return ComplianceReport(
        analysis_id=uuid4(),
        regulatory_document_id=uuid4(),
        policy_document_id=uuid4(),
        summary=summary,
        requirement_reports=[requirement_report],
    )


def test_service_satisfies_protocol():
    service = DeterministicReportExportService()

    assert isinstance(service, ReportExportService)


def test_export_json_returns_valid_json():
    report = build_compliance_report()
    service = DeterministicReportExportService()

    exported = service.export_json(report)
    parsed = json.loads(exported)

    assert parsed["analysis_id"] == str(report.analysis_id)
    assert parsed["summary"]["total_requirements"] == 1
    assert len(parsed["requirement_reports"]) == 1


def test_export_json_contains_requirement_data():
    report = build_compliance_report()
    service = DeterministicReportExportService()

    parsed = json.loads(service.export_json(report))
    requirement = parsed["requirement_reports"][0]

    assert (
        requirement["gap_status"]
        == GapStatus.PARTIALLY_ADDRESSED.value
    )
    assert requirement["risk_level"] == RiskLevel.HIGH.value
    assert requirement["confidence_score"] == 0.86


def test_export_csv_returns_header_and_row():
    report = build_compliance_report()
    service = DeterministicReportExportService()

    exported = service.export_csv(report)

    rows = list(
        csv.DictReader(
            StringIO(exported),
        )
    )

    assert len(rows) == 1

    row = rows[0]

    assert (
        row["automated_gap_status"]
        == GapStatus.PARTIALLY_ADDRESSED.value
    )
    assert (
        row["effective_gap_status"]
        == GapStatus.PARTIALLY_ADDRESSED.value
    )
    assert row["risk_level"] == RiskLevel.HIGH.value
    assert row["review_status"] == GapReviewStatus.PENDING.value


def test_export_csv_handles_commas_in_text():
    report = build_compliance_report()

    report.requirement_reports[0].requirement_summary = (
        "The organization must encrypt names, addresses, and account data."
    )

    service = DeterministicReportExportService()

    exported = service.export_csv(report)

    rows = list(
        csv.DictReader(
            StringIO(exported),
        )
    )

    assert rows[0]["requirement_summary"] == (
        "The organization must encrypt names, addresses, and account data."
    )


def test_export_empty_report_to_csv():
    report = ComplianceReport(
        analysis_id=uuid4(),
        regulatory_document_id=uuid4(),
        policy_document_id=uuid4(),
        summary=ReportSummary(
            total_requirements=0,
            fully_addressed_count=0,
            partially_addressed_count=0,
            not_addressed_count=0,
            contradicted_count=0,
            insufficient_evidence_count=0,
            high_risk_count=0,
            critical_risk_count=0,
            compliance_score=0.0,
            human_review_required_count=0,
        ),
        requirement_reports=[],
    )

    service = DeterministicReportExportService()

    exported = service.export_csv(report)

    rows = list(
        csv.DictReader(
            StringIO(exported),
        )
    )

    assert rows == []
    assert "requirement_id" in exported