import csv
import json
from io import StringIO
from typing import Protocol, runtime_checkable

from backend.models import ComplianceReport


@runtime_checkable
class ReportExportService(Protocol):
    def export_json(
        self,
        report: ComplianceReport,
    ) -> str:
        ...

    def export_csv(
        self,
        report: ComplianceReport,
    ) -> str:
        ...


class DeterministicReportExportService:
    """Exports structured compliance reports into portable formats."""

    def export_json(
        self,
        report: ComplianceReport,
    ) -> str:
        return report.model_dump_json(
            indent=2,
        )

    def export_csv(
        self,
        report: ComplianceReport,
    ) -> str:
        output = StringIO(newline="")

        fieldnames = [
            "requirement_id",
            "gap_assessment_id",
            "requirement_summary",
            "policy_summary",
            "automated_gap_status",
            "effective_gap_status",
            "gap_reason",
            "confidence_score",
            "confidence_level",
            "confidence_reason",
            "risk_score",
            "risk_level",
            "risk_reason",
            "recommended_action",
            "requires_human_review",
            "review_status",
            "reviewer_decision",
        ]

        writer = csv.DictWriter(
            output,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for requirement_report in report.requirement_reports:
            writer.writerow(
                {
                    "requirement_id": str(
                        requirement_report.requirement_id
                    ),
                    "gap_assessment_id": str(
                        requirement_report.gap_assessment_id
                    ),
                    "requirement_summary": (
                        requirement_report.requirement_summary
                    ),
                    "policy_summary": (
                        requirement_report.policy_summary
                    ),
                    "automated_gap_status": (
                        requirement_report.gap_status.value
                    ),
                    "effective_gap_status": (
                        requirement_report.effective_gap_status.value
                    ),
                    "gap_reason": requirement_report.gap_reason,
                    "confidence_score": (
                        requirement_report.confidence_score
                    ),
                    "confidence_level": (
                        requirement_report.confidence_level.value
                    ),
                    "confidence_reason": (
                        requirement_report.confidence_reason
                    ),
                    "risk_score": requirement_report.risk_score,
                    "risk_level": requirement_report.risk_level.value,
                    "risk_reason": requirement_report.risk_reason,
                    "recommended_action": (
                        requirement_report.recommended_action
                    ),
                    "requires_human_review": (
                        requirement_report.requires_human_review
                    ),
                    "review_status": (
                        requirement_report.review_status.value
                    ),
                    "reviewer_decision": (
                        requirement_report.reviewer_decision.value
                        if requirement_report.reviewer_decision
                        is not None
                        else ""
                    ),
                }
            )

        return output.getvalue()