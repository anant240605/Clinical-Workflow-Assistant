import json

from sqlalchemy.orm import Session

from models.report import ClinicalReport
from models.schemas import AnalyzeNoteResponse, ExtractedInfo, RecommendationSet


class ReportRepository:
    def __init__(self, db: Session):
        self.db = db

    def save(self, report: AnalyzeNoteResponse) -> None:
        record = ClinicalReport(
            id=report.id,
            created_at=report.created_at,
            raw_note=report.raw_note,
            cleaned_note=report.cleaned_note,
            summary=report.summary,
            extracted_info_json=report.extracted_info.model_dump_json(),
            recommendations_json=report.recommendations.model_dump_json(),
            report=report.report,
            metadata_json=json.dumps(report.metadata),
        )
        self.db.add(record)
        self.db.commit()

    def get(self, report_id: str) -> AnalyzeNoteResponse | None:
        record = self.db.get(ClinicalReport, report_id)
        if record is None:
            return None
        return AnalyzeNoteResponse(
            id=record.id,
            created_at=record.created_at,
            summary=record.summary,
            extracted_info=ExtractedInfo.model_validate_json(record.extracted_info_json),
            recommendations=RecommendationSet.model_validate_json(
                record.recommendations_json
            ),
            report=record.report,
            raw_note=record.raw_note,
            cleaned_note=record.cleaned_note,
            metadata=json.loads(record.metadata_json),
        )
