import json
from pathlib import Path

from models.schemas import AnalyzeNoteResponse
from services.config import settings


class FileStorage:
    def __init__(self, storage_dir: str | None = None):
        self.storage_dir = Path(storage_dir or settings.report_storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def save_report(self, report: AnalyzeNoteResponse) -> Path:
        path = self.storage_dir / f"{report.id}.json"
        with path.open("w", encoding="utf-8") as file:
            json.dump(report.model_dump(mode="json"), file, indent=2)
        return path
