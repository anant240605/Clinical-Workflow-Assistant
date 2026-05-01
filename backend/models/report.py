from datetime import datetime

from sqlalchemy import DateTime, Text
from sqlalchemy.orm import Mapped, mapped_column

from models.database import Base


class ClinicalReport(Base):
    __tablename__ = "clinical_reports"

    id: Mapped[str] = mapped_column(primary_key=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    raw_note: Mapped[str] = mapped_column(Text)
    cleaned_note: Mapped[str] = mapped_column(Text)
    summary: Mapped[str] = mapped_column(Text)
    extracted_info_json: Mapped[str] = mapped_column(Text)
    recommendations_json: Mapped[str] = mapped_column(Text)
    report: Mapped[str] = mapped_column(Text)
    metadata_json: Mapped[str] = mapped_column(Text)
