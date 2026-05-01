from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class AnalyzeNoteRequest(BaseModel):
    note: str = Field(..., min_length=1, examples=[
        "Patient reports fever for 3 days, mild cough, fatigue. Temperature 101 F. No prior conditions."
    ])


class VitalSign(BaseModel):
    name: str
    value: str
    unit: str | None = None
    raw_text: str | None = None


class MedicalReference(BaseModel):
    term: str
    category: str
    clinical_hint: str


class ExtractedInfo(BaseModel):
    symptoms: list[str] = Field(default_factory=list)
    diagnoses: list[str] = Field(default_factory=list)
    medications: list[str] = Field(default_factory=list)
    vital_signs: list[VitalSign] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)
    relevant_negatives: list[str] = Field(default_factory=list)
    medical_references: list[MedicalReference] = Field(default_factory=list)


class RecommendationSet(BaseModel):
    possible_conditions: list[str] = Field(default_factory=list)
    recommended_tests: list[str] = Field(default_factory=list)
    medications: list[str] = Field(default_factory=list)
    follow_ups: list[str] = Field(default_factory=list)
    safety_notes: list[str] = Field(default_factory=list)


class AnalyzeNoteResponse(BaseModel):
    id: str
    created_at: datetime
    summary: str
    extracted_info: ExtractedInfo
    recommendations: RecommendationSet
    report: str
    raw_note: str
    cleaned_note: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ReportResponse(AnalyzeNoteResponse):
    pass
