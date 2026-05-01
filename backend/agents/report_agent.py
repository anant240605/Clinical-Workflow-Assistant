from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from agents.base import Agent
from models.schemas import AnalyzeNoteResponse, ExtractedInfo, RecommendationSet, VitalSign
from services.file_storage import FileStorage
from services.llm_service import LLMService
from services.prompt_loader import PromptLoader


class ReportGenerationAgent(Agent):
    name = "report_generation_agent"

    def __init__(
        self,
        llm_service: LLMService,
        prompt_loader: PromptLoader,
        file_storage: FileStorage,
    ):
        self.llm_service = llm_service
        self.prompt_loader = prompt_loader
        self.file_storage = file_storage

    async def run(self, state: dict[str, Any]) -> dict[str, Any]:
        summary = await self.llm_service.complete_text(
            system_prompt=self.prompt_loader.get("system_prompt"),
            user_prompt=(
                f"{self.prompt_loader.get('summary_prompt')}\n\n"
                f"Cleaned note:\n{state['cleaned_note']}\n\n"
                f"Extracted information:\n{state['extracted_info']}"
            ),
        )

        report_text = await self.llm_service.complete_text(
            system_prompt=self.prompt_loader.get("system_prompt"),
            user_prompt=(
                f"{self.prompt_loader.get('report_prompt')}\n\n"
                f"Summary:\n{summary}\n\n"
                f"Extracted information:\n{state['extracted_info']}\n\n"
                f"Recommendations:\n{state['recommendations']}"
            ),
        )

        extracted_info = ExtractedInfo(
            symptoms=state["extracted_info"].get("symptoms", []),
            diagnoses=state["extracted_info"].get("diagnoses", []),
            medications=state["extracted_info"].get("medications", []),
            vital_signs=[
                VitalSign(**vital)
                for vital in state["extracted_info"].get("vital_signs", [])
                if isinstance(vital, dict)
            ],
            allergies=state["extracted_info"].get("allergies", []),
            relevant_negatives=state["extracted_info"].get("relevant_negatives", []),
            medical_references=state["extracted_info"].get("medical_references", []),
        )
        recommendations = RecommendationSet(**state["recommendations"])
        report = AnalyzeNoteResponse(
            id=str(uuid4()),
            created_at=datetime.now(UTC),
            summary=summary.strip(),
            extracted_info=extracted_info,
            recommendations=recommendations,
            report=report_text.strip(),
            raw_note=state["raw_note"],
            cleaned_note=state["cleaned_note"],
            metadata=state["metadata"],
        )
        self.file_storage.save_report(report)
        state["report"] = report
        return state
