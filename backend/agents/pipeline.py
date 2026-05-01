from agents.extraction_agent import MedicalExtractionAgent
from agents.input_agent import InputAgent
from agents.processing_agent import ProcessingAgent
from agents.reasoning_agent import ReasoningAgent
from agents.report_agent import ReportGenerationAgent
from models.schemas import AnalyzeNoteResponse
from services.file_storage import FileStorage
from services.llm_service import LLMService
from services.medical_api import MockMedicalAPI
from services.prompt_loader import PromptLoader


class ClinicalWorkflowPipeline:
    def __init__(self):
        prompt_loader = PromptLoader()
        llm_service = LLMService()
        medical_api = MockMedicalAPI()
        file_storage = FileStorage()
        self.agents = [
            InputAgent(),
            ProcessingAgent(prompt_loader),
            MedicalExtractionAgent(llm_service, prompt_loader, medical_api),
            ReasoningAgent(llm_service, prompt_loader),
            ReportGenerationAgent(llm_service, prompt_loader, file_storage),
        ]

    async def run(self, raw_note: str) -> AnalyzeNoteResponse:
        state = {"raw_note": raw_note}
        for agent in self.agents:
            state = await agent.run(state)
        return state["report"]
