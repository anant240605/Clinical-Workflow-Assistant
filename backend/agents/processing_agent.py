import re
from typing import Any

from agents.base import Agent
from services.prompt_loader import PromptLoader


class ProcessingAgent(Agent):
    name = "processing_agent"

    def __init__(self, prompt_loader: PromptLoader):
        self.prompt_loader = prompt_loader

    async def run(self, state: dict[str, Any]) -> dict[str, Any]:
        self.prompt_loader.get("processing_prompt")
        raw_note = state["raw_note"]
        cleaned = re.sub(r"\s+", " ", raw_note).strip()
        cleaned = cleaned.replace("Temp ", "Temperature ")
        state["cleaned_note"] = cleaned
        return state
