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
        replacements = [
            (r"\bc/o\b", "complains of"),
            (r"\bSOB\b", "shortness of breath"),
            (r"\bDOE\b", "dyspnea on exertion"),
            (r"\bN/V\b", "nausea and vomiting"),
            (r"\bO2\s*sat\b", "oxygen saturation"),
            (r"\bTemp\b", "Temperature"),
            (r"\bRx\b", "Prescription"),
        ]
        for pattern, replacement in replacements:
            cleaned = re.sub(pattern, replacement, cleaned, flags=re.I)
        state["cleaned_note"] = cleaned
        return state
