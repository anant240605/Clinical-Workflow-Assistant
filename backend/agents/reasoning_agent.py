from typing import Any

from agents.base import Agent
from services.llm_service import LLMService
from services.prompt_loader import PromptLoader
from services.safe_json import coerce_recommendations, parse_json_object


class ReasoningAgent(Agent):
    name = "reasoning_agent"

    def __init__(self, llm_service: LLMService, prompt_loader: PromptLoader):
        self.llm_service = llm_service
        self.prompt_loader = prompt_loader

    async def run(self, state: dict[str, Any]) -> dict[str, Any]:
        prompt = self.prompt_loader.get("reasoning_prompt")
        response = await self.llm_service.complete_json(
            system_prompt=self.prompt_loader.get("system_prompt"),
            user_prompt=(
                f"{prompt}\n\n"
                f"Cleaned note:\n{state['cleaned_note']}\n\n"
                f"Extracted information:\n{state['extracted_info']}"
            ),
        )
        state["recommendations"] = coerce_recommendations(parse_json_object(response))
        return state
