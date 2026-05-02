from typing import Any

from agents.base import Agent
from services.clinical_safety import ClinicalSafetyValidator
from services.llm_service import LLMService
from services.prompt_loader import PromptLoader
from services.safe_json import coerce_recommendations, parse_json_object


class ReasoningAgent(Agent):
    name = "reasoning_agent"

    def __init__(self, llm_service: LLMService, prompt_loader: PromptLoader):
        self.llm_service = llm_service
        self.prompt_loader = prompt_loader
        self.safety_validator = ClinicalSafetyValidator()

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
        recommendations = coerce_recommendations(parse_json_object(response))
        fallback = self.llm_service.heuristic_recommendations(state["extracted_info"])
        merged = self._merge_recommendations(recommendations, fallback)
        state["recommendations"] = self.safety_validator.validate(
            state["cleaned_note"],
            state["extracted_info"],
            merged,
        )
        return state

    def _merge_recommendations(
        self,
        primary: dict[str, list[str]],
        fallback: dict[str, list[str]],
    ) -> dict[str, list[str]]:
        merged = dict(primary)
        for key in [
            "clinical_impression",
            "possible_conditions",
            "recommended_tests",
            "medications",
            "medication_cautions",
            "red_flags",
            "missing_information",
            "follow_ups",
            "safety_notes",
        ]:
            merged[key] = self._merge_lists(primary.get(key, []), fallback.get(key, []))

        specific_conditions = [
            item
            for item in merged["possible_conditions"]
            if "insufficient information" not in item.lower()
        ]
        if specific_conditions:
            merged["possible_conditions"] = specific_conditions

        specific_medications = [
            item
            for item in merged["medications"]
            if "requires more clinical detail" not in item.lower()
        ]
        if specific_medications:
            merged["medications"] = specific_medications

        return merged

    def _merge_lists(self, primary: list[str], fallback: list[str]) -> list[str]:
        merged: list[str] = []
        seen: set[str] = set()
        for item in primary + fallback:
            value = str(item).strip()
            key = value.lower()
            if value and key not in seen:
                seen.add(key)
                merged.append(value)
        return merged
