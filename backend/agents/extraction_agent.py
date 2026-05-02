from typing import Any

from agents.base import Agent
from services.llm_service import LLMService
from services.medical_api import MockMedicalAPI
from services.prompt_loader import PromptLoader
from services.safe_json import coerce_extraction, parse_json_object


class MedicalExtractionAgent(Agent):
    name = "medical_extraction_agent"

    def __init__(
        self,
        llm_service: LLMService,
        prompt_loader: PromptLoader,
        medical_api: MockMedicalAPI,
    ):
        self.llm_service = llm_service
        self.prompt_loader = prompt_loader
        self.medical_api = medical_api

    async def run(self, state: dict[str, Any]) -> dict[str, Any]:
        prompt = self.prompt_loader.get("extraction_prompt")
        cleaned_note = state["cleaned_note"]
        response = await self.llm_service.complete_json(
            system_prompt=self.prompt_loader.get("system_prompt"),
            user_prompt=f"{prompt}\n\nClinical note:\n{cleaned_note}",
        )
        extraction = coerce_extraction(parse_json_object(response))
        fallback = self.llm_service.heuristic_extract(cleaned_note)
        extraction = self._merge_extraction(extraction, fallback)
        references = self.medical_api.lookup_terms(
            extraction.get("symptoms", [])
            + extraction.get("diagnoses", [])
            + extraction.get("medications", [])
        )
        extraction["medical_references"] = [
            reference.model_dump() if hasattr(reference, "model_dump") else reference
            for reference in references
        ]
        state["extracted_info"] = extraction
        return state

    def _merge_extraction(
        self,
        llm_extraction: dict[str, Any],
        fallback: dict[str, Any],
    ) -> dict[str, Any]:
        merged = dict(llm_extraction)
        for key in ["symptoms", "diagnoses", "medications", "allergies", "relevant_negatives"]:
            merged[key] = self._merge_lists(
                llm_extraction.get(key, []),
                fallback.get(key, []),
            )
        merged = self._remove_negated_extractions(merged)
        merged["vital_signs"] = self._merge_vitals(
            llm_extraction.get("vital_signs", []),
            fallback.get("vital_signs", []),
        )
        return merged

    def _merge_lists(self, primary: list[Any], fallback: list[Any]) -> list[Any]:
        merged: list[Any] = []
        seen: set[str] = set()
        for item in primary + fallback:
            key = str(item).strip().lower()
            if key and key not in seen:
                seen.add(key)
                merged.append(item)
        return merged

    def _merge_vitals(self, primary: list[Any], fallback: list[Any]) -> list[Any]:
        merged: list[Any] = []
        seen_names: set[str] = set()
        for item in fallback + primary:
            if not isinstance(item, dict):
                continue
            name = self._normalize_vital_name(str(item.get("name", "")))
            value = str(item.get("value", "")).strip()
            if name and value and name not in seen_names:
                seen_names.add(name)
                item["name"] = name
                merged.append(item)
        return merged

    def _normalize_vital_name(self, name: str) -> str:
        normalized = name.strip().lower()
        aliases = {
            "bp": "blood pressure",
            "blood pressure": "blood pressure",
            "hr": "heart rate",
            "pulse": "heart rate",
            "heart rate": "heart rate",
            "spo2": "oxygen saturation",
            "o2 sat": "oxygen saturation",
            "oxygen saturation": "oxygen saturation",
            "temp": "temperature",
            "temperature": "temperature",
            "rr": "respiratory rate",
            "respiratory rate": "respiratory rate",
        }
        return aliases.get(normalized, normalized)

    def _remove_negated_extractions(self, extraction: dict[str, Any]) -> dict[str, Any]:
        negatives = [
            str(item).lower()
            for item in extraction.get("relevant_negatives", [])
            if str(item).strip()
        ]
        if not negatives:
            return extraction

        for key in ["symptoms", "diagnoses", "medications"]:
            filtered = []
            for item in extraction.get(key, []):
                lowered = str(item).lower()
                if any(lowered and lowered in negative for negative in negatives):
                    continue
                filtered.append(item)
            extraction[key] = filtered
        return extraction
