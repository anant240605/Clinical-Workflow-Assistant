import json
import re
from typing import Any


def parse_json_object(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def coerce_extraction(data: dict[str, Any]) -> dict[str, Any]:
    vital_signs = []
    for item in _list(data.get("vital_signs")):
        if isinstance(item, dict):
            vital_signs.append(
                {
                    "name": str(item.get("name", "unknown")),
                    "value": str(item.get("value", "")),
                    "unit": item.get("unit"),
                    "raw_text": item.get("raw_text"),
                }
            )
        elif isinstance(item, str):
            vital_signs.append(
                {"name": "unspecified", "value": item, "unit": None, "raw_text": item}
            )

    return {
        "symptoms": [str(v) for v in _list(data.get("symptoms"))],
        "diagnoses": [str(v) for v in _list(data.get("diagnoses"))],
        "medications": [str(v) for v in _list(data.get("medications"))],
        "vital_signs": vital_signs,
        "allergies": [str(v) for v in _list(data.get("allergies"))],
        "relevant_negatives": [str(v) for v in _list(data.get("relevant_negatives"))],
    }


def coerce_recommendations(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "clinical_impression": [str(v) for v in _list(data.get("clinical_impression"))],
        "possible_conditions": [str(v) for v in _list(data.get("possible_conditions"))],
        "recommended_tests": [str(v) for v in _list(data.get("recommended_tests"))],
        "medications": [str(v) for v in _list(data.get("medications"))],
        "medication_cautions": [str(v) for v in _list(data.get("medication_cautions"))],
        "red_flags": [str(v) for v in _list(data.get("red_flags"))],
        "missing_information": [str(v) for v in _list(data.get("missing_information"))],
        "follow_ups": [str(v) for v in _list(data.get("follow_ups"))],
        "safety_notes": [str(v) for v in _list(data.get("safety_notes"))]
        or ["AI recommendations require clinician review."],
    }
