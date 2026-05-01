from typing import Any

from agents.base import Agent


class InputAgent(Agent):
    name = "input_agent"

    async def run(self, state: dict[str, Any]) -> dict[str, Any]:
        raw_note = state.get("raw_note", "").strip()
        if not raw_note:
            raise ValueError("Clinical note cannot be empty.")

        state["raw_note"] = raw_note
        state["metadata"] = {
            "input_character_count": len(raw_note),
            "input_word_count": len(raw_note.split()),
        }
        return state
