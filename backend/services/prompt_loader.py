import json
from pathlib import Path

from services.config import settings


class PromptLoader:
    def __init__(self, prompts_path: str | None = None):
        self.prompts_path = Path(prompts_path or settings.prompts_path)

    def load(self) -> dict[str, str]:
        if not self.prompts_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {self.prompts_path}")
        with self.prompts_path.open("r", encoding="utf-8") as file:
            prompts = json.load(file)
        if not isinstance(prompts, dict):
            raise ValueError("Prompt file must contain a JSON object.")
        return prompts

    def get(self, key: str) -> str:
        prompts = self.load()
        value = prompts.get(key)
        if not isinstance(value, str) or not value.strip():
            raise KeyError(f"Missing prompt key: {key}")
        return value
