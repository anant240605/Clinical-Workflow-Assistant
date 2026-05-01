from abc import ABC, abstractmethod
from typing import Any


class Agent(ABC):
    name: str

    @abstractmethod
    async def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """Run the agent and return the updated pipeline state."""
