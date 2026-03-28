"""Abstract base class for LLM providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMResponse:
    """Structured response from an LLM provider."""
    text: str
    model: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    latency_ms: Optional[float] = None


class BaseProvider(ABC):
    """Abstract base class for LLM providers used as judges."""

    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0.0,
                 max_tokens: int = 1024) -> LLMResponse:
        """Generate a response from the LLM.

        Args:
            prompt: The full prompt to send to the model.
            temperature: Sampling temperature (0.0 for deterministic).
            max_tokens: Maximum tokens in the response.

        Returns:
            LLMResponse with the model's output.
        """
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier string."""
        ...
