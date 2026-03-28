"""Abstract base class for all judge architectures."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from llm_judge_bench.providers.base import BaseProvider
from llm_judge_bench.rubrics.rubric import Rubric


@dataclass
class JudgeResult:
    """Structured output from a judge evaluation.

    Attributes:
        score: Numeric score assigned by the judge.
        raw_output: Full text output from the judge LLM.
        reasoning: Extracted chain-of-thought reasoning.
        dimension_scores: Per-rubric-dimension scores (e.g., accuracy: 4, clarity: 5).
        metadata: Additional metadata (latency, model, etc.).
    """
    score: float
    raw_output: str
    reasoning: str = ""
    dimension_scores: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseJudge(ABC):
    """Abstract base class for LLM judge architectures.

    All judge types share a common interface but differ in how they
    construct prompts and interpret model outputs:

    - PointwiseJudge: Scores a single response on absolute criteria.
    - PairwiseJudge: Compares two responses and picks a winner.
    - ReferenceBasedJudge: Evaluates a response against a gold reference.
    """

    def __init__(self, provider: BaseProvider, rubric: Rubric,
                 temperature: float = 0.0):
        self.provider = provider
        self.rubric = rubric
        self.temperature = temperature

    @abstractmethod
    def evaluate(self, **kwargs) -> JudgeResult:
        """Run a single evaluation. Subclasses define the signature."""
        ...

    @abstractmethod
    def build_prompt(self, **kwargs) -> str:
        """Construct the full prompt for the judge LLM."""
        ...

    def _parse_score(self, text: str) -> float:
        """Extract a numeric score from judge output.

        Looks for patterns like [[5]], [Score: 4], or standalone digits
        within the expected scale range.
        """
        import re

        # Pattern 1: [[score]]
        match = re.search(r'\[\[(\d+(?:\.\d+)?)\]\]', text)
        if match:
            return float(match.group(1))

        # Pattern 2: [Score: X] or Score: X
        match = re.search(r'[Ss]core:\s*(\d+(?:\.\d+)?)', text)
        if match:
            return float(match.group(1))

        # Pattern 3: "Rating: X"
        match = re.search(r'[Rr]ating:\s*(\d+(?:\.\d+)?)', text)
        if match:
            return float(match.group(1))

        # Fallback: find any number within scale range at end of text
        numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', text[-100:])
        scale_max = self.rubric.scale_max
        for n in reversed(numbers):
            val = float(n)
            if self.rubric.scale_min <= val <= scale_max:
                return val

        return -1.0  # Parsing failure sentinel

    def _parse_reasoning(self, text: str) -> str:
        """Extract reasoning/chain-of-thought from judge output."""
        import re

        # Look for content between reasoning markers
        patterns = [
            r'(?:Reasoning|Analysis|Explanation|Justification):\s*(.*?)(?=\n\s*(?:Score|Rating|\[\[))',
            r'(.*?)(?=\n\s*(?:Score|Rating|\[\[))',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                reasoning = match.group(1).strip()
                if len(reasoning) > 20:
                    return reasoning

        return text.strip()

    def _parse_dimension_scores(self, text: str) -> Dict[str, float]:
        """Extract per-dimension scores from structured judge output."""
        import re

        scores = {}
        for dim in self.rubric.dimensions:
            pattern = rf'{re.escape(dim)}[:\s]*(\d+(?:\.\d+)?)'
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                scores[dim] = float(match.group(1))
        return scores
