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

        Handles multiple output formats from different LLMs:
        - [[5]]                          (MT-Bench style, explicitly requested)
        - **Overall Score:** 7           (Gemini 2.5-flash markdown bold)
        - **Score:** 8/10                (fraction format)
        - Overall Score: 7               (plain text)
        - Score: 6                       (plain label)
        - Rating: 9                      (rating label)
        - Fallback: last number in range (catch-all)
        """
        import re

        # Pattern 1: [[score]] — explicitly requested format
        match = re.search(r'\[\[(\d+(?:\.\d+)?)\]\]', text)
        if match:
            return float(match.group(1))

        # Pattern 2: **Overall Score:** X or **Overall Score: X/10**
        # Handles Gemini 2.5-flash markdown bold output
        match = re.search(
            r'\*\*\s*(?:Overall\s+)?(?:Score|Rating)\s*:?\*?\*?\s*(\d+(?:\.\d+)?)',
            text, re.IGNORECASE
        )
        if match:
            val = float(match.group(1))
            if self.rubric.scale_min <= val <= self.rubric.scale_max:
                return val

        # Pattern 3: Overall Score: X (plain, no bold)
        match = re.search(
            r'[Oo]verall\s+(?:[Ss]core|[Rr]ating)\s*:?\s*(\d+(?:\.\d+)?)',
            text
        )
        if match:
            val = float(match.group(1))
            if self.rubric.scale_min <= val <= self.rubric.scale_max:
                return val

        # Pattern 4: Score: X or Rating: X (plain label)
        match = re.search(r'(?:[Ss]core|[Rr]ating)\s*:\s*(\d+(?:\.\d+)?)', text)
        if match:
            val = float(match.group(1))
            if self.rubric.scale_min <= val <= self.rubric.scale_max:
                return val

        # Pattern 5: X/10 fraction format (e.g. "8/10" or "score of 7/10")
        match = re.search(
            r'(\d+(?:\.\d+)?)\s*/\s*' + str(self.rubric.scale_max), text
        )
        if match:
            val = float(match.group(1))
            if self.rubric.scale_min <= val <= self.rubric.scale_max:
                return val

        # Fallback: last number within scale range in the final 300 chars
        numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', text[-300:])
        for n in reversed(numbers):
            val = float(n)
            if self.rubric.scale_min <= val <= self.rubric.scale_max:
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
