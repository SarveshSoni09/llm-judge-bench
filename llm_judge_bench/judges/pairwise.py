"""Pairwise (comparison) judge architecture.

Compares two model responses head-to-head and determines which is better.
This architecture is used in benchmarks like Chatbot Arena and AlpacaEval.
Includes position-bias mitigation via response-order swapping.
"""

import random
from llm_judge_bench.judges.base import BaseJudge, JudgeResult
from llm_judge_bench.providers.base import BaseProvider
from llm_judge_bench.rubrics.rubric import Rubric


class PairwiseJudge(BaseJudge):
    """Compares two responses and determines which is better.

    Implements position-bias mitigation by optionally evaluating both
    orderings (A-B and B-A) and checking for consistency.
    """

    def __init__(self, provider: BaseProvider, rubric: Rubric,
                 temperature: float = 0.0, swap_mitigation: bool = True):
        """
        Args:
            swap_mitigation: If True, evaluate both orderings to detect
                             position bias. Doubles the number of API calls.
        """
        super().__init__(provider, rubric, temperature)
        self.swap_mitigation = swap_mitigation

    def build_prompt(self, question: str, response_a: str,
                     response_b: str) -> str:
        """Construct the pairwise comparison prompt."""
        dimension_text = "\n".join(
            "  - {}: {}".format(dim, desc)
            for dim, desc in zip(self.rubric.dimensions,
                                 self.rubric.dimension_descriptions)
        )

        prompt = (
            "You are an expert evaluator comparing two AI assistant responses.\n\n"
            "## Evaluation Criteria\n"
            "{description}\n\n"
            "## Scoring Dimensions\n"
            "{dimensions}\n\n"
            "## Instructions\n"
            "1. Compare Response A and Response B on each evaluation dimension.\n"
            "2. Identify specific strengths and weaknesses of each response.\n"
            "3. Declare a winner: output exactly one of: [[A]], [[B]], or [[tie]].\n"
            "4. Explain your reasoning before giving the verdict.\n\n"
            "## User Question\n"
            "{question}\n\n"
            "## Response A\n"
            "{response_a}\n\n"
            "## Response B\n"
            "{response_b}\n\n"
            "## Your Evaluation\n"
            "Compare the two responses dimension by dimension, then declare your verdict."
        ).format(
            description=self.rubric.description,
            dimensions=dimension_text,
            question=question,
            response_a=response_a,
            response_b=response_b,
        )

        return prompt

    def _parse_verdict(self, text: str) -> str:
        """Extract the pairwise verdict from judge output."""
        import re

        match = re.search(r'\[\[(A|B|tie)\]\]', text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        text_lower = text.lower()
        if "response a is better" in text_lower or "response a wins" in text_lower:
            return "A"
        if "response b is better" in text_lower or "response b wins" in text_lower:
            return "B"
        if "tie" in text_lower or "equal" in text_lower:
            return "TIE"

        return "UNKNOWN"

    def evaluate(self, question: str, response_a: str,
                 response_b: str) -> JudgeResult:
        """Compare two responses with optional position-bias mitigation.

        Args:
            question: The user prompt / instruction.
            response_a: First model's response.
            response_b: Second model's response.

        Returns:
            JudgeResult where score encodes: 1.0=A wins, 0.0=B wins, 0.5=tie.
            metadata includes position_bias_detected flag.
        """
        # First evaluation: original order
        prompt_1 = self.build_prompt(question, response_a, response_b)
        result_1 = self.provider.generate(prompt_1, temperature=self.temperature)
        verdict_1 = self._parse_verdict(result_1.text)

        position_bias = False
        final_verdict = verdict_1

        if self.swap_mitigation:
            # Second evaluation: swapped order
            prompt_2 = self.build_prompt(question, response_b, response_a)
            result_2 = self.provider.generate(prompt_2, temperature=self.temperature)
            verdict_2_raw = self._parse_verdict(result_2.text)

            # Map swapped verdict back: if judge says A in swapped order, it means B
            verdict_2 = {"A": "B", "B": "A", "TIE": "TIE"}.get(
                verdict_2_raw, "UNKNOWN"
            )

            if verdict_1 != verdict_2:
                position_bias = True
                final_verdict = "TIE"  # Inconsistent → conservative tie
            else:
                final_verdict = verdict_1

        score_map = {"A": 1.0, "B": 0.0, "TIE": 0.5, "UNKNOWN": -1.0}

        return JudgeResult(
            score=score_map.get(final_verdict, -1.0),
            raw_output=result_1.text,
            reasoning=self._parse_reasoning(result_1.text),
            metadata={
                "judge_type": "pairwise",
                "model": result_1.model,
                "verdict": final_verdict,
                "verdict_original_order": verdict_1,
                "verdict_swapped_order": verdict_2 if self.swap_mitigation else None,
                "position_bias_detected": position_bias,
                "swap_mitigation": self.swap_mitigation,
                "rubric": self.rubric.name,
            },
        )
