"""Pointwise (single-response) judge architecture.

Evaluates a single model response against absolute quality criteria
defined in the rubric. This is the most common architecture used in
benchmarks like MT-Bench.
"""

from llm_judge_bench.judges.base import BaseJudge, JudgeResult
from llm_judge_bench.providers.base import BaseProvider
from llm_judge_bench.rubrics.rubric import Rubric


class PointwiseJudge(BaseJudge):
    """Scores a single response on absolute criteria.

    The judge evaluates one response at a time, assigning a score on the
    rubric's scale (e.g., 1-10) with chain-of-thought reasoning.
    """

    def __init__(self, provider: BaseProvider, rubric: Rubric,
                 temperature: float = 0.0):
        super().__init__(provider, rubric, temperature)

    def build_prompt(self, question: str, response: str,
                     context: str = "") -> str:
        """Construct the pointwise evaluation prompt.

        Args:
            question: The user prompt / instruction.
            response: The model's response to evaluate.
            context: Optional additional context (e.g., system prompt, category).
        """
        dimension_text = "\n".join(
            "  - {}: {}".format(dim, desc)
            for dim, desc in zip(self.rubric.dimensions,
                                 self.rubric.dimension_descriptions)
        )

        scale_min_label = self.rubric.scale_labels.get(str(self.rubric.scale_min), 'Lowest quality')
        scale_max_label = self.rubric.scale_labels.get(str(self.rubric.scale_max), 'Highest quality')
        context_block = "## Context\n" + context + "\n" if context else ""

        prompt = (
            "You are an expert evaluator assessing the quality of an AI assistant's response.\n\n"
            "## Evaluation Criteria\n"
            "{description}\n\n"
            "## Scoring Dimensions\n"
            "{dimensions}\n\n"
            "## Scale\n"
            "Rate from {s_min} to {s_max}.\n"
            "{s_min} = {s_min_label}\n"
            "{s_max} = {s_max_label}\n\n"
            "## Instructions\n"
            "1. Analyze the response against EACH evaluation dimension.\n"
            "2. Provide specific evidence from the response for each dimension.\n"
            "3. Assign a per-dimension score and an overall score.\n"
            "4. Format your overall score as [[score]] at the very end.\n\n"
            "{context_block}"
            "## User Question\n"
            "{question}\n\n"
            "## Assistant Response\n"
            "{response}\n\n"
            "## Your Evaluation\n"
            "Evaluate the response dimension by dimension, then provide an overall score."
        ).format(
            description=self.rubric.description,
            dimensions=dimension_text,
            s_min=self.rubric.scale_min,
            s_max=self.rubric.scale_max,
            s_min_label=scale_min_label,
            s_max_label=scale_max_label,
            context_block=context_block,
            question=question,
            response=response,
        )

        return prompt

    def evaluate(self, question: str, response: str,
                 context: str = "") -> JudgeResult:
        """Evaluate a single response.

        Args:
            question: The user prompt / instruction.
            response: The model's response to evaluate.
            context: Optional additional context.

        Returns:
            JudgeResult with score, reasoning, and dimension scores.
        """
        prompt = self.build_prompt(question, response, context)
        llm_response = self.provider.generate(
            prompt, temperature=self.temperature
        )

        score = self._parse_score(llm_response.text)
        reasoning = self._parse_reasoning(llm_response.text)
        dim_scores = self._parse_dimension_scores(llm_response.text)

        return JudgeResult(
            score=score,
            raw_output=llm_response.text,
            reasoning=reasoning,
            dimension_scores=dim_scores,
            metadata={
                "judge_type": "pointwise",
                "model": llm_response.model,
                "latency_ms": llm_response.latency_ms,
                "rubric": self.rubric.name,
            },
        )
