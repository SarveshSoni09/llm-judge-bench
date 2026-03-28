"""Reference-based judge architecture.

Evaluates a model response by comparing it against a gold-standard
reference answer. Useful for factual accuracy, knowledge-grounded tasks,
and benchmarks where ground-truth answers exist.
"""

from llm_judge_bench.judges.base import BaseJudge, JudgeResult
from llm_judge_bench.providers.base import BaseProvider
from llm_judge_bench.rubrics.rubric import Rubric


class ReferenceBasedJudge(BaseJudge):
    """Evaluates a response against a gold-standard reference answer.

    This architecture is critical for factual accuracy evaluation where
    ground truth exists — the judge must assess whether the candidate
    response covers the key facts in the reference without hallucination.
    """

    def __init__(self, provider: BaseProvider, rubric: Rubric,
                 temperature: float = 0.0):
        super().__init__(provider, rubric, temperature)

    def build_prompt(self, question: str, response: str,
                     reference: str) -> str:
        """Construct the reference-based evaluation prompt."""
        dimension_text = "\n".join(
            "  - {}: {}".format(dim, desc)
            for dim, desc in zip(self.rubric.dimensions,
                                 self.rubric.dimension_descriptions)
        )

        scale_min_label = self.rubric.scale_labels.get(str(self.rubric.scale_min), 'Lowest quality')
        scale_max_label = self.rubric.scale_labels.get(str(self.rubric.scale_max), 'Highest quality')

        prompt = (
            "You are an expert evaluator assessing an AI assistant's response "
            "against a gold-standard reference answer.\n\n"
            "## Evaluation Criteria\n"
            "{description}\n\n"
            "## Scoring Dimensions\n"
            "{dimensions}\n\n"
            "## Scale\n"
            "Rate from {s_min} to {s_max}.\n"
            "{s_min} = {s_min_label}\n"
            "{s_max} = {s_max_label}\n\n"
            "## Instructions\n"
            "1. Carefully compare the candidate response with the reference answer.\n"
            "2. Assess factual accuracy: does the response contain the key facts from the reference?\n"
            "3. Check for hallucinations: does the response introduce incorrect claims not in the reference?\n"
            "4. Evaluate completeness: does the response cover all important points in the reference?\n"
            "5. Provide dimension-by-dimension analysis.\n"
            "6. Format your overall score as [[score]] at the very end.\n\n"
            "## User Question\n"
            "{question}\n\n"
            "## Reference Answer (Gold Standard)\n"
            "{reference}\n\n"
            "## Candidate Response (To Evaluate)\n"
            "{response}\n\n"
            "## Your Evaluation\n"
            "Compare the candidate response against the reference, then provide your score."
        ).format(
            description=self.rubric.description,
            dimensions=dimension_text,
            s_min=self.rubric.scale_min,
            s_max=self.rubric.scale_max,
            s_min_label=scale_min_label,
            s_max_label=scale_max_label,
            question=question,
            reference=reference,
            response=response,
        )

        return prompt

    def evaluate(self, question: str, response: str,
                 reference: str) -> JudgeResult:
        """Evaluate a response against a reference answer.

        Args:
            question: The user prompt / instruction.
            response: The candidate model's response to evaluate.
            reference: The gold-standard reference answer.

        Returns:
            JudgeResult with factual accuracy score and analysis.
        """
        prompt = self.build_prompt(question, response, reference)
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
                "judge_type": "reference_based",
                "model": llm_response.model,
                "latency_ms": llm_response.latency_ms,
                "rubric": self.rubric.name,
            },
        )
