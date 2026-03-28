"""
LLM-Judge-Bench: A Meta-Evaluation Framework for LLM-as-a-Judge Systems

This framework implements multiple judge architectures, configurable evaluation
rubrics, inter-rater reliability metrics, and a meta-audit pipeline to evaluate
how well LLM judges align with human ground-truth annotations.
"""

__version__ = "0.1.0"

from llm_judge_bench.pipeline import EvaluationPipeline
from llm_judge_bench.rubrics.rubric import Rubric, RubricRegistry

__all__ = ["EvaluationPipeline", "Rubric", "RubricRegistry"]
