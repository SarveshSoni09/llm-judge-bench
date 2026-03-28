"""Main evaluation pipeline orchestrator.

Ties together judge architectures, rubrics, metrics, sampling, and
meta-auditing into a single configurable pipeline.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        """Fallback if tqdm is not installed."""
        return iterable

from llm_judge_bench.judges.base import BaseJudge, JudgeResult
from llm_judge_bench.judges.pointwise import PointwiseJudge
from llm_judge_bench.judges.pairwise import PairwiseJudge
from llm_judge_bench.judges.reference import ReferenceBasedJudge
from llm_judge_bench.metrics.agreement import AgreementMetrics
from llm_judge_bench.meta_audit.evaluator import MetaAuditor, MetaAuditReport
from llm_judge_bench.rubrics.rubric import Rubric, RubricRegistry
from llm_judge_bench.providers.base import BaseProvider
from llm_judge_bench.sampling.strategies import (
    StratifiedSampler, ConfidenceBasedSampler, SamplingReport
)


class EvaluationPipeline:
    """End-to-end LLM-as-a-Judge evaluation pipeline.

    Orchestrates:
    1. Loading evaluation data (questions, responses, human labels).
    2. Running judge evaluations across the dataset.
    3. Computing agreement metrics against human annotations.
    4. Running meta-audit analysis.
    5. Generating comprehensive reports.
    """

    def __init__(self, provider: BaseProvider, rubric: Rubric,
                 judge_type: str = "pointwise"):
        """
        Args:
            provider: LLM provider for the judge.
            rubric: Evaluation rubric to use.
            judge_type: One of "pointwise", "pairwise", "reference".
        """
        self.provider = provider
        self.rubric = rubric
        self.judge_type = judge_type

        if judge_type == "pointwise":
            self.judge = PointwiseJudge(provider, rubric)
        elif judge_type == "pairwise":
            self.judge = PairwiseJudge(provider, rubric)
        elif judge_type == "reference":
            self.judge = ReferenceBasedJudge(provider, rubric)
        else:
            raise ValueError(f"Unknown judge type: {judge_type}")

        self.results: List[JudgeResult] = []
        self.meta_auditor = MetaAuditor()

    def load_data(self, path: str) -> List[Dict]:
        """Load evaluation dataset from JSON file.

        Expected format: list of dicts with keys:
        - question: str
        - response: str (for pointwise/reference)
        - response_a, response_b: str (for pairwise)
        - reference: str (for reference-based)
        - human_score: float (ground truth)
        - category: str (optional)
        """
        with open(path, "r") as f:
            data = json.load(f)
        return data

    def run(self, data: List[Dict], max_items: Optional[int] = None,
            verbose: bool = True) -> List[JudgeResult]:
        """Run the judge on the evaluation dataset.

        Args:
            data: List of evaluation items.
            max_items: Optional limit on number of items to evaluate.
            verbose: Show progress bar.

        Returns:
            List of JudgeResult objects.
        """
        items = data[:max_items] if max_items else data
        self.results = []

        iterator = tqdm(items, desc=f"Running {self.judge_type} judge") if verbose else items

        for item in iterator:
            try:
                if self.judge_type == "pointwise":
                    result = self.judge.evaluate(
                        question=item["question"],
                        response=item["response"],
                        context=item.get("category", ""),
                    )
                elif self.judge_type == "pairwise":
                    result = self.judge.evaluate(
                        question=item["question"],
                        response_a=item["response_a"],
                        response_b=item["response_b"],
                    )
                elif self.judge_type == "reference":
                    result = self.judge.evaluate(
                        question=item["question"],
                        response=item["response"],
                        reference=item["reference"],
                    )
                self.results.append(result)
            except Exception as e:
                print(f"  Error evaluating item: {e}")
                self.results.append(JudgeResult(
                    score=-1.0, raw_output=f"ERROR: {e}", reasoning=""
                ))

        return self.results

    def meta_audit(self, data: List[Dict]) -> MetaAuditReport:
        """Run meta-audit comparing judge results to human annotations.

        Args:
            data: The same dataset used for run(), must include human_score.

        Returns:
            MetaAuditReport with comprehensive quality analysis.
        """
        judge_scores = [r.score for r in self.results]
        human_scores = [item.get("human_score", -1) for item in data[:len(self.results)]]
        categories = [item.get("category", "unknown") for item in data[:len(self.results)]]
        metadata = [r.metadata for r in self.results]

        return self.meta_auditor.audit(
            judge_scores=judge_scores,
            human_scores=human_scores,
            categories=categories,
            metadata=metadata,
        )

    def to_dataframe(self, data: List[Dict]) -> pd.DataFrame:
        """Convert results to a pandas DataFrame for analysis.

        Merges judge outputs with input data and human annotations.
        """
        records = []
        for i, (item, result) in enumerate(zip(data[:len(self.results)], self.results)):
            record = {
                "index": i,
                "question": item.get("question", "")[:100],
                "category": item.get("category", "unknown"),
                "human_score": item.get("human_score", -1),
                "judge_score": result.score,
                "abs_diff": abs(result.score - item.get("human_score", -1)),
                "judge_type": result.metadata.get("judge_type", ""),
                "latency_ms": result.metadata.get("latency_ms", 0),
            }
            # Add dimension scores
            for dim, score in result.dimension_scores.items():
                record[f"dim_{dim}"] = score
            records.append(record)

        return pd.DataFrame(records)

    def save_results(self, path: str, data: List[Dict]):
        """Save results and meta-audit report to JSON."""
        output = {
            "config": {
                "judge_type": self.judge_type,
                "rubric": self.rubric.name,
                "model": self.provider.model_name,
                "n_items": len(self.results),
            },
            "results": [
                {
                    "score": r.score,
                    "reasoning": r.reasoning[:500],
                    "dimension_scores": r.dimension_scores,
                    "metadata": {k: v for k, v in r.metadata.items()
                                 if isinstance(v, (str, int, float, bool, type(None)))},
                }
                for r in self.results
            ],
        }

        with open(path, "w") as f:
            json.dump(output, f, indent=2)
