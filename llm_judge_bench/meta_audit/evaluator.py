"""Meta-audit pipeline for evaluating LLM judge quality.

The meta-auditor answers the critical question: "How good is our judge?"
It compares LLM judge outputs against human ground-truth annotations
across multiple dimensions — agreement, bias, consistency, and
category-level performance breakdown.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from llm_judge_bench.metrics.agreement import AgreementMetrics, AgreementReport


@dataclass
class MetaAuditReport:
    """Comprehensive report on judge quality.

    Attributes:
        overall_agreement: Agreement metrics across all samples.
        category_agreement: Per-category agreement breakdown.
        consistency_scores: Self-consistency when judging same items multiple times.
        bias_analysis: Systematic biases detected in the judge.
        failure_analysis: Cases where the judge most disagreed with humans.
        position_bias_rate: For pairwise judges, rate of position-dependent verdicts.
    """
    overall_agreement: Optional[AgreementReport] = None
    category_agreement: Dict[str, AgreementReport] = field(default_factory=dict)
    consistency_scores: Dict[str, float] = field(default_factory=dict)
    bias_analysis: Dict[str, Any] = field(default_factory=dict)
    failure_analysis: List[Dict] = field(default_factory=list)
    position_bias_rate: Optional[float] = None

    def summary(self) -> str:
        """Human-readable meta-audit summary."""
        lines = ["=" * 55, "META-AUDIT REPORT", "=" * 55]

        if self.overall_agreement:
            lines.append("\n--- Overall Agreement ---")
            lines.append(self.overall_agreement.summary())

        if self.category_agreement:
            lines.append("\n--- Per-Category Weighted Kappa ---")
            for cat, report in self.category_agreement.items():
                lines.append(
                    f"  {cat:20s}: κ_w={report.weighted_kappa:+.3f}  "
                    f"(n={report.n_samples}, bias={report.bias:+.2f})"
                )

        if self.bias_analysis:
            lines.append("\n--- Bias Analysis ---")
            for key, val in self.bias_analysis.items():
                lines.append(f"  {key}: {val}")

        if self.position_bias_rate is not None:
            lines.append(f"\n--- Position Bias Rate: {self.position_bias_rate:.1%} ---")

        if self.consistency_scores:
            lines.append("\n--- Self-Consistency ---")
            for key, val in self.consistency_scores.items():
                lines.append(f"  {key}: {val:.3f}")

        if self.failure_analysis:
            lines.append(f"\n--- Top {len(self.failure_analysis)} Disagreements ---")
            for i, case in enumerate(self.failure_analysis[:5]):
                lines.append(
                    f"  {i+1}. |Δ|={case.get('abs_diff', '?'):.1f}  "
                    f"judge={case.get('judge_score', '?')}  "
                    f"human={case.get('human_score', '?')}  "
                    f"category={case.get('category', 'N/A')}"
                )

        return "\n".join(lines)


class MetaAuditor:
    """Evaluates the quality of an LLM judge against human annotations.

    This is the core meta-evaluation engine. Given a set of judge outputs
    and corresponding human annotations, it produces a comprehensive
    quality report covering agreement, bias, consistency, and failures.
    """

    def __init__(self, tolerance: int = 1):
        """
        Args:
            tolerance: Margin for near-match accuracy (default ±1).
        """
        self.tolerance = tolerance

    def audit(self, judge_scores: List[float], human_scores: List[float],
              categories: Optional[List[str]] = None,
              metadata: Optional[List[Dict]] = None) -> MetaAuditReport:
        """Run a full meta-audit comparing judge to human annotations.

        Args:
            judge_scores: Scores from the LLM judge.
            human_scores: Ground-truth human annotation scores.
            categories: Optional category labels for per-category breakdown.
            metadata: Optional per-item metadata for failure analysis.

        Returns:
            MetaAuditReport with comprehensive quality analysis.
        """
        report = MetaAuditReport()

        # Overall agreement
        report.overall_agreement = AgreementMetrics.compute(
            judge_scores, human_scores, tolerance=self.tolerance
        )

        # Per-category breakdown
        if categories:
            report.category_agreement = self._category_breakdown(
                judge_scores, human_scores, categories
            )

        # Bias analysis
        report.bias_analysis = self._analyze_bias(
            judge_scores, human_scores, categories
        )

        # Failure analysis (biggest disagreements)
        report.failure_analysis = self._failure_analysis(
            judge_scores, human_scores, categories, metadata
        )

        # Position bias (if available in metadata)
        if metadata:
            report.position_bias_rate = self._compute_position_bias(metadata)

        return report

    def audit_consistency(self, repeated_scores: Dict[str, List[float]]
                          ) -> Dict[str, float]:
        """Evaluate self-consistency of a judge across repeated evaluations.

        Args:
            repeated_scores: Mapping of item_id → list of scores from
                             repeated evaluations of the same item.

        Returns:
            Dict with consistency metrics:
              - mean_std: Average standard deviation across items.
              - max_range: Maximum score range observed for any single item.
              - pct_consistent: Percentage of items with zero score variance.
        """
        stds = []
        ranges = []

        for item_id, scores in repeated_scores.items():
            if len(scores) < 2:
                continue
            arr = np.array(scores)
            stds.append(np.std(arr))
            ranges.append(np.max(arr) - np.min(arr))

        if not stds:
            return {"mean_std": 0.0, "max_range": 0.0, "pct_consistent": 1.0}

        return {
            "mean_std": float(np.mean(stds)),
            "max_range": float(np.max(ranges)),
            "pct_consistent": float(np.mean([s == 0 for s in stds])),
        }

    def _category_breakdown(self, judge_scores: List[float],
                             human_scores: List[float],
                             categories: List[str]) -> Dict[str, AgreementReport]:
        """Compute per-category agreement metrics."""
        df = pd.DataFrame({
            "judge": judge_scores,
            "human": human_scores,
            "category": categories,
        })

        results = {}
        for cat, group in df.groupby("category"):
            if len(group) < 3:
                continue
            results[cat] = AgreementMetrics.compute(
                group["judge"].tolist(),
                group["human"].tolist(),
                tolerance=self.tolerance,
            )
        return results

    def _analyze_bias(self, judge_scores: List[float],
                       human_scores: List[float],
                       categories: Optional[List[str]] = None) -> Dict[str, Any]:
        """Detect systematic biases in judge scoring."""
        j = np.array(judge_scores)
        h = np.array(human_scores)
        valid = (j >= 0) & (h >= 0)
        j, h = j[valid], h[valid]

        diff = j - h
        analysis = {
            "mean_bias": float(np.mean(diff)),
            "median_bias": float(np.median(diff)),
            "std_bias": float(np.std(diff)),
            "pct_over_scoring": float(np.mean(diff > 0)),
            "pct_under_scoring": float(np.mean(diff < 0)),
            "severity_skew": "lenient" if np.mean(diff) > 0.5 else
                             "harsh" if np.mean(diff) < -0.5 else "balanced",
        }

        # Score-range bias: does the judge compress or expand the scale?
        analysis["judge_score_std"] = float(np.std(j))
        analysis["human_score_std"] = float(np.std(h))
        analysis["scale_usage"] = (
            "compressed" if np.std(j) < np.std(h) * 0.8 else
            "expanded" if np.std(j) > np.std(h) * 1.2 else "similar"
        )

        return analysis

    def _failure_analysis(self, judge_scores: List[float],
                           human_scores: List[float],
                           categories: Optional[List[str]] = None,
                           metadata: Optional[List[Dict]] = None
                           ) -> List[Dict]:
        """Identify the largest judge-human disagreements."""
        n = len(judge_scores)
        cases = []

        for i in range(n):
            if judge_scores[i] < 0 or human_scores[i] < 0:
                continue
            abs_diff = abs(judge_scores[i] - human_scores[i])
            case = {
                "index": i,
                "judge_score": judge_scores[i],
                "human_score": human_scores[i],
                "abs_diff": abs_diff,
                "category": categories[i] if categories else "N/A",
            }
            if metadata and i < len(metadata):
                case["metadata"] = metadata[i]
            cases.append(case)

        cases.sort(key=lambda x: x["abs_diff"], reverse=True)
        return cases[:10]  # Top 10 disagreements

    def _compute_position_bias(self, metadata: List[Dict]) -> Optional[float]:
        """Compute position bias rate for pairwise evaluations."""
        position_bias_flags = [
            m.get("position_bias_detected", None)
            for m in metadata if m
        ]
        flags = [f for f in position_bias_flags if f is not None]
        if not flags:
            return None
        return float(np.mean(flags))
