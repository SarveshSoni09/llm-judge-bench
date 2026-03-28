"""Sampling strategies for efficient quality auditing.

When evaluating thousands of model responses, it's often impractical to
have human annotators label every item. These sampling strategies select
representative subsets that maximize audit coverage while minimizing cost.

Strategies:
- Random: Simple random sampling (baseline).
- Stratified: Ensures proportional representation across categories.
- Confidence-based: Prioritizes items where the judge was most uncertain.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


@dataclass
class SamplingReport:
    """Report on the sampling outcome.

    Attributes:
        selected_indices: Indices of selected items.
        strategy: Name of the sampling strategy used.
        n_total: Total number of items available.
        n_selected: Number of items selected.
        category_counts: Per-category count of selected items.
        coverage_estimate: Estimated population coverage.
    """
    selected_indices: List[int]
    strategy: str
    n_total: int
    n_selected: int
    category_counts: Dict[str, int] = field(default_factory=dict)
    coverage_estimate: float = 0.0


class RandomSampler:
    """Simple random sampling without replacement."""

    def sample(self, n_total: int, n_select: int,
               seed: int = 42) -> SamplingReport:
        """Select n_select items randomly from n_total.

        Args:
            n_total: Total number of items.
            n_select: Number of items to select.
            seed: Random seed for reproducibility.
        """
        rng = np.random.RandomState(seed)
        n_select = min(n_select, n_total)
        indices = rng.choice(n_total, size=n_select, replace=False).tolist()

        return SamplingReport(
            selected_indices=sorted(indices),
            strategy="random",
            n_total=n_total,
            n_selected=n_select,
            coverage_estimate=n_select / n_total if n_total > 0 else 0.0,
        )


class StratifiedSampler:
    """Stratified sampling ensuring proportional category representation.

    Guarantees that the audit sample reflects the true distribution of
    categories (e.g., question types, difficulty levels) in the full dataset.
    """

    def sample(self, categories: List[str], n_select: int,
               min_per_category: int = 1,
               seed: int = 42) -> SamplingReport:
        """Select items with proportional category representation.

        Args:
            categories: Category label for each item.
            n_select: Total number of items to select.
            min_per_category: Minimum items from each category.
            seed: Random seed for reproducibility.
        """
        rng = np.random.RandomState(seed)
        n_total = len(categories)
        n_select = min(n_select, n_total)

        # Group indices by category
        cat_indices: Dict[str, List[int]] = {}
        for i, cat in enumerate(categories):
            cat_indices.setdefault(cat, []).append(i)

        # Compute proportional allocation
        n_categories = len(cat_indices)
        remaining = n_select
        allocation: Dict[str, int] = {}

        # First pass: guarantee minimums
        for cat, indices in cat_indices.items():
            alloc = min(min_per_category, len(indices))
            allocation[cat] = alloc
            remaining -= alloc

        # Second pass: proportional allocation of remainder
        if remaining > 0:
            total_available = sum(
                len(indices) - allocation[cat]
                for cat, indices in cat_indices.items()
            )
            for cat, indices in cat_indices.items():
                available = len(indices) - allocation[cat]
                proportion = available / total_available if total_available > 0 else 0
                extra = int(remaining * proportion)
                allocation[cat] += min(extra, available)

        # Select indices
        selected: List[int] = []
        cat_counts: Dict[str, int] = {}

        for cat, indices in cat_indices.items():
            n_cat = min(allocation.get(cat, 0), len(indices))
            chosen = rng.choice(indices, size=n_cat, replace=False).tolist()
            selected.extend(chosen)
            cat_counts[cat] = n_cat

        return SamplingReport(
            selected_indices=sorted(selected),
            strategy="stratified",
            n_total=n_total,
            n_selected=len(selected),
            category_counts=cat_counts,
            coverage_estimate=len(selected) / n_total if n_total > 0 else 0.0,
        )


class ConfidenceBasedSampler:
    """Confidence-based sampling that prioritizes uncertain judge outputs.

    Items where the judge expressed low confidence (via score variance
    across dimensions, or hedge words in reasoning) are prioritized
    for human audit, maximizing the information gain per audit dollar.
    """

    def sample(self, judge_scores: List[float],
               dimension_scores: Optional[List[Dict[str, float]]] = None,
               n_select: int = 50,
               uncertainty_threshold: float = 0.3,
               seed: int = 42) -> SamplingReport:
        """Select items with highest judge uncertainty for audit.

        Uncertainty is estimated from:
        1. Distance to scale midpoint (scores near middle → uncertain).
        2. Variance across dimensions (high variance → inconsistent judgment).

        Args:
            judge_scores: Overall judge scores.
            dimension_scores: Optional per-dimension scores for each item.
            n_select: Number of items to select.
            uncertainty_threshold: Minimum uncertainty to be eligible.
            seed: Random seed for reproducibility.
        """
        rng = np.random.RandomState(seed)
        n_total = len(judge_scores)
        n_select = min(n_select, n_total)

        uncertainties = np.zeros(n_total)

        scores = np.array(judge_scores)
        valid = scores >= 0

        if valid.sum() > 0:
            # Component 1: distance to midpoint (normalized)
            s_min, s_max = scores[valid].min(), scores[valid].max()
            midpoint = (s_min + s_max) / 2
            scale_range = (s_max - s_min) if s_max > s_min else 1.0
            midpoint_dist = 1.0 - np.abs(scores - midpoint) / (scale_range / 2)
            midpoint_dist = np.clip(midpoint_dist, 0, 1)
            uncertainties += midpoint_dist * 0.5

        # Component 2: dimension score variance
        if dimension_scores:
            for i, dim_scores in enumerate(dimension_scores):
                if dim_scores:
                    vals = list(dim_scores.values())
                    if len(vals) > 1:
                        uncertainties[i] += np.std(vals) / 5.0 * 0.5

        # Rank by uncertainty and select top-n
        ranked = np.argsort(-uncertainties)
        selected = ranked[:n_select].tolist()

        return SamplingReport(
            selected_indices=sorted(selected),
            strategy="confidence_based",
            n_total=n_total,
            n_selected=len(selected),
            coverage_estimate=len(selected) / n_total if n_total > 0 else 0.0,
        )
