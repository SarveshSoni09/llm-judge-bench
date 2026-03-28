"""Inter-rater reliability and agreement metrics.

Pure-numpy implementations of standard metrics for evaluating agreement
between LLM judges and human annotators. Zero external dependencies
beyond numpy, making this portable and easy to audit.

Metrics implemented from scratch:
- Cohen's Kappa (two raters, categorical)
- Weighted Kappa (quadratic weights, ordinal scales)
- Krippendorff's Alpha (multiple raters, handles missing data)
- Pearson and Spearman correlation (with t-distribution p-values)
- Exact and near-match accuracy
- Mean Absolute Error and Root Mean Squared Error
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class AgreementReport:
    """Comprehensive agreement analysis between two rating sets.

    Attributes:
        cohens_kappa: Cohen's Kappa for exact categorical agreement.
        weighted_kappa: Quadratic-weighted Kappa for ordinal data.
        krippendorffs_alpha: Krippendorff's Alpha (handles missing data).
        pearson_r: Pearson correlation coefficient.
        pearson_p: P-value for Pearson correlation.
        spearman_rho: Spearman rank correlation.
        spearman_p: P-value for Spearman correlation.
        exact_accuracy: Fraction of exact score matches.
        near_accuracy: Fraction matching within ±1 of tolerance.
        mae: Mean Absolute Error between scores.
        rmse: Root Mean Squared Error between scores.
        n_samples: Number of samples compared.
        score_distribution_judge: Histogram of judge scores.
        score_distribution_human: Histogram of human scores.
        bias: Mean signed difference (judge - human). Positive = judge scores higher.
    """
    cohens_kappa: float = 0.0
    weighted_kappa: float = 0.0
    krippendorffs_alpha: float = 0.0
    pearson_r: float = 0.0
    pearson_p: float = 1.0
    spearman_rho: float = 0.0
    spearman_p: float = 1.0
    exact_accuracy: float = 0.0
    near_accuracy: float = 0.0
    mae: float = 0.0
    rmse: float = 0.0
    n_samples: int = 0
    bias: float = 0.0
    score_distribution_judge: Dict[int, int] = field(default_factory=dict)
    score_distribution_human: Dict[int, int] = field(default_factory=dict)

    def summary(self) -> str:
        """Human-readable summary of the agreement report."""
        lines = [
            f"Agreement Report (n={self.n_samples})",
            f"{'='*45}",
            f"Cohen's Kappa (exact):      {self.cohens_kappa:+.3f}",
            f"Weighted Kappa (ordinal):    {self.weighted_kappa:+.3f}",
            f"Krippendorff's Alpha:        {self.krippendorffs_alpha:+.3f}",
            f"Pearson r:                   {self.pearson_r:+.3f} (p={self.pearson_p:.4f})",
            f"Spearman ρ:                  {self.spearman_rho:+.3f} (p={self.spearman_p:.4f})",
            f"Exact accuracy:              {self.exact_accuracy:.1%}",
            f"Near accuracy (±1):          {self.near_accuracy:.1%}",
            f"MAE:                         {self.mae:.3f}",
            f"RMSE:                        {self.rmse:.3f}",
            f"Bias (judge - human):        {self.bias:+.3f}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "cohens_kappa": self.cohens_kappa,
            "weighted_kappa": self.weighted_kappa,
            "krippendorffs_alpha": self.krippendorffs_alpha,
            "pearson_r": self.pearson_r,
            "pearson_p": self.pearson_p,
            "spearman_rho": self.spearman_rho,
            "spearman_p": self.spearman_p,
            "exact_accuracy": self.exact_accuracy,
            "near_accuracy": self.near_accuracy,
            "mae": self.mae,
            "rmse": self.rmse,
            "n_samples": self.n_samples,
            "bias": self.bias,
        }


class AgreementMetrics:
    """Compute inter-rater agreement between judge scores and human labels.

    All metrics are implemented from scratch using only numpy for
    maximum portability and auditability.
    """

    @staticmethod
    def compute(judge_scores: List[float], human_scores: List[float],
                tolerance: int = 1) -> AgreementReport:
        """Compute all agreement metrics between two sets of ratings.

        Args:
            judge_scores: Scores assigned by the LLM judge.
            human_scores: Ground-truth scores from human annotators.
            tolerance: Margin for near-match accuracy (default ±1).

        Returns:
            Comprehensive AgreementReport.
        """
        assert len(judge_scores) == len(human_scores), \
            "Score lists must have equal length"

        j = np.array(judge_scores, dtype=float)
        h = np.array(human_scores, dtype=float)

        # Filter out parsing failures (sentinel value -1)
        valid = (j >= 0) & (h >= 0)
        j = j[valid]
        h = h[valid]
        n = len(j)

        if n < 2:
            return AgreementReport(n_samples=n)

        # Convert to int for kappa calculations
        j_int = np.round(j).astype(int)
        h_int = np.round(h).astype(int)

        # Cohen's Kappa (exact agreement)
        kappa = AgreementMetrics._cohens_kappa(h_int, j_int)

        # Weighted Kappa (quadratic weights for ordinal data)
        w_kappa = AgreementMetrics._weighted_kappa(h_int, j_int, weights="quadratic")

        # Krippendorff's Alpha
        alpha = AgreementMetrics._krippendorffs_alpha(j, h)

        # Correlations
        pearson_r, pearson_p = AgreementMetrics._pearson_correlation(j, h)
        spearman_rho, spearman_p = AgreementMetrics._spearman_correlation(j, h)

        # Accuracy metrics
        exact_acc = float(np.mean(j_int == h_int))
        near_acc = float(np.mean(np.abs(j - h) <= tolerance))

        # Error metrics
        mae = float(np.mean(np.abs(h - j)))
        rmse = float(np.sqrt(np.mean((j - h) ** 2)))

        # Bias
        bias = float(np.mean(j - h))

        # Score distributions
        j_dist = dict(zip(*np.unique(j_int, return_counts=True)))
        h_dist = dict(zip(*np.unique(h_int, return_counts=True)))

        return AgreementReport(
            cohens_kappa=float(kappa),
            weighted_kappa=float(w_kappa),
            krippendorffs_alpha=float(alpha),
            pearson_r=float(pearson_r),
            pearson_p=float(pearson_p),
            spearman_rho=float(spearman_rho),
            spearman_p=float(spearman_p),
            exact_accuracy=exact_acc,
            near_accuracy=near_acc,
            mae=mae,
            rmse=rmse,
            n_samples=n,
            bias=bias,
            score_distribution_judge={int(k): int(v) for k, v in j_dist.items()},
            score_distribution_human={int(k): int(v) for k, v in h_dist.items()},
        )

    # ── Pure-numpy implementations ──────────────────────────────────

    @staticmethod
    def _cohens_kappa(y1: np.ndarray, y2: np.ndarray) -> float:
        """Cohen's Kappa for exact categorical agreement.

        κ = (p_o - p_e) / (1 - p_e)
        where p_o = observed agreement, p_e = expected agreement by chance.
        """
        labels = np.union1d(y1, y2)
        n = len(y1)
        if n == 0:
            return 0.0

        # Build confusion matrix
        k = len(labels)
        label_to_idx = {l: i for i, l in enumerate(labels)}
        cm = np.zeros((k, k), dtype=float)
        for a, b in zip(y1, y2):
            cm[label_to_idx[a], label_to_idx[b]] += 1

        # Observed agreement
        p_o = np.trace(cm) / n

        # Expected agreement by chance
        row_sums = cm.sum(axis=1)
        col_sums = cm.sum(axis=0)
        p_e = np.sum(row_sums * col_sums) / (n * n)

        if p_e == 1.0:
            return 1.0 if p_o == 1.0 else 0.0
        return (p_o - p_e) / (1.0 - p_e)

    @staticmethod
    def _weighted_kappa(y1: np.ndarray, y2: np.ndarray,
                         weights: str = "quadratic") -> float:
        """Weighted Cohen's Kappa for ordinal scales.

        Uses quadratic weights: w_ij = 1 - (i-j)² / (k-1)²
        This penalizes larger disagreements more heavily.
        """
        labels = np.sort(np.union1d(y1, y2))
        n = len(y1)
        if n == 0:
            return 0.0

        k = len(labels)
        label_to_idx = {l: i for i, l in enumerate(labels)}

        # Build confusion matrix
        cm = np.zeros((k, k), dtype=float)
        for a, b in zip(y1, y2):
            cm[label_to_idx[a], label_to_idx[b]] += 1

        # Weight matrix
        W = np.zeros((k, k), dtype=float)
        for i in range(k):
            for j in range(k):
                if weights == "quadratic":
                    W[i, j] = (i - j) ** 2 / ((k - 1) ** 2) if k > 1 else 0.0
                else:  # linear
                    W[i, j] = abs(i - j) / (k - 1) if k > 1 else 0.0

        # Observed and expected matrices
        cm_norm = cm / n
        row_sums = cm_norm.sum(axis=1)
        col_sums = cm_norm.sum(axis=0)
        expected = np.outer(row_sums, col_sums)

        weighted_observed = np.sum(W * cm_norm)
        weighted_expected = np.sum(W * expected)

        if weighted_expected == 0:
            return 1.0 if weighted_observed == 0 else 0.0
        return 1.0 - weighted_observed / weighted_expected

    @staticmethod
    def _pearson_correlation(x: np.ndarray, y: np.ndarray
                              ) -> Tuple[float, float]:
        """Pearson correlation with t-test p-value."""
        n = len(x)
        if n < 3:
            return 0.0, 1.0

        mx, my = np.mean(x), np.mean(y)
        dx, dy = x - mx, y - my
        denom = np.sqrt(np.sum(dx**2) * np.sum(dy**2))
        if denom == 0:
            return 0.0, 1.0

        r = float(np.sum(dx * dy) / denom)
        r = np.clip(r, -1.0, 1.0)

        # t-test for significance
        if abs(r) == 1.0:
            return r, 0.0
        t_stat = r * np.sqrt((n - 2) / (1 - r**2))
        p_value = AgreementMetrics._t_distribution_pvalue(t_stat, n - 2)
        return r, p_value

    @staticmethod
    def _spearman_correlation(x: np.ndarray, y: np.ndarray
                               ) -> Tuple[float, float]:
        """Spearman rank correlation (Pearson on ranks)."""
        def _rank(arr):
            order = np.argsort(arr)
            ranks = np.empty_like(order, dtype=float)
            ranks[order] = np.arange(1, len(arr) + 1, dtype=float)
            # Handle ties with average ranks
            for val in np.unique(arr):
                mask = arr == val
                if np.sum(mask) > 1:
                    ranks[mask] = np.mean(ranks[mask])
            return ranks

        return AgreementMetrics._pearson_correlation(_rank(x), _rank(y))

    @staticmethod
    def _t_distribution_pvalue(t_stat: float, df: int) -> float:
        """Approximate two-tailed p-value from t-distribution.

        Uses the regularized incomplete beta function approximation.
        """
        if df <= 0:
            return 1.0
        x = df / (df + t_stat**2)
        # Approximate using the normal distribution for large df
        if df > 30:
            from math import erfc, sqrt
            z = abs(t_stat)
            p = erfc(z / sqrt(2))
            return float(p)
        # For smaller df, use a simple approximation
        # Based on Abramowitz and Stegun approximation
        t = abs(t_stat)
        p = 1.0 / (1.0 + 0.3183099 * t * (1 + t**2 / df) ** (-0.5 * (df + 1)) * df**0.5)
        # Rough two-tailed p-value
        a = 0.5 * df
        p_approx = 2.0 * (1.0 - AgreementMetrics._incomplete_beta_approx(a, 0.5, x))
        return float(np.clip(p_approx, 0.0, 1.0))

    @staticmethod
    def _incomplete_beta_approx(a: float, b: float, x: float) -> float:
        """Simple approximation of regularized incomplete beta function.

        Uses the continued fraction expansion for I_x(a, b).
        Accurate enough for p-value estimation in inter-rater contexts.
        """
        if x <= 0:
            return 0.0
        if x >= 1:
            return 1.0

        # Use normal approximation for reasonable accuracy
        from math import lgamma, exp, sqrt

        try:
            # Beta function normalization
            log_beta = lgamma(a) + lgamma(b) - lgamma(a + b)
            log_front = a * math.log(x) + b * math.log(1 - x) - log_beta - math.log(a)

            # Lentz's continued fraction
            result = 1.0
            c = 1.0
            d = 1.0 / max(1.0 - (a + b) * x / (a + 1), 1e-30)
            result = d

            for m in range(1, 100):
                # Even step
                num = m * (b - m) * x / ((a + 2*m - 1) * (a + 2*m))
                d = 1.0 / max(1.0 + num * d, 1e-30)
                c = max(1.0 + num / c, 1e-30)
                result *= d * c

                # Odd step
                num = -(a + m) * (a + b + m) * x / ((a + 2*m) * (a + 2*m + 1))
                d = 1.0 / max(1.0 + num * d, 1e-30)
                c = max(1.0 + num / c, 1e-30)
                result *= d * c

            return float(np.clip(exp(log_front) * result, 0, 1))
        except (ValueError, OverflowError):
            return 0.5

    @staticmethod
    def _krippendorffs_alpha(scores_a: np.ndarray,
                              scores_b: np.ndarray) -> float:
        """Compute Krippendorff's Alpha for interval data with two raters.

        Uses the standard formulation:
            α = 1 - D_observed / D_expected

        where D_observed is the mean squared difference within units and
        D_expected is the mean squared difference across all value pairs.
        """
        n = len(scores_a)
        if n < 2:
            return 0.0

        # Observed disagreement: mean squared diff within each unit
        d_observed = np.mean((scores_a - scores_b) ** 2)

        # Expected disagreement: all pairwise differences across all values
        all_values = np.concatenate([scores_a, scores_b])
        m = len(all_values)
        mean_sq_diff = 0.0
        for i in range(m):
            for j_idx in range(i + 1, m):
                mean_sq_diff += (all_values[i] - all_values[j_idx]) ** 2
        d_expected = mean_sq_diff / (m * (m - 1) / 2)

        if d_expected == 0:
            return 1.0 if d_observed == 0 else 0.0

        return 1.0 - d_observed / d_expected

    @staticmethod
    def compute_multi_rater(ratings_matrix: np.ndarray) -> float:
        """Compute Krippendorff's Alpha for multiple raters.

        Args:
            ratings_matrix: Shape (n_items, n_raters). Use np.nan for missing.

        Returns:
            Krippendorff's Alpha for the full rater panel.
        """
        n_items, n_raters = ratings_matrix.shape

        # Observed disagreement
        d_o_sum = 0.0
        n_o = 0
        for i in range(n_items):
            values = ratings_matrix[i, ~np.isnan(ratings_matrix[i])]
            m = len(values)
            if m < 2:
                continue
            for a in range(m):
                for b in range(a + 1, m):
                    d_o_sum += (values[a] - values[b]) ** 2
                    n_o += 1

        if n_o == 0:
            return 0.0
        d_observed = d_o_sum / n_o

        # Expected disagreement (efficient via variance)
        all_values = ratings_matrix[~np.isnan(ratings_matrix)]
        m_total = len(all_values)
        if m_total < 2:
            return 0.0

        d_expected = np.var(all_values) * (m_total / (m_total - 1))

        if d_expected == 0:
            return 1.0 if d_observed == 0 else 0.0

        return 1.0 - d_observed / d_expected
