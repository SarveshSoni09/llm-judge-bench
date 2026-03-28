"""Unit tests for agreement metrics and meta-audit pipeline."""

import unittest
import numpy as np

from llm_judge_bench.metrics.agreement import AgreementMetrics
from llm_judge_bench.meta_audit.evaluator import MetaAuditor
from llm_judge_bench.rubrics.rubric import Rubric, RubricRegistry
from llm_judge_bench.sampling.strategies import (
    RandomSampler, StratifiedSampler, ConfidenceBasedSampler
)


class TestAgreementMetrics(unittest.TestCase):
    """Test inter-rater agreement metric computations."""

    def test_perfect_agreement(self):
        """Identical scores should yield perfect agreement."""
        scores = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        report = AgreementMetrics.compute(scores, scores)

        self.assertEqual(report.exact_accuracy, 1.0)
        self.assertEqual(report.near_accuracy, 1.0)
        self.assertEqual(report.mae, 0.0)
        self.assertEqual(report.rmse, 0.0)
        self.assertEqual(report.bias, 0.0)
        self.assertAlmostEqual(report.pearson_r, 1.0, places=5)
        self.assertEqual(report.n_samples, 10)

    def test_no_agreement(self):
        """Opposite scores should yield poor agreement."""
        scores_a = [1.0, 2.0, 3.0, 4.0, 5.0]
        scores_b = [10.0, 9.0, 8.0, 7.0, 6.0]
        report = AgreementMetrics.compute(scores_a, scores_b)

        self.assertEqual(report.exact_accuracy, 0.0)
        self.assertLess(report.pearson_r, -0.9)
        self.assertGreater(report.mae, 3.0)

    def test_partial_agreement(self):
        """Correlated scores with noise should yield moderate agreement."""
        np.random.seed(42)
        human = [3, 5, 7, 9, 2, 4, 6, 8, 1, 10]
        judge = [4, 5, 6, 8, 3, 4, 7, 7, 2, 9]
        report = AgreementMetrics.compute(judge, human)

        self.assertGreater(report.pearson_r, 0.8)
        self.assertGreater(report.near_accuracy, 0.5)
        self.assertEqual(report.n_samples, 10)

    def test_handles_sentinel_values(self):
        """Scores with -1 sentinel (parse failure) should be filtered."""
        judge = [5.0, -1.0, 7.0, 8.0]
        human = [5.0, 6.0, -1.0, 8.0]
        report = AgreementMetrics.compute(judge, human)

        # Only 2 valid pairs: (5,5) and (8,8)
        self.assertEqual(report.n_samples, 2)

    def test_krippendorff_perfect(self):
        """Perfect agreement should give Krippendorff's Alpha ≈ 1."""
        scores = [1.0, 5.0, 10.0, 3.0, 7.0]
        report = AgreementMetrics.compute(scores, scores)
        self.assertAlmostEqual(report.krippendorffs_alpha, 1.0, places=3)

    def test_bias_detection(self):
        """Systematic positive bias should be detected."""
        human = [3.0, 5.0, 7.0, 4.0, 6.0]
        judge = [5.0, 7.0, 9.0, 6.0, 8.0]  # All +2
        report = AgreementMetrics.compute(judge, human)

        self.assertAlmostEqual(report.bias, 2.0, places=1)


class TestMetaAuditor(unittest.TestCase):
    """Test meta-audit pipeline."""

    def test_basic_audit(self):
        """Meta-audit should produce a complete report."""
        judge = [5, 6, 7, 8, 4, 5, 7, 9, 3, 8]
        human = [5, 5, 8, 8, 3, 6, 7, 8, 2, 9]
        categories = ["A", "A", "B", "B", "A", "B", "A", "B", "A", "B"]

        auditor = MetaAuditor()
        report = auditor.audit(judge, human, categories)

        self.assertIsNotNone(report.overall_agreement)
        self.assertIn("A", report.category_agreement)
        self.assertIn("B", report.category_agreement)
        self.assertGreater(len(report.failure_analysis), 0)
        self.assertIn("mean_bias", report.bias_analysis)

    def test_consistency_audit(self):
        """Self-consistency with identical repeated scores should be perfect."""
        repeated = {
            "item_0": [5.0, 5.0, 5.0],
            "item_1": [8.0, 8.0, 8.0],
        }
        auditor = MetaAuditor()
        result = auditor.audit_consistency(repeated)

        self.assertEqual(result["mean_std"], 0.0)
        self.assertEqual(result["pct_consistent"], 1.0)


class TestRubrics(unittest.TestCase):
    """Test rubric loading and functionality."""

    def test_load_default_rubrics(self):
        """Default rubrics should load successfully."""
        registry = RubricRegistry.with_defaults()
        rubrics = registry.list_rubrics()

        self.assertIn("helpfulness", rubrics)
        self.assertIn("safety", rubrics)
        self.assertIn("factual_accuracy", rubrics)

    def test_rubric_weighted_score(self):
        """Weighted score computation should respect dimension weights."""
        rubric = Rubric(
            name="test",
            description="Test rubric",
            dimensions=["a", "b"],
            dimension_descriptions=["dim a", "dim b"],
            dimension_weights={"a": 2.0, "b": 1.0},
        )

        score = rubric.weighted_score({"a": 10.0, "b": 4.0})
        expected = (10.0 * 2.0 + 4.0 * 1.0) / (2.0 + 1.0)
        self.assertAlmostEqual(score, expected, places=5)


class TestSampling(unittest.TestCase):
    """Test sampling strategies."""

    def test_random_sampler(self):
        """Random sampler should select correct number of items."""
        sampler = RandomSampler()
        report = sampler.sample(100, 10)
        self.assertEqual(report.n_selected, 10)
        self.assertEqual(len(report.selected_indices), 10)

    def test_stratified_sampler(self):
        """Stratified sampler should cover all categories."""
        categories = ["A"] * 30 + ["B"] * 50 + ["C"] * 20
        sampler = StratifiedSampler()
        report = sampler.sample(categories, n_select=20, min_per_category=2)

        self.assertGreaterEqual(report.category_counts.get("A", 0), 2)
        self.assertGreaterEqual(report.category_counts.get("B", 0), 2)
        self.assertGreaterEqual(report.category_counts.get("C", 0), 2)

    def test_confidence_sampler(self):
        """Confidence-based sampler should select correct number."""
        scores = [5.0, 3.0, 8.0, 5.0, 2.0, 9.0, 5.0, 4.0, 6.0, 5.0]
        sampler = ConfidenceBasedSampler()
        report = sampler.sample(scores, n_select=5)
        self.assertEqual(report.n_selected, 5)


if __name__ == "__main__":
    unittest.main()
