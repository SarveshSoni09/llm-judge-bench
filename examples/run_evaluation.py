"""Example: Run a full pointwise evaluation pipeline with meta-audit.

Usage:
    export GOOGLE_API_KEY=your_key_here
    python examples/run_evaluation.py
"""

import json
from pathlib import Path

from llm_judge_bench.providers.gemini import GeminiProvider
from llm_judge_bench.rubrics.rubric import RubricRegistry
from llm_judge_bench.pipeline import EvaluationPipeline
from llm_judge_bench.sampling.strategies import StratifiedSampler


def main():
    # 1. Initialize provider and rubric
    provider = GeminiProvider(model="gemini-2.5-flash")  # Uses GOOGLE_API_KEY env var
    registry = RubricRegistry.with_defaults()
    rubric = registry.get("helpfulness")

    print(f"Rubric: {rubric.name}")
    print(f"Dimensions: {rubric.dimensions}")
    print(f"Scale: {rubric.scale_min}-{rubric.scale_max}")

    # 2. Load dataset
    data_path = Path(__file__).parent.parent / "data" / "mt_bench_sample.json"
    pipeline = EvaluationPipeline(provider, rubric, judge_type="pointwise")
    data = pipeline.load_data(str(data_path))
    print(f"\nLoaded {len(data)} evaluation items")

    # 3. Optional: Use stratified sampling for a subset
    categories = [item["category"] for item in data]
    sampler = StratifiedSampler()
    sample_report = sampler.sample(categories, n_select=10, min_per_category=2)
    print(f"Sampled {sample_report.n_selected} items: {sample_report.category_counts}")

    sampled_data = [data[i] for i in sample_report.selected_indices]

    # 4. Run judge evaluations
    print("\nRunning pointwise judge evaluations...")
    results = pipeline.run(sampled_data)

    # 5. Display results
    df = pipeline.to_dataframe(sampled_data)
    print("\n--- Results ---")
    print(df[["category", "human_score", "judge_score", "abs_diff"]].to_string())

    # 6. Run meta-audit
    print("\n--- Meta-Audit ---")
    report = pipeline.meta_audit(sampled_data)
    print(report.summary())

    # 7. Save results
    output_path = Path(__file__).parent.parent / "outputs"
    output_path.mkdir(exist_ok=True)
    pipeline.save_results(str(output_path / "evaluation_results.json"), sampled_data)
    print(f"\nResults saved to {output_path / 'evaluation_results.json'}")


if __name__ == "__main__":
    main()
