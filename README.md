# LLM-Judge-Bench

A meta-evaluation framework for LLM-as-a-Judge systems. Build, evaluate, and audit LLM judges with rigorous statistical metrics.

## Why This Exists

LLM-as-a-Judge is becoming the standard for evaluating model quality at scale. But **how do you evaluate the evaluator?** This framework answers that question with:

- **Multiple judge architectures** (pointwise, pairwise, reference-based)
- **Configurable YAML rubrics** for non-technical stakeholder collaboration
- **Inter-rater reliability metrics** (Cohen's Kappa, Krippendorff's Alpha, correlations)
- **Meta-audit pipeline** that detects bias, inconsistency, and failure modes
- **Intelligent sampling strategies** for cost-efficient human auditing

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Pointwise  │     │   Pairwise   │     │  Reference   │
│    Judge     │     │    Judge     │     │    Judge     │
│  (absolute   │     │  (A vs B +   │     │  (vs gold    │
│   scoring)   │     │  position    │     │   standard)  │
│              │     │  bias ctrl)  │     │              │
└──────┬───────┘     └──────┬───────┘     └──────┬───────┘
       │                    │                    │
       └────────────────────┼────────────────────┘
                            │
                    ┌───────▼───────┐
                    │  YAML Rubrics │
                    │  (dimensions, │
                    │   weights,    │
                    │   scales)     │
                    └───────┬───────┘
                            │
              ┌─────────────┼─────────────┐
              │             │             │
      ┌───────▼──────┐ ┌───▼────┐ ┌──────▼───────┐
      │  Agreement   │ │  Meta  │ │   Sampling   │
      │  Metrics     │ │ Audit  │ │  Strategies  │
      │ (κ, α, r, ρ)│ │Pipeline│ │(random, strat│
      │              │ │        │ │ confidence)  │
      └──────────────┘ └────────┘ └──────────────┘
```

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Set API key (Gemini free tier)
export GOOGLE_API_KEY=your_key_here

# Run example evaluation
python examples/run_evaluation.py

# Or explore the interactive notebook
jupyter notebook notebooks/demo.ipynb
```

## Judge Architectures

### Pointwise Judge
Scores a single response on absolute quality criteria (1-10 scale). Used by MT-Bench.

```python
from llm_judge_bench.judges import PointwiseJudge
from llm_judge_bench.providers import GeminiProvider
from llm_judge_bench.rubrics import RubricRegistry

provider = GeminiProvider()
rubric = RubricRegistry.with_defaults().get("helpfulness")
judge = PointwiseJudge(provider, rubric)

result = judge.evaluate(
    question="Explain quantum computing",
    response="Quantum computing uses qubits..."
)
print(f"Score: {result.score}, Dimensions: {result.dimension_scores}")
```

### Pairwise Judge
Compares two responses with **position-bias mitigation** (evaluates both orderings).

```python
from llm_judge_bench.judges import PairwiseJudge

judge = PairwiseJudge(provider, rubric, swap_mitigation=True)
result = judge.evaluate(
    question="Explain quantum computing",
    response_a="Quantum computing uses qubits...",
    response_b="Quantum computers are fast..."
)
print(f"Winner: {result.metadata['verdict']}")
print(f"Position bias detected: {result.metadata['position_bias_detected']}")
```

### Reference-Based Judge
Evaluates responses against gold-standard reference answers. Detects hallucinations.

```python
from llm_judge_bench.judges import ReferenceBasedJudge

judge = ReferenceBasedJudge(provider, rubric)
result = judge.evaluate(
    question="What is the capital of France?",
    response="The capital of France is Paris, located on the Seine river.",
    reference="Paris is the capital of France."
)
```

## Configurable Rubrics (YAML)

Define evaluation criteria without code changes:

```yaml
# config/helpfulness.yaml
name: helpfulness
description: Evaluates overall response quality
scale:
  min: 1
  max: 10
dimensions:
  - name: Accuracy
    description: Factual correctness
    weight: 2.0
  - name: Completeness
    description: Covers all aspects of the question
    weight: 1.5
  - name: Clarity
    description: Well-organized and easy to understand
    weight: 1.0
```

## Meta-Audit Pipeline

The killer feature: evaluate how good your judge actually is.

```python
from llm_judge_bench.meta_audit import MetaAuditor

auditor = MetaAuditor(tolerance=1)
report = auditor.audit(
    judge_scores=[8, 7, 9, 3, 6],
    human_scores=[7, 8, 9, 2, 5],
    categories=["writing", "math", "coding", "writing", "math"]
)

print(report.summary())
# Output includes:
# - Cohen's Kappa, Krippendorff's Alpha
# - Per-category agreement breakdown
# - Bias analysis (lenient/harsh/balanced)
# - Largest judge-human disagreements
```

## Sampling Strategies

Efficiently select items for human audit:

```python
from llm_judge_bench.sampling import StratifiedSampler, ConfidenceBasedSampler

# Proportional category coverage
sample = StratifiedSampler().sample(categories, n_select=100, min_per_category=5)

# Prioritize uncertain judge outputs
sample = ConfidenceBasedSampler().sample(judge_scores, n_select=50)
```

## Agreement Metrics

Full suite of inter-rater reliability metrics:

| Metric | Use Case |
|--------|----------|
| Cohen's Kappa | Exact categorical agreement (2 raters) |
| Weighted Kappa | Ordinal scale agreement (penalizes larger disagreements) |
| Krippendorff's Alpha | Multiple raters, handles missing data |
| Pearson r | Linear correlation between scores |
| Spearman ρ | Rank-order correlation |
| Exact/Near Accuracy | Percentage of exact or ±1 matches |
| MAE / RMSE | Error magnitude metrics |
| Bias | Systematic over/under-scoring |

## Running Tests

```bash
python -m pytest tests/ -v
```

## Project Structure

```
llm-judge-bench/
├── config/                    # YAML rubric definitions
│   ├── helpfulness.yaml
│   ├── safety.yaml
│   └── factual_accuracy.yaml
├── data/                      # Benchmark datasets
│   └── mt_bench_sample.json
├── llm_judge_bench/           # Core library
│   ├── judges/                # Judge architectures
│   │   ├── pointwise.py
│   │   ├── pairwise.py
│   │   └── reference.py
│   ├── metrics/               # Agreement metrics
│   │   └── agreement.py
│   ├── meta_audit/            # Meta-evaluation pipeline
│   │   └── evaluator.py
│   ├── rubrics/               # Rubric system
│   │   └── rubric.py
│   ├── sampling/              # Audit sampling strategies
│   │   └── strategies.py
│   ├── providers/             # LLM provider integrations
│   │   ├── base.py
│   │   └── gemini.py
│   └── pipeline.py            # Pipeline orchestrator
├── notebooks/
│   └── demo.ipynb             # Interactive demo notebook
├── examples/
│   └── run_evaluation.py      # Example script
└── tests/
    └── test_metrics.py        # Unit tests
```

## License

MIT
