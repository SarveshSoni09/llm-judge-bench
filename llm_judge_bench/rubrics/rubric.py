"""Rubric system for configurable evaluation criteria.

Rubrics define the evaluation dimensions, scoring scales, and quality
descriptors that judges use to assess model responses. They are loaded
from YAML files, enabling non-technical stakeholders to define and
iterate on evaluation criteria without code changes.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml


@dataclass
class Rubric:
    """A structured evaluation rubric.

    Attributes:
        name: Unique identifier for this rubric.
        description: High-level description of what this rubric evaluates.
        dimensions: List of evaluation dimension names (e.g., ["accuracy", "clarity"]).
        dimension_descriptions: Description of each dimension (parallel to dimensions).
        scale_min: Minimum score value (inclusive).
        scale_max: Maximum score value (inclusive).
        scale_labels: Human-readable labels for scale points (e.g., {"1": "Poor", "10": "Excellent"}).
        dimension_weights: Optional weights for computing weighted overall score.
    """
    name: str
    description: str
    dimensions: List[str]
    dimension_descriptions: List[str]
    scale_min: int = 1
    scale_max: int = 10
    scale_labels: Dict[str, str] = field(default_factory=dict)
    dimension_weights: Dict[str, float] = field(default_factory=dict)

    def weighted_score(self, dimension_scores: Dict[str, float]) -> float:
        """Compute a weighted overall score from dimension scores.

        If no weights are specified, uses equal weighting.

        Args:
            dimension_scores: Mapping of dimension name → score.

        Returns:
            Weighted average score across available dimensions.
        """
        if not dimension_scores:
            return 0.0

        total_weight = 0.0
        weighted_sum = 0.0

        for dim, score in dimension_scores.items():
            weight = self.dimension_weights.get(dim, 1.0)
            weighted_sum += score * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    @classmethod
    def from_yaml(cls, path: str) -> "Rubric":
        """Load a rubric from a YAML configuration file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        dims = data.get("dimensions", [])
        dim_names = [d["name"] for d in dims]
        dim_descs = [d.get("description", "") for d in dims]
        dim_weights = {d["name"]: d.get("weight", 1.0) for d in dims}

        return cls(
            name=data["name"],
            description=data.get("description", ""),
            dimensions=dim_names,
            dimension_descriptions=dim_descs,
            scale_min=data.get("scale", {}).get("min", 1),
            scale_max=data.get("scale", {}).get("max", 10),
            scale_labels=data.get("scale", {}).get("labels", {}),
            dimension_weights=dim_weights,
        )


class RubricRegistry:
    """Registry for managing multiple rubrics.

    Loads all YAML rubrics from a directory and provides lookup by name.
    """

    def __init__(self):
        self._rubrics: Dict[str, Rubric] = {}

    def register(self, rubric: Rubric):
        """Register a rubric by name."""
        self._rubrics[rubric.name] = rubric

    def get(self, name: str) -> Rubric:
        """Retrieve a rubric by name."""
        if name not in self._rubrics:
            raise KeyError(f"Rubric '{name}' not found. Available: {list(self._rubrics.keys())}")
        return self._rubrics[name]

    def load_directory(self, directory: str):
        """Load all YAML rubric files from a directory."""
        path = Path(directory)
        for yaml_file in path.glob("*.yaml"):
            rubric = Rubric.from_yaml(str(yaml_file))
            self.register(rubric)

    def list_rubrics(self) -> List[str]:
        """Return names of all registered rubrics."""
        return list(self._rubrics.keys())

    @classmethod
    def with_defaults(cls) -> "RubricRegistry":
        """Create a registry pre-loaded with built-in rubrics."""
        registry = cls()
        default_dir = Path(__file__).parent.parent.parent / "config"
        if default_dir.exists():
            registry.load_directory(str(default_dir))
        return registry
