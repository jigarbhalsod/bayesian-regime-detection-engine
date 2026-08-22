from dataclasses import dataclass, field
from typing import Any


@dataclass
class RegimeResult:
    """
    Structured output returned by regime detectors.
    """

    records: list[dict[str, Any]] = field(default_factory=list)

    regime_labels: list[Any] = field(default_factory=list)

    regime_probabilities: list[dict[Any, float]] = field(
        default_factory=list
    )

    confidence_scores: list[float] = field(default_factory=list)

    metrics: dict[str, Any] = field(default_factory=dict)

    metadata: dict[str, Any] = field(default_factory=dict)