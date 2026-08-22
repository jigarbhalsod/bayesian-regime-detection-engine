from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AnalysisResult:
    """
    Standard result returned by a financial analyzer.
    """

    records: list[dict[str, Any]]
    metrics: dict[str, Any] = field(default_factory=dict)
    metric_names: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)