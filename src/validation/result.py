from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ValidationResult:
    """
    Standardized output produced by validation components.
    """

    validator_name: str
    is_valid: bool
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.validator_name, str):
            raise TypeError("validator_name must be a string")

        if not self.validator_name.strip():
            raise ValueError("validator_name cannot be empty")

        if not isinstance(self.is_valid, bool):
            raise TypeError("is_valid must be a boolean")

        if not isinstance(self.metrics, dict):
            raise TypeError("metrics must be a dictionary")

        for name, value in self.metrics.items():
            if not isinstance(name, str):
                raise TypeError("metric names must be strings")

            if not name.strip():
                raise ValueError("metric names cannot be empty")

            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("metric values must be numeric")

        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dictionary")