from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any


def _validate_positive_integer(value: Any, field_name: str) -> int:
    """Validate a positive integer."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be a positive integer.")

    if value <= 0:
        raise ValueError(f"{field_name} must be greater than 0.")

    return value


def _validate_non_negative_finite_number(
    value: Any,
    field_name: str,
) -> float:
    """Validate a finite non-negative numeric value."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a numeric value.")

    value = float(value)

    if not isfinite(value):
        raise ValueError(f"{field_name} must be finite.")

    if value < 0:
        raise ValueError(f"{field_name} must be greater than or equal to 0.")

    return value


def _validate_boolean(value: Any, field_name: str) -> bool:
    """Validate a strict boolean value."""
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a boolean.")

    return value


def _validate_method(value: Any) -> str:
    """Validate and normalize the explanation method."""
    if not isinstance(value, str):
        raise TypeError("method must be a string.")

    value = value.strip()

    if not value:
        raise ValueError("method must not be empty.")

    return value


@dataclass
class ExplainabilityConfig:
    """
    Configuration for explainability components.

    Parameters
    ----------
    top_k:
        Maximum number of top features to retain.
    importance_threshold:
        Minimum feature importance threshold.
    attribution_threshold:
        Minimum absolute attribution threshold.
    normalize:
        Whether explanation scores should be normalized.
    method:
        Name of the explainability method.
    """

    top_k: int = 10
    importance_threshold: float = 0.0
    attribution_threshold: float = 0.0
    normalize: bool = True
    method: str = "default"

    def __post_init__(self) -> None:
        self.top_k = _validate_positive_integer(
            self.top_k,
            "top_k",
        )

        self.importance_threshold = _validate_non_negative_finite_number(
            self.importance_threshold,
            "importance_threshold",
        )

        self.attribution_threshold = _validate_non_negative_finite_number(
            self.attribution_threshold,
            "attribution_threshold",
        )

        self.normalize = _validate_boolean(
            self.normalize,
            "normalize",
        )

        self.method = _validate_method(self.method)