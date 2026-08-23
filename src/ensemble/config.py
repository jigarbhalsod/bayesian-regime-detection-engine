from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EnsembleConfig:
    """
    Configuration for the model ensemble and uncertainty layer.
    """

    method: str = "weighted_average"
    normalize_weights: bool = True
    min_confidence: float = 0.0
    max_confidence: float = 1.0
    uncertainty_floor: float = 0.0
    uncertainty_ceiling: float = 1.0

    def __post_init__(self) -> None:
        method = self._validate_method(self.method)

        min_confidence = self._validate_probability(
            self.min_confidence,
            "min_confidence",
        )

        max_confidence = self._validate_probability(
            self.max_confidence,
            "max_confidence",
        )

        if min_confidence > max_confidence:
            raise ValueError(
                "min_confidence must be less than or equal "
                "to max_confidence."
            )

        uncertainty_floor = self._validate_probability(
            self.uncertainty_floor,
            "uncertainty_floor",
        )

        uncertainty_ceiling = self._validate_probability(
            self.uncertainty_ceiling,
            "uncertainty_ceiling",
        )

        if uncertainty_floor > uncertainty_ceiling:
            raise ValueError(
                "uncertainty_floor must be less than or equal "
                "to uncertainty_ceiling."
            )

        if not isinstance(self.normalize_weights, bool):
            raise TypeError(
                "normalize_weights must be a boolean."
            )

        object.__setattr__(
            self,
            "method",
            method,
        )

        object.__setattr__(
            self,
            "min_confidence",
            min_confidence,
        )

        object.__setattr__(
            self,
            "max_confidence",
            max_confidence,
        )

        object.__setattr__(
            self,
            "uncertainty_floor",
            uncertainty_floor,
        )

        object.__setattr__(
            self,
            "uncertainty_ceiling",
            uncertainty_ceiling,
        )

    @staticmethod
    def _validate_method(value: Any) -> str:
        if not isinstance(value, str):
            raise TypeError(
                "method must be a string."
            )

        method = value.strip().lower()

        if not method:
            raise ValueError(
                "method cannot be empty."
            )

        allowed_methods = {
            "weighted_average",
            "mean",
        }

        if method not in allowed_methods:
            raise ValueError(
                "method must be one of: "
                "weighted_average, mean."
            )

        return method

    @staticmethod
    def _validate_probability(
        value: Any,
        field_name: str,
    ) -> float:
        if isinstance(value, bool) or not isinstance(
            value,
            (int, float),
        ):
            raise TypeError(
                f"{field_name} must be numeric."
            )

        value = float(value)

        if not math.isfinite(value):
            raise ValueError(
                f"{field_name} must be finite."
            )

        if value < 0.0 or value > 1.0:
            raise ValueError(
                f"{field_name} must be between 0 and 1."
            )

        return value