from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class StressTestResult:
    """
    Summary of stress and edge-case checks performed on numeric data.
    """

    sample_count: int
    minimum: float
    maximum: float
    mean: float
    standard_deviation: float
    value_range: float
    has_constant_values: bool
    has_extreme_magnitude: bool
    is_finite: bool


class StressTester:
    """
    Perform basic stress and edge-case analysis on numeric sequences.

    The component is intentionally deterministic and reusable by the
    broader validation framework.
    """

    def __init__(
        self,
        extreme_threshold: float = 1_000_000.0,
    ) -> None:
        self.extreme_threshold = (
            self._validate_extreme_threshold(extreme_threshold)
        )

    def analyze(
        self,
        values: Sequence[float],
    ) -> StressTestResult:
        """
        Analyze numeric data for boundary and stress characteristics.
        """
        self._validate_values(values)

        numeric_values = [
            float(value)
            for value in values
        ]

        minimum = min(numeric_values)
        maximum = max(numeric_values)
        mean = self._mean(numeric_values)

        standard_deviation = (
            self._sample_standard_deviation(
                numeric_values,
                mean,
            )
        )

        value_range = maximum - minimum

        has_constant_values = (
            value_range == 0.0
        )

        has_extreme_magnitude = any(
            abs(value) >= self.extreme_threshold
            for value in numeric_values
        )

        return StressTestResult(
            sample_count=len(numeric_values),
            minimum=minimum,
            maximum=maximum,
            mean=mean,
            standard_deviation=standard_deviation,
            value_range=value_range,
            has_constant_values=has_constant_values,
            has_extreme_magnitude=has_extreme_magnitude,
            is_finite=True,
        )

    @staticmethod
    def _validate_extreme_threshold(
        extreme_threshold: float,
    ) -> float:
        if (
            isinstance(extreme_threshold, bool)
            or not isinstance(
                extreme_threshold,
                (int, float),
            )
        ):
            raise TypeError(
                "extreme_threshold must be numeric"
            )

        threshold = float(extreme_threshold)

        if not math.isfinite(threshold):
            raise ValueError(
                "extreme_threshold must be finite"
            )

        if threshold <= 0.0:
            raise ValueError(
                "extreme_threshold must be greater than zero"
            )

        return threshold

    @staticmethod
    def _validate_values(
        values: Sequence[float],
    ) -> None:
        if isinstance(values, (str, bytes)):
            raise TypeError(
                "values must be a sequence, not a string or bytes"
            )

        if not isinstance(values, Sequence):
            raise TypeError(
                "values must be a sequence"
            )

        if len(values) == 0:
            raise ValueError(
                "values cannot be empty"
            )

        for value in values:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
            ):
                raise TypeError(
                    "values must contain only numeric values"
                )

            if not math.isfinite(float(value)):
                raise ValueError(
                    "values must contain only finite values"
                )

    @staticmethod
    def _mean(
        values: Sequence[float],
    ) -> float:
        return sum(values) / len(values)

    @staticmethod
    def _sample_standard_deviation(
        values: Sequence[float],
        mean: float,
    ) -> float:
        if len(values) < 2:
            return 0.0

        squared_deviations = sum(
            (value - mean) ** 2
            for value in values
        )

        return math.sqrt(
            squared_deviations
            / (len(values) - 1)
        )