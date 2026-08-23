from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class RobustnessResult:
    """
    Summary of performance robustness across validation windows.
    """

    window_count: int
    mean_performance: float
    standard_deviation: float
    minimum_performance: float
    maximum_performance: float
    performance_range: float
    coefficient_of_variation: float
    stability_threshold: float
    is_stable: bool


class RobustnessAnalyzer:
    """
    Analyze the robustness and stability of model performance across
    multiple validation windows.

    A lower coefficient of variation indicates more stable performance.
    """

    def __init__(
        self,
        stability_threshold: float = 0.10,
    ) -> None:
        self.stability_threshold = (
            self._validate_stability_threshold(
                stability_threshold
            )
        )

    def analyze(
        self,
        performances: Sequence[float],
    ) -> RobustnessResult:
        """
        Analyze performance stability across validation windows.
        """
        self._validate_performances(performances)

        values = [
            float(value)
            for value in performances
        ]

        mean_performance = self._mean(values)

        standard_deviation = (
            self._sample_standard_deviation(
                values,
                mean_performance,
            )
        )

        minimum_performance = min(values)
        maximum_performance = max(values)

        performance_range = (
            maximum_performance
            - minimum_performance
        )

        if mean_performance == 0.0:
            coefficient_of_variation = (
                0.0
                if standard_deviation == 0.0
                else math.inf
            )
        else:
            coefficient_of_variation = (
                standard_deviation
                / abs(mean_performance)
            )

        is_stable = (
            coefficient_of_variation
            <= self.stability_threshold
        )

        return RobustnessResult(
            window_count=len(values),
            mean_performance=mean_performance,
            standard_deviation=standard_deviation,
            minimum_performance=minimum_performance,
            maximum_performance=maximum_performance,
            performance_range=performance_range,
            coefficient_of_variation=coefficient_of_variation,
            stability_threshold=self.stability_threshold,
            is_stable=is_stable,
        )

    @staticmethod
    def _validate_stability_threshold(
        stability_threshold: float,
    ) -> float:
        if (
            isinstance(stability_threshold, bool)
            or not isinstance(
                stability_threshold,
                (int, float),
            )
        ):
            raise TypeError(
                "stability_threshold must be numeric"
            )

        stability_threshold = float(
            stability_threshold
        )

        if not math.isfinite(stability_threshold):
            raise ValueError(
                "stability_threshold must be finite"
            )

        if stability_threshold < 0.0:
            raise ValueError(
                "stability_threshold must be non-negative"
            )

        return stability_threshold

    @staticmethod
    def _validate_performances(
        performances: Sequence[float],
    ) -> None:
        if isinstance(performances, (str, bytes)):
            raise TypeError(
                "performances must be a sequence, "
                "not a string or bytes"
            )

        if not isinstance(performances, Sequence):
            raise TypeError(
                "performances must be a sequence"
            )

        if len(performances) == 0:
            raise ValueError(
                "performances cannot be empty"
            )

        for value in performances:
            if (
                isinstance(value, bool)
                or not isinstance(
                    value,
                    (int, float),
                )
            ):
                raise TypeError(
                    "performance values must be numeric"
                )

            if not math.isfinite(float(value)):
                raise ValueError(
                    "performance values must be finite"
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