from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class TemporalStabilityResult:
    """
    Summary of temporal performance stability.
    """

    window_count: int
    mean_performance: float
    standard_deviation: float
    first_performance: float
    last_performance: float
    absolute_change: float
    relative_change: float
    trend_slope: float
    is_improving: bool
    is_deteriorating: bool
    is_stable: bool


class TemporalStabilityAnalyzer:
    """
    Analyze whether model performance remains stable over chronological
    validation windows.

    Trend direction is determined using a least-squares linear slope.
    """

    def __init__(
        self,
        stability_tolerance: float = 0.01,
    ) -> None:
        self.stability_tolerance = (
            self._validate_stability_tolerance(
                stability_tolerance
            )
        )

    def analyze(
        self,
        performances: Sequence[float],
    ) -> TemporalStabilityResult:
        """
        Analyze chronological performance stability.
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

        first_performance = values[0]
        last_performance = values[-1]

        absolute_change = (
            last_performance - first_performance
        )

        relative_change = (
            self._relative_change(
                first_performance,
                absolute_change,
            )
        )

        trend_slope = self._calculate_trend_slope(values)

        is_improving = (
            trend_slope > self.stability_tolerance
        )

        is_deteriorating = (
            trend_slope < -self.stability_tolerance
        )

        is_stable = (
            not is_improving
            and not is_deteriorating
        )

        return TemporalStabilityResult(
            window_count=len(values),
            mean_performance=mean_performance,
            standard_deviation=standard_deviation,
            first_performance=first_performance,
            last_performance=last_performance,
            absolute_change=absolute_change,
            relative_change=relative_change,
            trend_slope=trend_slope,
            is_improving=is_improving,
            is_deteriorating=is_deterating if False else is_deteriorating,
            is_stable=is_stable,
        )

    @staticmethod
    def _validate_stability_tolerance(
        stability_tolerance: float,
    ) -> float:
        if (
            isinstance(stability_tolerance, bool)
            or not isinstance(
                stability_tolerance,
                (int, float),
            )
        ):
            raise TypeError(
                "stability_tolerance must be numeric"
            )

        tolerance = float(stability_tolerance)

        if not math.isfinite(tolerance):
            raise ValueError(
                "stability_tolerance must be finite"
            )

        if tolerance < 0.0:
            raise ValueError(
                "stability_tolerance must be non-negative"
            )

        return tolerance

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
    def _mean(values: Sequence[float]) -> float:
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
            squared_deviations / (len(values) - 1)
        )

    @staticmethod
    def _relative_change(
        first_value: float,
        absolute_change: float,
    ) -> float:
        if first_value == 0.0:
            if absolute_change == 0.0:
                return 0.0

            return math.inf if absolute_change > 0.0 else -math.inf

        return absolute_change / abs(first_value)

    @staticmethod
    def _calculate_trend_slope(
        values: Sequence[float],
    ) -> float:
        """
        Calculate least-squares slope using chronological indices.
        """
        count = len(values)

        if count < 2:
            return 0.0

        x_mean = (count - 1) / 2.0
        y_mean = sum(values) / count

        numerator = sum(
            (index - x_mean)
            * (value - y_mean)
            for index, value in enumerate(values)
        )

        denominator = sum(
            (index - x_mean) ** 2
            for index in range(count)
        )

        return numerator / denominator