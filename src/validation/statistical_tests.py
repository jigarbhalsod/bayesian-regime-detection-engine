from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Sequence


@dataclass(frozen=True)
class StatisticalTestResult:
    """
    Result of a paired statistical significance test.
    """

    sample_size: int
    mean_difference: float
    standard_deviation: float
    standard_error: float
    test_statistic: float
    p_value: float
    alpha: float
    is_significant: bool


class PairedSignificanceTester:
    """
    Dependency-light paired significance tester.

    Compares two aligned samples using a two-sided paired test based
    on the normal approximation.

    A positive mean_difference means model_a performed better than
    model_b on average.
    """

    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = self._validate_alpha(alpha)

    def test(
        self,
        model_a: Sequence[float],
        model_b: Sequence[float],
    ) -> StatisticalTestResult:
        """
        Compare two aligned performance samples.
        """
        self._validate_inputs(model_a, model_b)

        differences = [
            float(a) - float(b)
            for a, b in zip(model_a, model_b)
        ]

        sample_size = len(differences)
        mean_difference = self._mean(differences)

        standard_deviation = self._sample_standard_deviation(
            differences,
            mean_difference,
        )

        standard_error = (
            standard_deviation / math.sqrt(sample_size)
        )

        if standard_error == 0.0:
            if mean_difference == 0.0:
                test_statistic = 0.0
                p_value = 1.0
            else:
                test_statistic = math.inf
                p_value = 0.0
        else:
            test_statistic = (
                mean_difference / standard_error
            )

            normal = NormalDist()
            p_value = 2.0 * (
                1.0
                - normal.cdf(abs(test_statistic))
            )

            p_value = max(0.0, min(1.0, p_value))

        is_significant = p_value < self.alpha

        return StatisticalTestResult(
            sample_size=sample_size,
            mean_difference=mean_difference,
            standard_deviation=standard_deviation,
            standard_error=standard_error,
            test_statistic=test_statistic,
            p_value=p_value,
            alpha=self.alpha,
            is_significant=is_significant,
        )

    @staticmethod
    def _validate_alpha(alpha: float) -> float:
        if isinstance(alpha, bool) or not isinstance(
            alpha,
            (int, float),
        ):
            raise TypeError("alpha must be numeric")

        alpha = float(alpha)

        if not math.isfinite(alpha):
            raise ValueError("alpha must be finite")

        if alpha <= 0.0 or alpha >= 1.0:
            raise ValueError(
                "alpha must be between 0 and 1"
            )

        return alpha

    @staticmethod
    def _validate_inputs(
        model_a: Sequence[float],
        model_b: Sequence[float],
    ) -> None:
        if isinstance(model_a, (str, bytes)):
            raise TypeError(
                "model_a must be a sequence, not a string or bytes"
            )

        if isinstance(model_b, (str, bytes)):
            raise TypeError(
                "model_b must be a sequence, not a string or bytes"
            )

        if not isinstance(model_a, Sequence):
            raise TypeError("model_a must be a sequence")

        if not isinstance(model_b, Sequence):
            raise TypeError("model_b must be a sequence")

        if len(model_a) == 0:
            raise ValueError("model_a cannot be empty")

        if len(model_b) == 0:
            raise ValueError("model_b cannot be empty")

        if len(model_a) != len(model_b):
            raise ValueError(
                "model_a and model_b must have the same length"
            )

        if len(model_a) < 2:
            raise ValueError(
                "at least two paired observations are required"
            )

        for value in [*model_a, *model_b]:
            if isinstance(value, bool) or not isinstance(
                value,
                (int, float),
            ):
                raise TypeError(
                    "model values must be numeric"
                )

            if not math.isfinite(float(value)):
                raise ValueError(
                    "model values must be finite"
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