from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class BootstrapConfidenceInterval:
    """
    Result of bootstrap confidence interval estimation.
    """

    sample_size: int
    original_estimate: float
    bootstrap_mean: float
    standard_error: float
    confidence_level: float
    lower_bound: float
    upper_bound: float
    n_resamples: int


class BootstrapConfidenceIntervalEstimator:
    """
    Dependency-light bootstrap confidence interval estimator.

    Uses bootstrap resampling with replacement and percentile-based
    confidence intervals for the sample mean.
    """

    def __init__(
        self,
        n_resamples: int = 1000,
        confidence_level: float = 0.95,
        random_state: int | None = None,
    ) -> None:
        self.n_resamples = self._validate_n_resamples(
            n_resamples
        )
        self.confidence_level = (
            self._validate_confidence_level(confidence_level)
        )
        self.random_state = self._validate_random_state(
            random_state
        )

    def estimate(
        self,
        samples: Sequence[float],
    ) -> BootstrapConfidenceInterval:
        """
        Estimate a bootstrap confidence interval for the sample mean.
        """
        self._validate_samples(samples)

        numeric_samples = [
            float(value)
            for value in samples
        ]

        original_estimate = self._mean(numeric_samples)

        rng = random.Random(self.random_state)

        bootstrap_estimates = []

        sample_size = len(numeric_samples)

        for _ in range(self.n_resamples):
            resample = [
                rng.choice(numeric_samples)
                for _ in range(sample_size)
            ]

            bootstrap_estimates.append(
                self._mean(resample)
            )

        bootstrap_mean = self._mean(
            bootstrap_estimates
        )

        standard_error = (
            self._sample_standard_deviation(
                bootstrap_estimates,
                bootstrap_mean,
            )
        )

        bootstrap_estimates.sort()

        alpha = 1.0 - self.confidence_level

        lower_quantile = alpha / 2.0
        upper_quantile = 1.0 - lower_quantile

        lower_bound = self._percentile(
            bootstrap_estimates,
            lower_quantile,
        )

        upper_bound = self._percentile(
            bootstrap_estimates,
            upper_quantile,
        )

        return BootstrapConfidenceInterval(
            sample_size=sample_size,
            original_estimate=original_estimate,
            bootstrap_mean=bootstrap_mean,
            standard_error=standard_error,
            confidence_level=self.confidence_level,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            n_resamples=self.n_resamples,
        )

    @staticmethod
    def _validate_n_resamples(
        n_resamples: int,
    ) -> int:
        if isinstance(n_resamples, bool) or not isinstance(
            n_resamples,
            int,
        ):
            raise TypeError("n_resamples must be an integer")

        if n_resamples < 2:
            raise ValueError(
                "n_resamples must be at least 2"
            )

        return n_resamples

    @staticmethod
    def _validate_confidence_level(
        confidence_level: float,
    ) -> float:
        if isinstance(
            confidence_level,
            bool,
        ) or not isinstance(
            confidence_level,
            (int, float),
        ):
            raise TypeError(
                "confidence_level must be numeric"
            )

        confidence_level = float(confidence_level)

        if not math.isfinite(confidence_level):
            raise ValueError(
                "confidence_level must be finite"
            )

        if (
            confidence_level <= 0.0
            or confidence_level >= 1.0
        ):
            raise ValueError(
                "confidence_level must be between 0 and 1"
            )

        return confidence_level

    @staticmethod
    def _validate_random_state(
        random_state: int | None,
    ) -> int | None:
        if random_state is None:
            return None

        if isinstance(random_state, bool) or not isinstance(
            random_state,
            int,
        ):
            raise TypeError(
                "random_state must be an integer or None"
            )

        return random_state

    @staticmethod
    def _validate_samples(
        samples: Sequence[float],
    ) -> None:
        if isinstance(samples, (str, bytes)):
            raise TypeError(
                "samples must be a sequence, not a string or bytes"
            )

        if not isinstance(samples, Sequence):
            raise TypeError("samples must be a sequence")

        if len(samples) == 0:
            raise ValueError("samples cannot be empty")

        for value in samples:
            if isinstance(value, bool) or not isinstance(
                value,
                (int, float),
            ):
                raise TypeError(
                    "sample values must be numeric"
                )

            if not math.isfinite(float(value)):
                raise ValueError(
                    "sample values must be finite"
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
    def _percentile(
        sorted_values: Sequence[float],
        quantile: float,
    ) -> float:
        """
        Linear-interpolated percentile.
        """
        if len(sorted_values) == 1:
            return sorted_values[0]

        position = quantile * (
            len(sorted_values) - 1
        )

        lower_index = math.floor(position)
        upper_index = math.ceil(position)

        if lower_index == upper_index:
            return sorted_values[lower_index]

        weight = position - lower_index

        return (
            sorted_values[lower_index]
            + weight
            * (
                sorted_values[upper_index]
                - sorted_values[lower_index]
            )
        )