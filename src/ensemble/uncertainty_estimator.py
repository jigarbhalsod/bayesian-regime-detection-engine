from typing import Any, Dict

from .result import EnsembleResult
from .uncertainty import (
    UncertaintyLevel,
    UncertaintyResult,
)


class PredictiveUncertaintyEstimator:
    """
    Estimate predictive uncertainty from an ensemble probability
    distribution.

    The uncertainty score is derived from:

        uncertainty = 1 - max(probability)

    Higher maximum probability means greater predictive confidence
    and therefore lower uncertainty.
    """

    def __init__(
        self,
        low_threshold: float = 0.25,
        high_threshold: float = 0.50,
    ) -> None:
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

        self._validate_thresholds()

    def estimate(
        self,
        result: EnsembleResult,
    ) -> UncertaintyResult:
        """
        Estimate predictive uncertainty from an EnsembleResult.
        """

        if not isinstance(result, EnsembleResult):
            raise TypeError(
                "PredictiveUncertaintyEstimator requires an "
                "EnsembleResult instance."
            )

        probabilities = dict(result.probabilities)

        self._validate_probabilities(probabilities)

        max_probability = max(probabilities.values())

        uncertainty_score = 1.0 - max_probability

        confidence_score = max_probability

        uncertainty_level = self._classify_uncertainty(
            uncertainty_score
        )

        return UncertaintyResult(
            prediction=result.prediction,
            uncertainty_score=uncertainty_score,
            confidence_score=confidence_score,
            uncertainty_level=uncertainty_level,
            probabilities=probabilities,
            components={
                "predictive": uncertainty_score,
            },
            metadata={
                "estimator": "predictive",
                "max_probability": max_probability,
            },
        )

    def _validate_thresholds(self) -> None:
        """
        Validate uncertainty classification thresholds.
        """

        if not isinstance(self.low_threshold, (int, float)):
            raise TypeError(
                "low_threshold must be numeric."
            )

        if not isinstance(self.high_threshold, (int, float)):
            raise TypeError(
                "high_threshold must be numeric."
            )

        if not 0.0 <= self.low_threshold <= 1.0:
            raise ValueError(
                "low_threshold must be between 0.0 and 1.0."
            )

        if not 0.0 <= self.high_threshold <= 1.0:
            raise ValueError(
                "high_threshold must be between 0.0 and 1.0."
            )

        if self.low_threshold > self.high_threshold:
            raise ValueError(
                "low_threshold cannot exceed high_threshold."
            )

    @staticmethod
    def _validate_probabilities(
        probabilities: Dict[Any, float],
    ) -> None:
        """
        Validate the probability distribution before estimation.
        """

        if not probabilities:
            raise ValueError(
                "Cannot estimate uncertainty from empty probabilities."
            )

        total = 0.0

        for label, probability in probabilities.items():
            if not isinstance(probability, (int, float)):
                raise TypeError(
                    f"Probability for '{label}' must be numeric."
                )

            if probability < 0.0:
                raise ValueError(
                    f"Probability for '{label}' cannot be negative."
                )

            if probability > 1.0:
                raise ValueError(
                    f"Probability for '{label}' cannot exceed 1.0."
                )

            total += probability

        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                "Probabilities must sum to 1.0."
            )

    def _classify_uncertainty(
        self,
        uncertainty_score: float,
    ) -> UncertaintyLevel:
        """
        Convert a numeric uncertainty score into a qualitative level.
        """

        if uncertainty_score <= self.low_threshold:
            return UncertaintyLevel.LOW

        if uncertainty_score <= self.high_threshold:
            return UncertaintyLevel.MEDIUM

        return UncertaintyLevel.HIGH