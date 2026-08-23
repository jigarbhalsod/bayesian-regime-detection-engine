import math
from typing import Any, Dict, List, Tuple

from .result import EnsembleResult
from .uncertainty import (
    UncertaintyLevel,
    UncertaintyResult,
)


class ConfidenceEntropyAnalyzer:
    """
    Analyze confidence and entropy from an ensemble probability
    distribution.

    Calculates:
    - maximum probability confidence
    - confidence margin between the top two probabilities
    - normalized Shannon entropy
    - entropy-derived uncertainty score
    """

    def __init__(
        self,
        low_threshold: float = 0.33,
        high_threshold: float = 0.66,
    ) -> None:
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

        self._validate_thresholds()

    def analyze(
        self,
        result: EnsembleResult,
    ) -> UncertaintyResult:
        """
        Analyze confidence and entropy for an ensemble result.
        """

        if not isinstance(result, EnsembleResult):
            raise TypeError(
                "ConfidenceEntropyAnalyzer requires an "
                "EnsembleResult instance."
            )

        probabilities = dict(result.probabilities)

        self._validate_probabilities(probabilities)

        confidence = max(probabilities.values())

        sorted_probabilities = sorted(
            probabilities.values(),
            reverse=True,
        )

        margin = self._calculate_margin(sorted_probabilities)

        entropy = self._calculate_normalized_entropy(
            probabilities
        )

        uncertainty_level = self._classify_uncertainty(
            entropy
        )

        return UncertaintyResult(
            prediction=result.prediction,
            uncertainty_score=entropy,
            confidence_score=confidence,
            entropy=entropy,
            uncertainty_level=uncertainty_level,
            probabilities=probabilities,
            components={
                "entropy": entropy,
                "confidence_margin": margin,
            },
            metadata={
                "analyzer": "confidence_entropy",
                "confidence_margin": margin,
                "class_count": len(probabilities),
            },
        )

    def _validate_thresholds(self) -> None:
        """
        Validate entropy classification thresholds.
        """

        if not isinstance(
            self.low_threshold,
            (int, float),
        ):
            raise TypeError(
                "low_threshold must be numeric."
            )

        if not isinstance(
            self.high_threshold,
            (int, float),
        ):
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
        Validate the probability distribution.
        """

        if not probabilities:
            raise ValueError(
                "Cannot analyze empty probabilities."
            )

        total = 0.0

        for label, probability in probabilities.items():
            if not isinstance(
                probability,
                (int, float),
            ):
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

    @staticmethod
    def _calculate_margin(
        sorted_probabilities: List[float],
    ) -> float:
        """
        Calculate the difference between the top two probabilities.
        """

        if len(sorted_probabilities) == 1:
            return sorted_probabilities[0]

        return (
            sorted_probabilities[0]
            - sorted_probabilities[1]
        )

    @staticmethod
    def _calculate_normalized_entropy(
        probabilities: Dict[Any, float],
    ) -> float:
        """
        Calculate normalized Shannon entropy.

        Zero-probability terms are ignored because:

            lim p→0 p * log(p) = 0
        """

        class_count = len(probabilities)

        if class_count <= 1:
            return 0.0

        entropy = -sum(
            probability * math.log(probability)
            for probability in probabilities.values()
            if probability > 0.0
        )

        max_entropy = math.log(class_count)

        if max_entropy == 0.0:
            return 0.0

        return entropy / max_entropy

    def _classify_uncertainty(
        self,
        entropy: float,
    ) -> UncertaintyLevel:
        """
        Convert normalized entropy into an uncertainty level.
        """

        if entropy <= self.low_threshold:
            return UncertaintyLevel.LOW

        if entropy <= self.high_threshold:
            return UncertaintyLevel.MEDIUM

        return UncertaintyLevel.HIGH