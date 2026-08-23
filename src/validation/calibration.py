from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class CalibrationBin:
    """
    Calibration statistics for one confidence interval.
    """

    lower_bound: float
    upper_bound: float
    count: int
    mean_confidence: float
    accuracy: float


@dataclass(frozen=True)
class CalibrationValidationResult:
    """
    Standardized probability and calibration validation results.
    """

    brier_score: float
    log_loss: float
    expected_calibration_error: float
    maximum_calibration_error: float
    bins: list[CalibrationBin]


class ProbabilityCalibrationValidator:
    """
    Validate multiclass predicted probabilities and calibration.

    Metrics:
    - Multiclass Brier Score
    - Multiclass Log Loss
    - Expected Calibration Error (ECE)
    - Maximum Calibration Error (MCE)

    Calibration is computed using the model's maximum predicted
    probability as confidence and whether the corresponding predicted
    class is correct.
    """

    def __init__(
        self,
        num_bins: int = 10,
        tolerance: float = 1e-9,
    ) -> None:
        self.num_bins = self._validate_positive_integer(
            num_bins,
            "num_bins",
        )
        self.tolerance = self._validate_tolerance(tolerance)

    def evaluate(
        self,
        actual: Sequence[Any],
        probabilities: Sequence[dict[Any, float]],
    ) -> CalibrationValidationResult:
        """
        Evaluate predicted probability distributions.
        """
        self._validate_inputs(actual, probabilities)

        labels = self._get_labels(actual, probabilities)

        brier_score = self._calculate_brier_score(
            actual,
            probabilities,
            labels,
        )

        log_loss = self._calculate_log_loss(
            actual,
            probabilities,
        )

        bins = self._calculate_bins(
            actual,
            probabilities,
        )

        expected_calibration_error = (
            self._calculate_expected_calibration_error(
                bins,
                len(actual),
            )
        )

        maximum_calibration_error = (
            self._calculate_maximum_calibration_error(bins)
        )

        return CalibrationValidationResult(
            brier_score=brier_score,
            log_loss=log_loss,
            expected_calibration_error=expected_calibration_error,
            maximum_calibration_error=maximum_calibration_error,
            bins=bins,
        )

    @staticmethod
    def _validate_positive_integer(
        value: int,
        name: str,
    ) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")

        if value <= 0:
            raise ValueError(f"{name} must be greater than 0")

        return value

    @staticmethod
    def _validate_tolerance(value: float) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("tolerance must be numeric")

        value = float(value)

        if value < 0.0:
            raise ValueError("tolerance must be non-negative")

        return value

    def _validate_inputs(
        self,
        actual: Sequence[Any],
        probabilities: Sequence[dict[Any, float]],
    ) -> None:
        if isinstance(actual, (str, bytes)):
            raise TypeError(
                "actual must be a sequence, not a string or bytes"
            )

        if isinstance(probabilities, (str, bytes)):
            raise TypeError(
                "probabilities must be a sequence, not a string or bytes"
            )

        if not isinstance(actual, Sequence):
            raise TypeError("actual must be a sequence")

        if not isinstance(probabilities, Sequence):
            raise TypeError("probabilities must be a sequence")

        if len(actual) == 0:
            raise ValueError("actual cannot be empty")

        if len(probabilities) == 0:
            raise ValueError("probabilities cannot be empty")

        if len(actual) != len(probabilities):
            raise ValueError(
                "actual and probabilities must have the same length"
            )

        for actual_label, distribution in zip(
            actual,
            probabilities,
        ):
            self._validate_distribution(
                actual_label,
                distribution,
            )

    def _validate_distribution(
        self,
        actual_label: Any,
        distribution: dict[Any, float],
    ) -> None:
        if not isinstance(distribution, dict):
            raise TypeError(
                "each probability distribution must be a dictionary"
            )

        if not distribution:
            raise ValueError(
                "probability distribution cannot be empty"
            )

        if actual_label not in distribution:
            raise ValueError(
                "actual label must exist in its probability distribution"
            )

        total_probability = 0.0

        for probability in distribution.values():
            if isinstance(probability, bool) or not isinstance(
                probability,
                (int, float),
            ):
                raise TypeError(
                    "probability values must be numeric"
                )

            probability = float(probability)

            if not math.isfinite(probability):
                raise ValueError(
                    "probability values must be finite"
                )

            if probability < 0.0 or probability > 1.0:
                raise ValueError(
                    "probability values must be between 0 and 1"
                )

            total_probability += probability

        if abs(total_probability - 1.0) > self.tolerance:
            raise ValueError(
                "probabilities must sum to 1.0 within tolerance"
            )

    @staticmethod
    def _get_labels(
        actual: Sequence[Any],
        probabilities: Sequence[dict[Any, float]],
    ) -> list[Any]:
        labels: list[Any] = []

        for label in actual:
            if label not in labels:
                labels.append(label)

        for distribution in probabilities:
            for label in distribution:
                if label not in labels:
                    labels.append(label)

        return labels

    @staticmethod
    def _calculate_brier_score(
        actual: Sequence[Any],
        probabilities: Sequence[dict[Any, float]],
        labels: Sequence[Any],
    ) -> float:
        total_error = 0.0

        for actual_label, distribution in zip(
            actual,
            probabilities,
        ):
            for label in labels:
                probability = float(
                    distribution.get(label, 0.0)
                )
                outcome = 1.0 if label == actual_label else 0.0

                total_error += (
                    probability - outcome
                ) ** 2

        return total_error / len(actual)

    @staticmethod
    def _calculate_log_loss(
        actual: Sequence[Any],
        probabilities: Sequence[dict[Any, float]],
    ) -> float:
        total_loss = 0.0
        epsilon = 1e-15

        for actual_label, distribution in zip(
            actual,
            probabilities,
        ):
            probability = float(distribution[actual_label])
            probability = min(
                max(probability, epsilon),
                1.0 - epsilon,
            )

            total_loss -= math.log(probability)

        return total_loss / len(actual)

    def _calculate_bins(
        self,
        actual: Sequence[Any],
        probabilities: Sequence[dict[Any, float]],
    ) -> list[CalibrationBin]:
        bin_confidences: list[list[float]] = [
            [] for _ in range(self.num_bins)
        ]
        bin_correctness: list[list[float]] = [
            [] for _ in range(self.num_bins)
        ]

        for actual_label, distribution in zip(
            actual,
            probabilities,
        ):
            predicted_label, confidence = max(
                distribution.items(),
                key=lambda item: item[1],
            )

            confidence = float(confidence)

            bin_index = min(
                int(confidence * self.num_bins),
                self.num_bins - 1,
            )

            bin_confidences[bin_index].append(confidence)

            is_correct = (
                1.0
                if predicted_label == actual_label
                else 0.0
            )

            bin_correctness[bin_index].append(is_correct)

        bins: list[CalibrationBin] = []

        for index in range(self.num_bins):
            lower_bound = index / self.num_bins
            upper_bound = (index + 1) / self.num_bins

            confidences = bin_confidences[index]
            correctness = bin_correctness[index]

            count = len(confidences)

            if count == 0:
                mean_confidence = 0.0
                accuracy = 0.0
            else:
                mean_confidence = sum(confidences) / count
                accuracy = sum(correctness) / count

            bins.append(
                CalibrationBin(
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                    count=count,
                    mean_confidence=mean_confidence,
                    accuracy=accuracy,
                )
            )

        return bins

    @staticmethod
    def _calculate_expected_calibration_error(
        bins: Sequence[CalibrationBin],
        total_samples: int,
    ) -> float:
        if total_samples <= 0:
            return 0.0

        return sum(
            (bin_.count / total_samples)
            * abs(bin_.accuracy - bin_.mean_confidence)
            for bin_ in bins
        )

    @staticmethod
    def _calculate_maximum_calibration_error(
        bins: Sequence[CalibrationBin],
    ) -> float:
        populated_errors = [
            abs(bin_.accuracy - bin_.mean_confidence)
            for bin_ in bins
            if bin_.count > 0
        ]

        if not populated_errors:
            return 0.0

        return max(populated_errors)