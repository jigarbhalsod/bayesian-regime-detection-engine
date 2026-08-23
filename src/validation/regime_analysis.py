from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class RegimeStatistics:
    """
    Classification statistics for one actual regime.
    """

    support: int
    correct_predictions: int
    accuracy: float


@dataclass(frozen=True)
class Misclassification:
    """
    Summary of one actual -> predicted regime confusion.
    """

    actual_regime: Any
    predicted_regime: Any
    count: int


@dataclass(frozen=True)
class ClassificationAnalysisResult:
    """
    Complete regime classification analysis.
    """

    labels: list[Any]
    confusion_matrix: dict[Any, dict[Any, int]]
    per_regime: dict[Any, RegimeStatistics]
    total_observations: int
    correct_predictions: int
    incorrect_predictions: int
    most_common_misclassification: Misclassification | None


class RegimeClassificationAnalyzer:
    """
    Analyze regime classification behavior.

    Provides:
    - Confusion matrix
    - Per-regime support
    - Per-regime accuracy
    - Correct and incorrect prediction counts
    - Most common regime misclassification
    """

    def analyze(
        self,
        actual: Sequence[Any],
        predicted: Sequence[Any],
    ) -> ClassificationAnalysisResult:
        """
        Analyze actual versus predicted regimes.
        """
        self._validate_inputs(actual, predicted)

        labels = self._get_labels(actual, predicted)

        confusion_matrix = self._build_confusion_matrix(
            actual,
            predicted,
            labels,
        )

        per_regime = self._calculate_per_regime_statistics(
            confusion_matrix,
            labels,
        )

        total_observations = len(actual)

        correct_predictions = sum(
            actual_value == predicted_value
            for actual_value, predicted_value in zip(
                actual,
                predicted,
            )
        )

        incorrect_predictions = (
            total_observations - correct_predictions
        )

        most_common_misclassification = (
            self._find_most_common_misclassification(
                confusion_matrix,
                labels,
            )
        )

        return ClassificationAnalysisResult(
            labels=labels,
            confusion_matrix=confusion_matrix,
            per_regime=per_regime,
            total_observations=total_observations,
            correct_predictions=correct_predictions,
            incorrect_predictions=incorrect_predictions,
            most_common_misclassification=(
                most_common_misclassification
            ),
        )

    @staticmethod
    def _validate_inputs(
        actual: Sequence[Any],
        predicted: Sequence[Any],
    ) -> None:
        if isinstance(actual, (str, bytes)):
            raise TypeError(
                "actual must be a sequence, not a string or bytes"
            )

        if isinstance(predicted, (str, bytes)):
            raise TypeError(
                "predicted must be a sequence, not a string or bytes"
            )

        if not isinstance(actual, Sequence):
            raise TypeError("actual must be a sequence")

        if not isinstance(predicted, Sequence):
            raise TypeError("predicted must be a sequence")

        if len(actual) == 0:
            raise ValueError("actual cannot be empty")

        if len(predicted) == 0:
            raise ValueError("predicted cannot be empty")

        if len(actual) != len(predicted):
            raise ValueError(
                "actual and predicted must have the same length"
            )

    @staticmethod
    def _get_labels(
        actual: Sequence[Any],
        predicted: Sequence[Any],
    ) -> list[Any]:
        labels: list[Any] = []

        for value in actual:
            if value not in labels:
                labels.append(value)

        for value in predicted:
            if value not in labels:
                labels.append(value)

        return labels

    @staticmethod
    def _build_confusion_matrix(
        actual: Sequence[Any],
        predicted: Sequence[Any],
        labels: Sequence[Any],
    ) -> dict[Any, dict[Any, int]]:
        matrix = {
            actual_label: {
                predicted_label: 0
                for predicted_label in labels
            }
            for actual_label in labels
        }

        for actual_value, predicted_value in zip(
            actual,
            predicted,
        ):
            matrix[actual_value][predicted_value] += 1

        return matrix

    @staticmethod
    def _calculate_per_regime_statistics(
        confusion_matrix: dict[Any, dict[Any, int]],
        labels: Sequence[Any],
    ) -> dict[Any, RegimeStatistics]:
        statistics: dict[Any, RegimeStatistics] = {}

        for label in labels:
            support = sum(
                confusion_matrix[label].values()
            )

            correct_predictions = confusion_matrix[label][label]

            accuracy = (
                correct_predictions / support
                if support > 0
                else 0.0
            )

            statistics[label] = RegimeStatistics(
                support=support,
                correct_predictions=correct_predictions,
                accuracy=accuracy,
            )

        return statistics

    @staticmethod
    def _find_most_common_misclassification(
        confusion_matrix: dict[Any, dict[Any, int]],
        labels: Sequence[Any],
    ) -> Misclassification | None:
        most_common: Misclassification | None = None

        for actual_label in labels:
            for predicted_label in labels:
                if actual_label == predicted_label:
                    continue

                count = (
                    confusion_matrix[actual_label][predicted_label]
                )

                if count == 0:
                    continue

                if (
                    most_common is None
                    or count > most_common.count
                ):
                    most_common = Misclassification(
                        actual_regime=actual_label,
                        predicted_regime=predicted_label,
                        count=count,
                    )

        return most_common