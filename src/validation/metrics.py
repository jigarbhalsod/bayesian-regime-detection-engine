from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class ClassMetrics:
    """
    Performance metrics for a single class/regime.
    """

    precision: float
    recall: float
    f1_score: float
    support: int


@dataclass(frozen=True)
class PredictionMetricsResult:
    """
    Standardized prediction performance metrics.
    """

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    balanced_accuracy: float
    per_class: dict[Any, ClassMetrics]


class PredictionMetrics:
    """
    Calculate classification performance metrics for hard predictions.

    Provides macro-averaged precision, recall, and F1 score together
    with overall accuracy, balanced accuracy, and per-class metrics.
    """

    def evaluate(
        self,
        actual: Sequence[Any],
        predicted: Sequence[Any],
    ) -> PredictionMetricsResult:
        """
        Evaluate predicted classes against actual classes.
        """
        self._validate_inputs(actual, predicted)

        labels = self._get_labels(actual, predicted)
        total = len(actual)

        correct = sum(
            actual_value == predicted_value
            for actual_value, predicted_value in zip(actual, predicted)
        )

        accuracy = correct / total

        per_class: dict[Any, ClassMetrics] = {}

        for label in labels:
            true_positive = sum(
                actual_value == label and predicted_value == label
                for actual_value, predicted_value in zip(actual, predicted)
            )

            false_positive = sum(
                actual_value != label and predicted_value == label
                for actual_value, predicted_value in zip(actual, predicted)
            )

            false_negative = sum(
                actual_value == label and predicted_value != label
                for actual_value, predicted_value in zip(actual, predicted)
            )

            support = sum(
                actual_value == label
                for actual_value in actual
            )

            precision = self._safe_divide(
                true_positive,
                true_positive + false_positive,
            )

            recall = self._safe_divide(
                true_positive,
                true_positive + false_negative,
            )

            f1_score = self._calculate_f1(
                precision,
                recall,
            )

            per_class[label] = ClassMetrics(
                precision=precision,
                recall=recall,
                f1_score=f1_score,
                support=support,
            )

        precision = self._mean(
            metric.precision
            for metric in per_class.values()
        )

        recall = self._mean(
            metric.recall
            for metric in per_class.values()
        )

        f1_score = self._mean(
            metric.f1_score
            for metric in per_class.values()
        )

        balanced_accuracy = recall

        return PredictionMetricsResult(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            balanced_accuracy=balanced_accuracy,
            per_class=per_class,
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
    def _safe_divide(
        numerator: int,
        denominator: int,
    ) -> float:
        if denominator == 0:
            return 0.0

        return numerator / denominator

    @staticmethod
    def _calculate_f1(
        precision: float,
        recall: float,
    ) -> float:
        if precision + recall == 0:
            return 0.0

        return (
            2 * precision * recall
        ) / (
            precision + recall
        )

    @staticmethod
    def _mean(values: Any) -> float:
        values = list(values)

        if not values:
            return 0.0

        return sum(values) / len(values)