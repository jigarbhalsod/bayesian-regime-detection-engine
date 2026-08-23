from __future__ import annotations

from typing import Any

import numpy as np

from src.monitoring.models import MetricResult
from src.monitoring.performance import ModelPerformanceMonitor


class ClassificationPerformanceMonitor(ModelPerformanceMonitor):
    """
    Monitor classification model performance.

    Calculates accuracy, precision, recall and F1 score using
    macro averaging across all observed classes.
    """

    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> tuple[MetricResult, ...]:
        labels = np.unique(
            np.concatenate(
                [
                    y_true,
                    y_pred,
                ]
            )
        )

        accuracy = float(
            np.mean(y_true == y_pred)
        )

        precisions: list[float] = []
        recalls: list[float] = []
        f1_scores: list[float] = []

        for label in labels:
            true_positive = int(
                np.sum(
                    (y_true == label)
                    & (y_pred == label)
                )
            )

            false_positive = int(
                np.sum(
                    (y_true != label)
                    & (y_pred == label)
                )
            )

            false_negative = int(
                np.sum(
                    (y_true == label)
                    & (y_pred != label)
                )
            )

            precision_denominator = (
                true_positive + false_positive
            )

            recall_denominator = (
                true_positive + false_negative
            )

            precision = (
                true_positive / precision_denominator
                if precision_denominator > 0
                else 0.0
            )

            recall = (
                true_positive / recall_denominator
                if recall_denominator > 0
                else 0.0
            )

            f1_denominator = precision + recall

            f1_score = (
                2 * precision * recall / f1_denominator
                if f1_denominator > 0
                else 0.0
            )

            precisions.append(float(precision))
            recalls.append(float(recall))
            f1_scores.append(float(f1_score))

        precision = float(np.mean(precisions))
        recall = float(np.mean(recalls))
        f1_score = float(np.mean(f1_scores))

        return (
            MetricResult(
                name="accuracy",
                value=accuracy,
                threshold=self.config.get_threshold(
                    "accuracy"
                ),
            ),
            MetricResult(
                name="precision",
                value=precision,
                threshold=self.config.get_threshold(
                    "precision"
                ),
            ),
            MetricResult(
                name="recall",
                value=recall,
                threshold=self.config.get_threshold(
                    "recall"
                ),
            ),
            MetricResult(
                name="f1_score",
                value=f1_score,
                threshold=self.config.get_threshold(
                    "f1_score"
                ),
            ),
        )

    def monitor(
        self,
        data: Any,
    ):
        return super().monitor(data)