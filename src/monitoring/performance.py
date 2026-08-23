from __future__ import annotations

from typing import Any

import numpy as np

from src.monitoring.base import BaseMonitor
from src.monitoring.models import (
    MetricResult,
    MonitoringResult,
    MonitoringStatus,
)


class ModelPerformanceMonitor(BaseMonitor):
    """
    Base implementation for monitoring model performance metrics.

    Subclasses provide concrete metric calculation through
    calculate_metrics().
    """

    def monitor(
        self,
        data: Any,
    ) -> MonitoringResult:
        y_true, y_pred = self._validate_data(data)

        metrics = self.calculate_metrics(
            y_true,
            y_pred,
        )

        if not isinstance(metrics, tuple):
            raise TypeError(
                "calculate_metrics must return a tuple."
            )

        for metric in metrics:
            if not isinstance(metric, MetricResult):
                raise TypeError(
                    "calculate_metrics must return "
                    "MetricResult instances."
                )

        status = self._determine_status(metrics)

        return MonitoringResult(
            monitor_name=self.name,
            status=status,
            metrics=metrics,
            metadata={
                "sample_count": int(len(y_true)),
            },
        )

    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> tuple[MetricResult, ...]:
        raise NotImplementedError(
            "Subclasses must implement calculate_metrics()."
        )

    @staticmethod
    def _validate_data(
        data: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(data, dict):
            raise TypeError(
                "data must be a dictionary containing "
                "'y_true' and 'y_pred'."
            )

        required_keys = {
            "y_true",
            "y_pred",
        }

        if set(data.keys()) != required_keys:
            raise ValueError(
                "data must contain exactly 'y_true' and "
                "'y_pred'."
            )

        y_true = ModelPerformanceMonitor._validate_array(
            data["y_true"],
            "y_true",
        )

        y_pred = ModelPerformanceMonitor._validate_array(
            data["y_pred"],
            "y_pred",
        )

        if len(y_true) != len(y_pred):
            raise ValueError(
                "y_true and y_pred must have the same length."
            )

        return y_true, y_pred

    @staticmethod
    def _validate_array(
        values: Any,
        name: str,
    ) -> np.ndarray:
        if isinstance(
            values,
            (str, bytes, dict),
        ):
            raise TypeError(
                f"{name} must be a one-dimensional sequence."
            )

        try:
            array = np.asarray(values)
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise TypeError(
                f"{name} must be a one-dimensional sequence."
            ) from exc

        if array.ndim != 1:
            raise ValueError(
                f"{name} must be one-dimensional."
            )

        if len(array) == 0:
            raise ValueError(
                f"{name} must not be empty."
            )

        return array

    def _determine_status(
        self,
        metrics: tuple[MetricResult, ...],
    ) -> MonitoringStatus:
        for metric in metrics:
            if metric.threshold is None:
                continue

            if self._is_threshold_violated(metric):
                return MonitoringStatus.WARNING

        return MonitoringStatus.OK

    @staticmethod
    def _is_threshold_violated(
        metric: MetricResult,
    ) -> bool:
        """
        Determine whether a metric violates its configured threshold.

        Error and loss metrics are lower-is-better, while metrics such
        as accuracy, precision, recall, F1, and R-squared are
        higher-is-better.
        """
        threshold = float(metric.threshold)
        value = float(metric.value)

        lower_is_better_metrics = {
            "loss",
            "mae",
            "mse",
            "rmse",
        }

        lower_is_better = (
            metric.name in lower_is_better_metrics
            or metric.name.endswith("_loss")
            or metric.name.endswith("_error")
            or metric.name.endswith("_mae")
            or metric.name.endswith("_mse")
            or metric.name.endswith("_rmse")
        )

        if lower_is_better:
            return value > threshold

        return value < threshold