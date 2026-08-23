from __future__ import annotations

from typing import Any

from src.monitoring.base import BaseMonitor
from src.monitoring.models import (
    MetricResult,
    MonitoringResult,
    MonitoringStatus,
)


class PerformanceDegradationMonitor(BaseMonitor):
    """
    Compare current model performance against a reference baseline.

    Positive degradation means performance became worse.

    For higher-is-better metrics:
        degradation = reference - current

    For lower-is-better metrics:
        degradation = current - reference
    """

    _LOWER_IS_BETTER = {
        "loss",
        "mae",
        "mse",
        "rmse",
    }

    def monitor(
        self,
        data: Any,
    ) -> MonitoringResult:
        reference, current = self._validate_data(data)

        metric_names = sorted(
            reference.keys()
        )

        metrics: list[MetricResult] = []

        for metric_name in metric_names:
            reference_value = reference[metric_name]
            current_value = current[metric_name]

            degradation = self._calculate_degradation(
                metric_name,
                reference_value,
                current_value,
            )

            threshold = self.config.get_threshold(
                metric_name
            )

            metrics.append(
                MetricResult(
                    name=f"{metric_name}_degradation",
                    value=degradation,
                    threshold=threshold,
                )
            )

        metrics_tuple = tuple(metrics)

        status = self._determine_status(
            metrics_tuple
        )

        return MonitoringResult(
            monitor_name=self.name,
            status=status,
            metrics=metrics_tuple,
            metadata={
                "metric_count": len(metrics_tuple),
                "reference": reference,
                "current": current,
            },
        )

    @classmethod
    def _calculate_degradation(
        cls,
        metric_name: str,
        reference_value: float,
        current_value: float,
    ) -> float:
        if cls._is_lower_better(metric_name):
            return float(
                current_value - reference_value
            )

        return float(
            reference_value - current_value
        )

    @classmethod
    def _is_lower_better(
        cls,
        metric_name: str,
    ) -> bool:
        return (
            metric_name in cls._LOWER_IS_BETTER
            or metric_name.endswith("_loss")
            or metric_name.endswith("_error")
            or metric_name.endswith("_mae")
            or metric_name.endswith("_mse")
            or metric_name.endswith("_rmse")
        )

    @staticmethod
    def _validate_data(
        data: Any,
    ) -> tuple[dict[str, float], dict[str, float]]:
        if not isinstance(data, dict):
            raise TypeError(
                "data must be a dictionary."
            )

        required_keys = {
            "reference",
            "current",
        }

        if set(data.keys()) != required_keys:
            raise ValueError(
                "data must contain exactly "
                "'reference' and 'current'."
            )

        reference = PerformanceDegradationMonitor._validate_metrics(
            data["reference"],
            "reference",
        )

        current = PerformanceDegradationMonitor._validate_metrics(
            data["current"],
            "current",
        )

        if set(reference.keys()) != set(current.keys()):
            raise ValueError(
                "reference and current must contain "
                "the same metric names."
            )

        return reference, current

    @staticmethod
    def _validate_metrics(
        metrics: Any,
        name: str,
    ) -> dict[str, float]:
        if not isinstance(metrics, dict):
            raise TypeError(
                f"{name} must be a dictionary."
            )

        if not metrics:
            raise ValueError(
                f"{name} must not be empty."
            )

        validated: dict[str, float] = {}

        for metric_name, metric_value in metrics.items():
            if (
                not isinstance(metric_name, str)
                or not metric_name.strip()
            ):
                raise ValueError(
                    f"{name} contains an invalid metric name."
                )

            if isinstance(metric_value, bool) or not isinstance(
                metric_value,
                (int, float),
            ):
                raise TypeError(
                    f"{name}.{metric_name} must be numeric."
                )

            validated[
                metric_name.strip()
            ] = float(metric_value)

        return validated

    def _determine_status(
        self,
        metrics: tuple[MetricResult, ...],
    ) -> MonitoringStatus:
        for metric in metrics:
            if metric.threshold is None:
                continue

            if float(metric.value) > float(metric.threshold):
                return MonitoringStatus.WARNING

        return MonitoringStatus.OK