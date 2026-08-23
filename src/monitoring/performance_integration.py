from __future__ import annotations

from typing import Any

from src.monitoring.base import BaseMonitor
from src.monitoring.models import (
    MonitoringResult,
    MonitoringStatus,
)
from src.monitoring.performance_degradation import (
    PerformanceDegradationMonitor,
)
from src.monitoring.performance import ModelPerformanceMonitor


class PerformanceMonitoringIntegration:
    """
    Integrates model performance monitoring with optional
    performance degradation detection.
    """

    def __init__(
        self,
        performance_monitor: ModelPerformanceMonitor,
        degradation_monitor: PerformanceDegradationMonitor | None = None,
    ) -> None:
        if not isinstance(
            performance_monitor,
            ModelPerformanceMonitor,
        ):
            raise TypeError(
                "performance_monitor must be a "
                "ModelPerformanceMonitor instance."
            )

        if (
            degradation_monitor is not None
            and not isinstance(
                degradation_monitor,
                PerformanceDegradationMonitor,
            )
        ):
            raise TypeError(
                "degradation_monitor must be a "
                "PerformanceDegradationMonitor instance."
            )

        self.performance_monitor = performance_monitor
        self.degradation_monitor = degradation_monitor

    def execute(
        self,
        data: Any,
        reference_metrics: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        performance_result = self.performance_monitor.execute(
            data
        )

        result: dict[str, Any] = {
            "performance": performance_result,
            "degradation": None,
            "status": performance_result.status,
        }

        if reference_metrics is None:
            return result

        if self.degradation_monitor is None:
            raise ValueError(
                "reference_metrics requires a "
                "degradation_monitor."
            )

        current_metrics = self._extract_metrics(
            performance_result
        )

        degradation_result = self.degradation_monitor.execute(
            {
                "reference": reference_metrics,
                "current": current_metrics,
            }
        )

        result["degradation"] = degradation_result
        result["status"] = self._combine_statuses(
            performance_result.status,
            degradation_result.status,
        )

        return result

    @staticmethod
    def _extract_metrics(
        result: MonitoringResult,
    ) -> dict[str, float]:
        return {
            metric.name: float(metric.value)
            for metric in result.metrics
        }

    @staticmethod
    def _combine_statuses(
        performance_status: MonitoringStatus,
        degradation_status: MonitoringStatus,
    ) -> MonitoringStatus:
        if (
            performance_status
            == MonitoringStatus.WARNING
            or degradation_status
            == MonitoringStatus.WARNING
        ):
            return MonitoringStatus.WARNING

        return MonitoringStatus.OK