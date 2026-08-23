from __future__ import annotations

from typing import Iterable

from src.monitoring.alert_manager import AlertManager
from src.monitoring.alerts import Alert
from src.monitoring.base import MonitoringResult


class AlertIntegration:
    """
    Integrates monitoring results with AlertManager and produces
    a combined monitoring and alert report.
    """

    def __init__(self, alert_manager: AlertManager) -> None:
        if not isinstance(alert_manager, AlertManager):
            raise TypeError(
                "alert_manager must be an instance of AlertManager"
            )

        self.alert_manager = alert_manager

    def execute(
        self,
        results: Iterable[MonitoringResult],
    ) -> dict:
        """
        Evaluate monitoring results and return a combined report.

        Returns:
            {
                "results": tuple[MonitoringResult, ...],
                "alerts": tuple[Alert, ...],
                "result_count": int,
                "alert_count": int,
            }
        """

        if isinstance(results, (str, bytes)):
            raise TypeError(
                "results must be an iterable of MonitoringResult instances"
            )

        try:
            validated_results = tuple(results)
        except TypeError as exc:
            raise TypeError(
                "results must be an iterable of MonitoringResult instances"
            ) from exc

        for result in validated_results:
            if not isinstance(result, MonitoringResult):
                raise TypeError(
                    "results must contain only MonitoringResult instances"
                )

        generated_alerts = tuple(
            self.alert_manager.evaluate(validated_results)
        )

        return {
            "results": validated_results,
            "alerts": generated_alerts,
            "result_count": len(validated_results),
            "alert_count": len(generated_alerts),
        }

    def execute_one(
        self,
        result: MonitoringResult,
    ) -> dict:
        """
        Evaluate one monitoring result and return a combined report.
        """

        if not isinstance(result, MonitoringResult):
            raise TypeError(
                "result must be an instance of MonitoringResult"
            )

        return self.execute((result,))

    def evaluate(
        self,
        results: Iterable[MonitoringResult],
    ) -> dict:
        """
        Alias for execute().
        """

        return self.execute(results)

    def evaluate_one(
        self,
        result: MonitoringResult,
    ) -> dict:
        """
        Alias for execute_one().
        """

        return self.execute_one(result)