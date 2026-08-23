from __future__ import annotations

from typing import Iterable

from src.monitoring.alerts import Alert, AlertRule
from src.monitoring.models import MonitoringResult


class AlertManager:
    """
    Evaluates monitoring results against configured alert rules
    and produces matching alerts.
    """

    def __init__(
        self,
        rules: Iterable[AlertRule],
    ) -> None:
        if not isinstance(rules, Iterable):
            raise TypeError(
                "rules must be an iterable of AlertRule instances."
            )

        self._rules = tuple(rules)

        for rule in self._rules:
            if not isinstance(rule, AlertRule):
                raise TypeError(
                    "rules must contain only AlertRule instances."
                )

    @property
    def rules(self) -> tuple[AlertRule, ...]:
        """Return configured alert rules."""

        return self._rules

    def evaluate(
        self,
        results: Iterable[MonitoringResult],
    ) -> list[Alert]:
        """
        Evaluate monitoring results against applicable rules.

        A rule is evaluated only against a result whose
        monitor_name matches the rule source.
        """

        if not isinstance(results, Iterable):
            raise TypeError(
                "results must be an iterable of MonitoringResult instances."
            )

        validated_results = tuple(results)

        for result in validated_results:
            if not isinstance(result, MonitoringResult):
                raise TypeError(
                    "results must contain only MonitoringResult instances."
                )

        alerts: list[Alert] = []

        for result in validated_results:
            for rule in self._rules:
                if rule.source != result.monitor_name:
                    continue

                alert = rule.create_alert(result)

                if alert is not None:
                    alerts.append(alert)

        return alerts

    def evaluate_one(
        self,
        result: MonitoringResult,
    ) -> list[Alert]:
        """Evaluate a single monitoring result."""

        if not isinstance(result, MonitoringResult):
            raise TypeError(
                "result must be a MonitoringResult instance."
            )

        return self.evaluate((result,))