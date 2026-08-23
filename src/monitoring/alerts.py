from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.monitoring.models import MonitoringResult, MonitoringStatus


class AlertSeverity(str, Enum):
    """Severity levels used by monitoring alerts."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass(frozen=True)
class Alert:
    """
    Immutable alert generated from a monitoring result.
    """

    source: str
    severity: AlertSeverity
    message: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError(
                "source must be a non-empty string."
            )

        if not isinstance(self.severity, AlertSeverity):
            raise TypeError(
                "severity must be an AlertSeverity instance."
            )

        if not isinstance(self.message, str) or not self.message.strip():
            raise ValueError(
                "message must be a non-empty string."
            )

        if not isinstance(self.metadata, dict):
            raise TypeError(
                "metadata must be a dictionary."
            )

        normalized_metadata: dict[str, Any] = {}

        for key, value in self.metadata.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError(
                    "metadata keys must be non-empty strings."
                )

            normalized_metadata[key.strip()] = value

        object.__setattr__(
            self,
            "source",
            self.source.strip(),
        )
        object.__setattr__(
            self,
            "message",
            self.message.strip(),
        )
        object.__setattr__(
            self,
            "metadata",
            normalized_metadata,
        )


@dataclass(frozen=True)
class AlertRule:
    """
    Rule for converting monitoring statuses into alerts.
    """

    source: str
    severity: AlertSeverity
    statuses: tuple[MonitoringStatus, ...] = (
        MonitoringStatus.WARNING,
    )
    message_template: str = (
        "{source} reported status {status}."
    )

    def __post_init__(self) -> None:
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError(
                "source must be a non-empty string."
            )

        if not isinstance(self.severity, AlertSeverity):
            raise TypeError(
                "severity must be an AlertSeverity instance."
            )

        if not isinstance(self.statuses, tuple):
            raise TypeError(
                "statuses must be a tuple."
            )

        if not self.statuses:
            raise ValueError(
                "statuses must not be empty."
            )

        for status in self.statuses:
            if not isinstance(status, MonitoringStatus):
                raise TypeError(
                    "statuses must contain MonitoringStatus values."
                )

        if (
            not isinstance(self.message_template, str)
            or not self.message_template.strip()
        ):
            raise ValueError(
                "message_template must be a non-empty string."
            )

        object.__setattr__(
            self,
            "source",
            self.source.strip(),
        )
        object.__setattr__(
            self,
            "message_template",
            self.message_template.strip(),
        )

    def matches(
        self,
        result: MonitoringResult,
    ) -> bool:
        """
        Return True when this rule applies to the result.
        """

        if not isinstance(result, MonitoringResult):
            raise TypeError(
                "result must be a MonitoringResult instance."
            )

        return result.status in self.statuses

    def create_alert(
        self,
        result: MonitoringResult,
    ) -> Alert | None:
        """
        Create an alert when the rule matches the result.
        """

        if not isinstance(result, MonitoringResult):
            raise TypeError(
                "result must be a MonitoringResult instance."
            )

        if not self.matches(result):
            return None

        message = self.message_template.format(
            source=self.source,
            status=result.status.value,
        )

        return Alert(
            source=self.source,
            severity=self.severity,
            message=message,
            metadata={
                "monitor_status": result.status.value,
            },
        )