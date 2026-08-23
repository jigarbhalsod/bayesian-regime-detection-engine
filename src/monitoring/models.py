from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class MonitoringStatus(str, Enum):
    """
    Overall health status produced by monitoring.
    """

    OK = "ok"
    WARNING = "warning"
    CRITICAL = "critical"
    ERROR = "error"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class MetricResult:
    """
    Represents one monitored metric.
    """

    name: str
    value: float
    threshold: float | None = None
    status: MonitoringStatus = MonitoringStatus.OK
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized_name = self.name.strip()

        if not normalized_name:
            raise ValueError("Metric name cannot be empty.")

        if isinstance(self.value, bool) or not isinstance(
            self.value,
            (int, float),
        ):
            raise TypeError("Metric value must be numeric.")

        if not isinstance(self.status, MonitoringStatus):
            raise TypeError(
                "status must be an instance of MonitoringStatus."
            )

        if self.threshold is not None:
            if isinstance(self.threshold, bool) or not isinstance(
                self.threshold,
                (int, float),
            ):
                raise TypeError(
                    "Metric threshold must be numeric."
                )

            object.__setattr__(
                self,
                "threshold",
                float(self.threshold),
            )

        normalized_metadata: dict[str, Any] = {
            str(key).strip(): value
            for key, value in self.metadata.items()
        }

        if any(not key for key in normalized_metadata):
            raise ValueError(
                "Metric metadata keys cannot be empty."
            )

        object.__setattr__(self, "name", normalized_name)
        object.__setattr__(self, "value", float(self.value))
        object.__setattr__(
            self,
            "metadata",
            normalized_metadata,
        )


@dataclass(frozen=True)
class MonitoringResult:
    """
    Standard result returned by a monitor execution.
    """

    monitor_name: str
    status: MonitoringStatus
    metrics: tuple[MetricResult, ...] = ()
    message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def __post_init__(self) -> None:
        normalized_name = self.monitor_name.strip()

        if not normalized_name:
            raise ValueError("Monitor name cannot be empty.")

        if not isinstance(self.status, MonitoringStatus):
            raise TypeError(
                "status must be an instance of MonitoringStatus."
            )

        metrics = tuple(self.metrics)

        if not all(
            isinstance(metric, MetricResult)
            for metric in metrics
        ):
            raise TypeError(
                "All metrics must be MetricResult instances."
            )

        if self.message is not None:
            normalized_message = self.message.strip() or None
            object.__setattr__(
                self,
                "message",
                normalized_message,
            )

        normalized_metadata: dict[str, Any] = {
            str(key).strip(): value
            for key, value in self.metadata.items()
        }

        if any(not key for key in normalized_metadata):
            raise ValueError(
                "Monitoring result metadata keys cannot be empty."
            )

        if not isinstance(self.timestamp, datetime):
            raise TypeError(
                "timestamp must be a datetime instance."
            )

        if self.timestamp.tzinfo is None:
            raise ValueError(
                "timestamp must be timezone-aware."
            )

        object.__setattr__(
            self,
            "monitor_name",
            normalized_name,
        )
        object.__setattr__(
            self,
            "metrics",
            metrics,
        )
        object.__setattr__(
            self,
            "metadata",
            normalized_metadata,
        )


@dataclass(frozen=True)
class MonitoringSnapshot:
    """
    Aggregated snapshot of monitoring results.
    """

    name: str
    results: tuple[MonitoringResult, ...]
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def __post_init__(self) -> None:
        normalized_name = self.name.strip()

        if not normalized_name:
            raise ValueError("Snapshot name cannot be empty.")

        results = tuple(self.results)

        if not results:
            raise ValueError(
                "MonitoringSnapshot must contain at least one result."
            )

        if not all(
            isinstance(result, MonitoringResult)
            for result in results
        ):
            raise TypeError(
                "All results must be MonitoringResult instances."
            )

        normalized_metadata: dict[str, Any] = {
            str(key).strip(): value
            for key, value in self.metadata.items()
        }

        if any(not key for key in normalized_metadata):
            raise ValueError(
                "Snapshot metadata keys cannot be empty."
            )

        if not isinstance(self.timestamp, datetime):
            raise TypeError(
                "timestamp must be a datetime instance."
            )

        if self.timestamp.tzinfo is None:
            raise ValueError(
                "timestamp must be timezone-aware."
            )

        object.__setattr__(self, "name", normalized_name)
        object.__setattr__(self, "results", results)
        object.__setattr__(
            self,
            "metadata",
            normalized_metadata,
        )

    @property
    def status(self) -> MonitoringStatus:
        """
        Return the highest-severity status across all results.
        """

        priorities = {
            MonitoringStatus.OK: 0,
            MonitoringStatus.WARNING: 1,
            MonitoringStatus.UNKNOWN: 2,
            MonitoringStatus.CRITICAL: 3,
            MonitoringStatus.ERROR: 4,
        }

        return max(
            (
                result.status
                for result in self.results
            ),
            key=lambda status: priorities[status],
        )