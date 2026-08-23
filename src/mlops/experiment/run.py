from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4


class ExperimentStatus(str, Enum):
    """
    Lifecycle states of an experiment run.
    """

    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ExperimentRun:
    """
    Represents a single execution of an experiment.
    """

    experiment_id: str
    run_id: str = field(default_factory=lambda: str(uuid4()))
    status: ExperimentStatus = ExperimentStatus.CREATED
    started_at: datetime | None = None
    completed_at: datetime | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def __post_init__(self) -> None:
        self.experiment_id = self.experiment_id.strip()

        if not self.experiment_id:
            raise ValueError("experiment_id must not be empty.")

        self.run_id = self.run_id.strip()

        if not self.run_id:
            raise ValueError("run_id must not be empty.")

        if not isinstance(self.status, ExperimentStatus):
            raise TypeError("status must be an ExperimentStatus.")

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(timezone.utc)

    def start(self) -> None:
        """
        Start the experiment run.
        """
        if self.status is not ExperimentStatus.CREATED:
            raise RuntimeError(
                "Only a created experiment run can be started."
            )

        self.status = ExperimentStatus.RUNNING
        self.started_at = self._utc_now()

    def log_metric(self, name: str, value: float) -> None:
        """
        Record a numeric metric for the active run.
        """
        if self.status is not ExperimentStatus.RUNNING:
            raise RuntimeError(
                "Metrics can only be logged while the run is running."
            )

        name = name.strip()

        if not name:
            raise ValueError("metric name must not be empty.")

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("metric value must be numeric.")

        self.metrics[name] = float(value)

    def log_artifact(self, name: str, artifact: Any) -> None:
        """
        Record an artifact for the active run.
        """
        if self.status is not ExperimentStatus.RUNNING:
            raise RuntimeError(
                "Artifacts can only be logged while the run is running."
            )

        name = name.strip()

        if not name:
            raise ValueError("artifact name must not be empty.")

        self.artifacts[name] = artifact

    def complete(self) -> None:
        """
        Mark the experiment run as successfully completed.
        """
        if self.status is not ExperimentStatus.RUNNING:
            raise RuntimeError(
                "Only a running experiment can be completed."
            )

        self.status = ExperimentStatus.COMPLETED
        self.completed_at = self._utc_now()

    def fail(self, error: str) -> None:
        """
        Mark the experiment run as failed.
        """
        if self.status is not ExperimentStatus.RUNNING:
            raise RuntimeError(
                "Only a running experiment can be failed."
            )

        error = error.strip()

        if not error:
            raise ValueError("error must not be empty.")

        self.status = ExperimentStatus.FAILED
        self.error = error
        self.completed_at = self._utc_now()