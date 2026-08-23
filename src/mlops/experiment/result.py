from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping


@dataclass(frozen=True)
class ExperimentResult:
    """
    Immutable structured result produced by an experiment run.
    """

    experiment_id: str
    run_id: str
    success: bool
    metrics: Mapping[str, float] = field(default_factory=dict)
    artifacts: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    error: str | None = None

    def __post_init__(self) -> None:
        experiment_id = self.experiment_id.strip()
        run_id = self.run_id.strip()

        if not experiment_id:
            raise ValueError("experiment_id must not be empty.")

        if not run_id:
            raise ValueError("run_id must not be empty.")

        if not isinstance(self.success, bool):
            raise TypeError("success must be a boolean.")

        if self.success and self.error is not None:
            raise ValueError(
                "error must be None when success is True."
            )

        if not self.success and self.error is not None:
            error = self.error.strip()

            if not error:
                raise ValueError(
                    "error must not be empty when provided."
                )

            object.__setattr__(self, "error", error)

        for name, value in self.metrics.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError(
                    "metric names must be non-empty strings."
                )

            if not isinstance(value, (int, float)):
                raise TypeError(
                    "metric values must be numeric."
                )

        object.__setattr__(self, "experiment_id", experiment_id)
        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(
            self,
            "metrics",
            MappingProxyType(dict(self.metrics)),
        )
        object.__setattr__(
            self,
            "artifacts",
            MappingProxyType(dict(self.artifacts)),
        )
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata)),
        )