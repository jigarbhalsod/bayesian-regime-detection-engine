from __future__ import annotations

from typing import Any
from uuid import uuid4

from .base import BaseExperimentTracker
from .config import ExperimentConfig
from .result import ExperimentResult
from .run import ExperimentRun


class InMemoryExperimentTracker(BaseExperimentTracker):
    """
    In-memory implementation of the experiment tracking interface.

    Intended for local development, testing, and lightweight experimentation.
    """

    def __init__(self) -> None:
        self._experiments: dict[str, ExperimentConfig] = {}
        self._runs: dict[str, ExperimentRun] = {}

    def create_experiment(
        self,
        config: ExperimentConfig,
    ) -> str:
        """
        Create an experiment and return its unique identifier.
        """
        if not isinstance(config, ExperimentConfig):
            raise TypeError("config must be an ExperimentConfig.")

        experiment_id = str(uuid4())
        self._experiments[experiment_id] = config

        return experiment_id

    def start_run(
        self,
        experiment_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> ExperimentRun:
        """
        Create and start a new run for an existing experiment.
        """
        experiment_id = experiment_id.strip()

        if not experiment_id:
            raise ValueError("experiment_id must not be empty.")

        if experiment_id not in self._experiments:
            raise KeyError(
                f"Experiment '{experiment_id}' does not exist."
            )

        if metadata is not None and not isinstance(metadata, dict):
            raise TypeError("metadata must be a dictionary or None.")

        run = ExperimentRun(
            experiment_id=experiment_id,
            metadata=dict(metadata or {}),
        )

        run.start()
        self._runs[run.run_id] = run

        return run

    def get_run(
        self,
        run_id: str,
    ) -> ExperimentRun:
        """
        Retrieve an experiment run by its identifier.
        """
        run_id = run_id.strip()

        if not run_id:
            raise ValueError("run_id must not be empty.")

        try:
            return self._runs[run_id]
        except KeyError as exc:
            raise KeyError(
                f"Run '{run_id}' does not exist."
            ) from exc

    def complete_run(
        self,
        run_id: str,
    ) -> ExperimentResult:
        """
        Complete a run and return its structured result.
        """
        run = self.get_run(run_id)
        run.complete()

        return ExperimentResult(
            experiment_id=run.experiment_id,
            run_id=run.run_id,
            success=True,
            metrics=run.metrics,
            artifacts=run.artifacts,
            metadata=run.metadata,
        )

    def fail_run(
        self,
        run_id: str,
        error: str,
    ) -> ExperimentResult:
        """
        Fail a run and return its structured result.
        """
        run = self.get_run(run_id)
        run.fail(error)

        return ExperimentResult(
            experiment_id=run.experiment_id,
            run_id=run.run_id,
            success=False,
            metrics=run.metrics,
            artifacts=run.artifacts,
            metadata=run.metadata,
            error=run.error,
        )