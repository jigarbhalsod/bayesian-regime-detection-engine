from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .config import ExperimentConfig
from .result import ExperimentResult
from .run import ExperimentRun


class BaseExperimentTracker(ABC):
    """
    Abstract interface for experiment tracking implementations.
    """

    @abstractmethod
    def create_experiment(
        self,
        config: ExperimentConfig,
    ) -> str:
        """
        Create an experiment and return its unique identifier.
        """
        raise NotImplementedError

    @abstractmethod
    def start_run(
        self,
        experiment_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> ExperimentRun:
        """
        Create and start a new run for an experiment.
        """
        raise NotImplementedError

    @abstractmethod
    def get_run(
        self,
        run_id: str,
    ) -> ExperimentRun:
        """
        Retrieve an experiment run by its identifier.
        """
        raise NotImplementedError

    @abstractmethod
    def complete_run(
        self,
        run_id: str,
    ) -> ExperimentResult:
        """
        Complete a run and return its structured result.
        """
        raise NotImplementedError

    @abstractmethod
    def fail_run(
        self,
        run_id: str,
        error: str,
    ) -> ExperimentResult:
        """
        Fail a run and return its structured result.
        """
        raise NotImplementedError