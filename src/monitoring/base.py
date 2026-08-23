from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.monitoring.config import MonitoringConfig
from src.monitoring.models import MonitoringResult


class BaseMonitor(ABC):
    """
    Abstract base class for all monitoring components.
    """

    def __init__(
        self,
        config: MonitoringConfig,
    ) -> None:
        if not isinstance(config, MonitoringConfig):
            raise TypeError(
                "config must be a MonitoringConfig instance."
            )

        self._config = config

    @property
    def config(self) -> MonitoringConfig:
        """
        Return the monitor configuration.
        """
        return self._config

    @property
    def name(self) -> str:
        """
        Return the monitor name.
        """
        return self._config.name

    @property
    def enabled(self) -> bool:
        """
        Return whether monitoring is enabled.
        """
        return self._config.enabled

    def execute(
        self,
        data: Any,
    ) -> MonitoringResult:
        """
        Execute the monitor.

        Disabled monitors cannot be executed.
        """

        if not self.enabled:
            raise RuntimeError(
                f"Monitor '{self.name}' is disabled."
            )

        result = self.monitor(data)

        if not isinstance(result, MonitoringResult):
            raise TypeError(
                "monitor() must return a MonitoringResult."
            )

        if result.monitor_name != self.name:
            raise ValueError(
                "MonitoringResult monitor_name must match "
                "the monitor configuration name."
            )

        return result

    @abstractmethod
    def monitor(
        self,
        data: Any,
    ) -> MonitoringResult:
        """
        Perform monitor-specific analysis.
        """
        raise NotImplementedError