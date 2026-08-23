from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class MonitoringConfig:
    """
    Central configuration for monitoring components.
    """

    name: str
    enabled: bool = True
    interval_seconds: int = 60
    thresholds: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized_name = self.name.strip()

        if not normalized_name:
            raise ValueError("Monitoring config name cannot be empty.")

        if self.interval_seconds <= 0:
            raise ValueError(
                "interval_seconds must be greater than zero."
            )

        normalized_thresholds: dict[str, float] = {}

        for key, value in self.thresholds.items():
            normalized_key = str(key).strip()

            if not normalized_key:
                raise ValueError(
                    "Threshold names cannot be empty."
                )

            if isinstance(value, bool) or not isinstance(
                value,
                (int, float),
            ):
                raise TypeError(
                    f"Threshold '{normalized_key}' must be numeric."
                )

            normalized_thresholds[normalized_key] = float(value)

        normalized_metadata: dict[str, Any] = {
            str(key).strip(): value
            for key, value in self.metadata.items()
        }

        if any(not key for key in normalized_metadata):
            raise ValueError(
                "Metadata keys cannot be empty."
            )

        object.__setattr__(self, "name", normalized_name)
        object.__setattr__(
            self,
            "thresholds",
            normalized_thresholds,
        )
        object.__setattr__(
            self,
            "metadata",
            normalized_metadata,
        )

    def get_threshold(
        self,
        name: str,
        default: float | None = None,
    ) -> float | None:
        """
        Retrieve a configured threshold.
        """
        return self.thresholds.get(name, default)