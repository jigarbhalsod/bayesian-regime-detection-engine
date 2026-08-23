from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping


@dataclass(frozen=True)
class ExperimentConfig:
    """
    Immutable configuration for an experiment.
    """

    experiment_name: str
    description: str | None = None
    tags: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        experiment_name = self.experiment_name.strip()

        if not experiment_name:
            raise ValueError("experiment_name must not be empty.")

        if any(not isinstance(tag, str) or not tag.strip() for tag in self.tags):
            raise ValueError("tags must contain only non-empty strings.")

        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping.")

        object.__setattr__(self, "experiment_name", experiment_name)
        object.__setattr__(self, "tags", tuple(tag.strip() for tag in self.tags))
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata)),
        )