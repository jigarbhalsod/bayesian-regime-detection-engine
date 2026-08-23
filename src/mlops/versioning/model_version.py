from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .version import Version


@dataclass(frozen=True)
class ModelVersion:
    """
    Immutable version record for a trained model.
    """

    model_name: str
    version: Version
    model_type: str
    experiment_id: str | None = None
    source_run_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        model_name = self.model_name.strip()
        model_type = self.model_type.strip()

        if not model_name:
            raise ValueError("model_name must not be empty.")

        if not model_type:
            raise ValueError("model_type must not be empty.")

        if not isinstance(self.version, Version):
            raise TypeError("version must be a Version.")

        if self.experiment_id is not None:
            experiment_id = self.experiment_id.strip()

            if not experiment_id:
                raise ValueError(
                    "experiment_id must not be empty when provided."
                )

            object.__setattr__(
                self,
                "experiment_id",
                experiment_id,
            )

        if self.source_run_id is not None:
            source_run_id = self.source_run_id.strip()

            if not source_run_id:
                raise ValueError(
                    "source_run_id must not be empty when provided."
                )

            object.__setattr__(
                self,
                "source_run_id",
                source_run_id,
            )

        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping.")

        object.__setattr__(self, "model_name", model_name)
        object.__setattr__(self, "model_type", model_type)
        object.__setattr__(
            self,
            "metadata",
            dict(self.metadata),
        )