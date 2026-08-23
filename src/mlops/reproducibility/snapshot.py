from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .environment import EnvironmentSnapshot
from .seed import SeedState


@dataclass(frozen=True)
class ReproducibilitySnapshot:
    """
    Immutable record containing the information required
    to describe the reproducibility state of a run.
    """

    seed: SeedState
    environment: EnvironmentSnapshot
    experiment_id: str | None = None
    run_id: str | None = None
    dataset_version: str | None = None
    feature_version: str | None = None
    configuration: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.seed, SeedState):
            raise TypeError("seed must be a SeedState.")

        if not isinstance(
            self.environment,
            EnvironmentSnapshot,
        ):
            raise TypeError(
                "environment must be an EnvironmentSnapshot."
            )

        for name, value in (
            ("experiment_id", self.experiment_id),
            ("run_id", self.run_id),
            ("dataset_version", self.dataset_version),
            ("feature_version", self.feature_version),
        ):
            if value is not None:
                value = value.strip()

                if not value:
                    raise ValueError(
                        f"{name} must not be empty when provided."
                    )

                object.__setattr__(self, name, value)

        if self.configuration is not None:
            if not isinstance(self.configuration, Mapping):
                raise TypeError(
                    "configuration must be a mapping or None."
                )

            object.__setattr__(
                self,
                "configuration",
                dict(self.configuration),
            )