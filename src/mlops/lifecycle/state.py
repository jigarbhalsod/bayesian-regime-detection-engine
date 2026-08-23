from __future__ import annotations

from dataclasses import dataclass

from src.mlops.versioning import ModelVersion

from .stage import LifecycleStage


@dataclass(frozen=True)
class ModelLifecycleState:
    """
    Immutable lifecycle state for a specific model version.
    """

    model_version: ModelVersion
    stage: LifecycleStage = LifecycleStage.NONE

    def __post_init__(self) -> None:
        if not isinstance(self.model_version, ModelVersion):
            raise TypeError(
                "model_version must be a ModelVersion."
            )

        if not isinstance(self.stage, LifecycleStage):
            raise TypeError(
                "stage must be a LifecycleStage."
            )