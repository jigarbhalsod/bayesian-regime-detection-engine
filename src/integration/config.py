from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class IntegrationConfig:
    """
    Configuration for cross-model integration.

    Parameters
    ----------
    integration_name:
        Human-readable identifier for the integration workflow.
    min_models:
        Minimum number of models required for execution.
    fail_fast:
        Whether execution should stop immediately when a model fails.
    include_metadata:
        Whether model metadata should be included in integration results.
    """

    integration_name: str = "model_integration"
    min_models: int = 1
    fail_fast: bool = True
    include_metadata: bool = True

    def __post_init__(self) -> None:
        if not isinstance(
            self.integration_name,
            str,
        ):
            raise TypeError(
                "integration_name must be a string."
            )

        if not self.integration_name.strip():
            raise ValueError(
                "integration_name must not be empty."
            )

        if isinstance(self.min_models, bool) or not isinstance(
            self.min_models,
            int,
        ):
            raise TypeError(
                "min_models must be an integer."
            )

        if self.min_models <= 0:
            raise ValueError(
                "min_models must be greater than 0."
            )

        if not isinstance(self.fail_fast, bool):
            raise TypeError(
                "fail_fast must be a boolean."
            )

        if not isinstance(self.include_metadata, bool):
            raise TypeError(
                "include_metadata must be a boolean."
            )

    def to_dict(self) -> dict[str, Any]:
        """
        Return the configuration as a dictionary.
        """
        return asdict(self)