from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class FoundationModelConfig:
    """
    Configuration contract for time-series foundation models.

    Parameters
    ----------
    model_name:
        Identifier of the foundation model implementation.
    context_length:
        Number of historical observations supplied to the model.
    forecast_horizon:
        Number of future observations to forecast.
    n_features:
        Number of features in each time step.
    output_dim:
        Number of values produced per forecast step.
    """

    model_name: str = "foundation_model"
    context_length: int = 60
    forecast_horizon: int = 5
    n_features: int = 1
    output_dim: int = 1

    def __post_init__(self) -> None:
        self.model_name = self._validate_model_name(
            self.model_name
        )

        self.context_length = self._validate_positive_int(
            self.context_length,
            "context_length",
        )

        self.forecast_horizon = self._validate_positive_int(
            self.forecast_horizon,
            "forecast_horizon",
        )

        self.n_features = self._validate_positive_int(
            self.n_features,
            "n_features",
        )

        self.output_dim = self._validate_positive_int(
            self.output_dim,
            "output_dim",
        )

    @staticmethod
    def _validate_model_name(value: Any) -> str:
        if not isinstance(value, str):
            raise TypeError(
                "model_name must be a string."
            )

        value = value.strip()

        if not value:
            raise ValueError(
                "model_name cannot be empty."
            )

        return value.lower()

    @staticmethod
    def _validate_positive_int(
        value: Any,
        field_name: str,
    ) -> int:
        if isinstance(value, bool) or not isinstance(
            value,
            int,
        ):
            raise TypeError(
                f"{field_name} must be an integer."
            )

        if value < 1:
            raise ValueError(
                f"{field_name} must be greater than or equal to 1."
            )

        return value