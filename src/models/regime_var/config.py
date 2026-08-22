from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RegimeVARConfig:
    """
    Configuration for a regime-specific VAR model.
    """

    model_name: str = "regime_switching_var"
    n_regimes: int = 3
    lag_order: int = 1
    n_features: int = 1
    include_intercept: bool = True
    ridge_alpha: float = 0.0

    def __post_init__(self) -> None:
        self.model_name = self._validate_model_name(
            self.model_name
        )
        self.n_regimes = self._validate_positive_integer(
            self.n_regimes,
            "n_regimes",
        )
        self.lag_order = self._validate_positive_integer(
            self.lag_order,
            "lag_order",
        )
        self.n_features = self._validate_positive_integer(
            self.n_features,
            "n_features",
        )

        if not isinstance(self.include_intercept, bool):
            raise TypeError(
                "include_intercept must be a boolean."
            )

        self.ridge_alpha = self._validate_non_negative_float(
            self.ridge_alpha,
            "ridge_alpha",
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

        return value

    @staticmethod
    def _validate_positive_integer(
        value: Any,
        name: str,
    ) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(
                f"{name} must be an integer."
            )

        if value <= 0:
            raise ValueError(
                f"{name} must be greater than zero."
            )

        return value

    @staticmethod
    def _validate_non_negative_float(
        value: Any,
        name: str,
    ) -> float:
        if isinstance(value, bool) or not isinstance(
            value,
            (int, float),
        ):
            raise TypeError(
                f"{name} must be a number."
            )

        value = float(value)

        if value < 0.0:
            raise ValueError(
                f"{name} cannot be negative."
            )

        return value