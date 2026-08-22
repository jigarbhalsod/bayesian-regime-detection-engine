from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from ..config import AdvancedModelConfig


@dataclass
class BayesianModelConfig(AdvancedModelConfig):
    """
    Configuration for Bayesian neural regime models.

    The initial Bayesian implementation uses Monte Carlo Dropout
    to estimate predictive uncertainty.
    """

    model_name: str = "bayesian_neural"

    hidden_dims: Tuple[int, ...] = (64, 32)
    dropout_rate: float = 0.2
    mc_samples: int = 30

    learning_rate: float = 1e-3
    batch_size: int = 32
    n_epochs: int = 50

    weight_decay: float = 0.0

    def __post_init__(self) -> None:
        """Validate and normalize Bayesian model configuration."""
        super().__post_init__()

        self.hidden_dims = self._validate_hidden_dims(
            self.hidden_dims
        )

        self.dropout_rate = self._validate_dropout_rate(
            self.dropout_rate
        )

        self.mc_samples = self._validate_positive_int(
            self.mc_samples,
            "mc_samples",
        )

        self.learning_rate = self._validate_positive_float(
            self.learning_rate,
            "learning_rate",
        )

        self.batch_size = self._validate_positive_int(
            self.batch_size,
            "batch_size",
        )

        self.n_epochs = self._validate_positive_int(
            self.n_epochs,
            "n_epochs",
        )

        self.weight_decay = self._validate_non_negative_float(
            self.weight_decay,
            "weight_decay",
        )

    @staticmethod
    def _validate_hidden_dims(
        value: Any,
    ) -> Tuple[int, ...]:
        """
        Validate hidden neural-network layer dimensions.
        """
        if not isinstance(value, (tuple, list)):
            raise TypeError(
                "hidden_dims must be a tuple or list of integers."
            )

        if len(value) == 0:
            raise ValueError(
                "hidden_dims cannot be empty."
            )

        normalized = []

        for dimension in value:
            if isinstance(dimension, bool) or not isinstance(
                dimension,
                int,
            ):
                raise TypeError(
                    "Every hidden dimension must be an integer."
                )

            if dimension < 1:
                raise ValueError(
                    "Every hidden dimension must be greater "
                    "than or equal to 1."
                )

            normalized.append(dimension)

        return tuple(normalized)

    @staticmethod
    def _validate_dropout_rate(
        value: Any,
    ) -> float:
        """
        Validate dropout probability.

        Dropout must be in the interval [0, 1).
        """
        if isinstance(value, bool) or not isinstance(
            value,
            (int, float),
        ):
            raise TypeError(
                "dropout_rate must be a number."
            )

        value = float(value)

        if value < 0 or value >= 1:
            raise ValueError(
                "dropout_rate must be greater than or equal to 0 "
                "and less than 1."
            )

        return value

    @staticmethod
    def _validate_positive_int(
        value: Any,
        field_name: str,
    ) -> int:
        """Validate a strictly positive integer."""
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

    @staticmethod
    def _validate_positive_float(
        value: Any,
        field_name: str,
    ) -> float:
        """Validate a strictly positive numeric value."""
        if isinstance(value, bool) or not isinstance(
            value,
            (int, float),
        ):
            raise TypeError(
                f"{field_name} must be a number."
            )

        value = float(value)

        if value <= 0:
            raise ValueError(
                f"{field_name} must be greater than 0."
            )

        return value

    @staticmethod
    def _validate_non_negative_float(
        value: Any,
        field_name: str,
    ) -> float:
        """Validate a non-negative numeric value."""
        if isinstance(value, bool) or not isinstance(
            value,
            (int, float),
        ):
            raise TypeError(
                f"{field_name} must be a number."
            )

        value = float(value)

        if value < 0:
            raise ValueError(
                f"{field_name} must be greater than or equal to 0."
            )

        return value

    def to_dict(self) -> Dict[str, Any]:
        """Convert the Bayesian configuration into a dictionary."""
        data = super().to_dict()

        data.update(
            {
                "hidden_dims": self.hidden_dims,
                "dropout_rate": self.dropout_rate,
                "mc_samples": self.mc_samples,
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "n_epochs": self.n_epochs,
                "weight_decay": self.weight_decay,
            }
        )

        return data