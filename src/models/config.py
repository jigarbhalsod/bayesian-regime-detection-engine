from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AdvancedModelConfig:
    """
    Common configuration for advanced regime detection models.

    This configuration is intentionally model-agnostic so that HMM,
    Bayesian, switching, and deep learning models can share a common
    configuration contract.
    """

    model_name: str = "advanced_model"
    feature_columns: Optional[List[str]] = None
    n_regimes: int = 3
    random_state: Optional[int] = 42
    model_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.model_name = self._validate_model_name(self.model_name)
        self.feature_columns = self._validate_feature_columns(
            self.feature_columns
        )
        self.n_regimes = self._validate_n_regimes(self.n_regimes)
        self.random_state = self._validate_random_state(
            self.random_state
        )
        self.model_params = self._validate_model_params(
            self.model_params
        )

    @staticmethod
    def _validate_model_name(value: Any) -> str:
        if not isinstance(value, str):
            raise TypeError("model_name must be a string.")

        value = value.strip()

        if not value:
            raise ValueError("model_name cannot be empty.")

        return value

    @staticmethod
    def _validate_feature_columns(
        value: Optional[List[str]],
    ) -> Optional[List[str]]:
        if value is None:
            return None

        if not isinstance(value, (list, tuple)):
            raise TypeError(
                "feature_columns must be a list, tuple, or None."
            )

        validated: List[str] = []

        for column in value:
            if not isinstance(column, str):
                raise TypeError(
                    "Each feature column must be a string."
                )

            column = column.strip()

            if not column:
                raise ValueError(
                    "Feature column names cannot be empty."
                )

            validated.append(column)

        if len(validated) != len(set(validated)):
            raise ValueError(
                "feature_columns cannot contain duplicates."
            )

        return validated

    @staticmethod
    def _validate_n_regimes(value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("n_regimes must be an integer.")

        if value < 1:
            raise ValueError(
                "n_regimes must be greater than or equal to 1."
            )

        return value

    @staticmethod
    def _validate_random_state(value: Any) -> Optional[int]:
        if value is None:
            return None

        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(
                "random_state must be an integer or None."
            )

        return value

    @staticmethod
    def _validate_model_params(value: Any) -> Dict[str, Any]:
        if value is None:
            return {}

        if not isinstance(value, dict):
            raise TypeError("model_params must be a dictionary.")

        return dict(value)

    def get_param(self, name: str, default: Any = None) -> Any:
        """
        Return a model-specific parameter.
        """
        return self.model_params.get(name, default)

    def set_param(self, name: str, value: Any) -> None:
        """
        Set or update a model-specific parameter.
        """
        if not isinstance(name, str):
            raise TypeError("Parameter name must be a string.")

        name = name.strip()

        if not name:
            raise ValueError("Parameter name cannot be empty.")

        self.model_params[name] = value

    def to_dict(self) -> Dict[str, Any]:
        """
        Return a serializable copy of the configuration.
        """
        return {
            "model_name": self.model_name,
            "feature_columns": (
                list(self.feature_columns)
                if self.feature_columns is not None
                else None
            ),
            "n_regimes": self.n_regimes,
            "random_state": self.random_state,
            "model_params": dict(self.model_params),
        }