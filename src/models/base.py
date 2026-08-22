from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

from .config import AdvancedModelConfig
from .result import AdvancedModelResult


class BaseAdvancedRegimeModel(ABC):
    """
    Base interface for all advanced regime detection models.

    Subclasses are expected to implement model-specific fitting,
    regime prediction, and probability prediction where supported.
    """

    def __init__(
        self,
        config: AdvancedModelConfig | None = None,
    ) -> None:
        if config is None:
            config = AdvancedModelConfig()

        if not isinstance(config, AdvancedModelConfig):
            raise TypeError(
                "config must be an AdvancedModelConfig instance."
            )

        self.config = config
        self._is_fitted = False

    @property
    def model_name(self) -> str:
        """
        Return the configured model name.
        """
        return self.config.model_name

    @property
    def is_fitted(self) -> bool:
        """
        Return whether the model has been fitted.
        """
        return self._is_fitted

    @property
    def n_regimes(self) -> int:
        """
        Return the configured number of regimes.
        """
        return self.config.n_regimes

    def validate_input(self, data: Any) -> Sequence[Any]:
        """
        Perform common input validation.

        Advanced subclasses may override this method with additional
        validation while preserving the basic input contract.
        """
        if data is None:
            raise ValueError("Input data cannot be None.")

        if isinstance(data, (str, bytes)):
            raise TypeError(
                "Input data must be a collection, not a string."
            )

        if hasattr(data, "__len__"):
            if len(data) == 0:
                raise ValueError("Input data cannot be empty.")

        return data

    @abstractmethod
    def fit(self, data: Any) -> "BaseAdvancedRegimeModel":
        """
        Fit the model to input data.
        """

    @abstractmethod
    def predict(self, data: Any) -> AdvancedModelResult:
        """
        Predict market regimes.
        """

    def predict_proba(self, data: Any) -> Any:
        """
        Predict regime probabilities.

        Subclasses supporting probability prediction should override
        this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support "
            "probability prediction."
        )

    def _mark_fitted(self) -> None:
        """
        Mark the model as fitted.

        Intended to be called by subclasses after successful training.
        """
        self._is_fitted = True

    def _mark_unfitted(self) -> None:
        """
        Mark the model as unfitted.
        """
        self._is_fitted = False

    def _require_fitted(self) -> None:
        """
        Raise an error if the model has not been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} must be fitted before "
                "prediction."
            )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model_name={self.model_name!r}, "
            f"n_regimes={self.n_regimes}, "
            f"is_fitted={self.is_fitted})"
        )