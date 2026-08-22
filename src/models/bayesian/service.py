from __future__ import annotations

from typing import Any

from .base import BayesianModelResult
from .config import BayesianModelConfig
from .linear import BayesianLinearRegression


class BayesianModelService:
    """
    High-level service wrapper for Bayesian models.

    This layer provides a stable integration interface for
    creating, fitting, predicting, and inspecting the current
    Bayesian model implementation.
    """

    def __init__(
        self,
        config: BayesianModelConfig | None = None,
    ) -> None:
        self._model = BayesianLinearRegression(
            config=config,
        )

    @property
    def model(self) -> BayesianLinearRegression:
        """
        Return the underlying Bayesian model instance.
        """
        return self._model

    @property
    def is_fitted(self) -> bool:
        """
        Return whether the underlying model is fitted.
        """
        return self._model.is_fitted

    def fit(
        self,
        data: Any,
        targets: Any,
        **kwargs: Any,
    ) -> "BayesianModelService":
        """
        Fit the underlying Bayesian model.
        """
        self._model.fit(
            data=data,
            targets=targets,
            **kwargs,
        )

        return self

    def predict(
        self,
        data: Any,
        **kwargs: Any,
    ) -> BayesianModelResult:
        """
        Generate Bayesian predictions.
        """
        return self._model.predict(
            data=data,
            **kwargs,
        )

    def get_metadata(self) -> dict[str, Any]:
        """
        Return metadata from the underlying model.
        """
        return self._model.get_metadata()

    def get_model_summary(self) -> dict[str, Any]:
        """
        Return a fitted model summary.
        """
        return self._model.get_model_summary()

    def get_posterior_parameters(self) -> dict[str, Any]:
        """
        Return fitted posterior parameters.
        """
        return self._model.get_posterior_parameters()