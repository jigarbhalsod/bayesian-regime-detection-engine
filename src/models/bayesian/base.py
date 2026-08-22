from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .config import BayesianModelConfig


FloatArray = NDArray[np.float64]


@dataclass
class BayesianModelResult:
    """
    Container for predictions and uncertainty estimates
    produced by a Bayesian model.
    """

    mean: FloatArray
    variance: FloatArray
    metadata: dict[str, Any] | None = None


class BaseBayesianModel(ABC):
    """
    Abstract base class for Bayesian models.

    Defines the common lifecycle and validation utilities
    shared by Bayesian models in the project.
    """

    def __init__(
        self,
        config: BayesianModelConfig | None = None,
    ) -> None:
        self.config = (
            config
            if config is not None
            else BayesianModelConfig()
        )

        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        """
        Return whether the model has been fitted.
        """
        return self._is_fitted

    @abstractmethod
    def fit(
        self,
        data: Any,
        targets: Any,
        **kwargs: Any,
    ) -> "BaseBayesianModel":
        """
        Fit the Bayesian model.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        data: Any,
        **kwargs: Any,
    ) -> BayesianModelResult:
        """
        Generate predictions with uncertainty estimates.
        """
        raise NotImplementedError

    def _validate_fitted(self) -> None:
        """
        Raise an error if prediction is attempted before fitting.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Model must be fitted before prediction."
            )

    def _validate_features(
        self,
        data: Any,
    ) -> FloatArray:
        """
        Validate and convert model feature data.
        """
        if not isinstance(data, np.ndarray):
            raise TypeError(
                "data must be a numpy.ndarray."
            )

        if data.ndim != 2:
            raise ValueError(
                "data must be a 2-dimensional array."
            )

        if data.shape[0] == 0:
            raise ValueError(
                "data must contain at least one sample."
            )

        if data.shape[1] != self.config.n_features:
            raise ValueError(
                "data feature count does not match "
                "config.n_features."
            )

        if not np.issubdtype(
            data.dtype,
            np.number,
        ):
            raise TypeError(
                "data must contain numeric values."
            )

        if not np.all(np.isfinite(data)):
            raise ValueError(
                "data must contain only finite values."
            )

        return data.astype(
            np.float64,
            copy=False,
        )

    def _validate_targets(
        self,
        targets: Any,
        n_samples: int,
    ) -> FloatArray:
        """
        Validate and convert target data.
        """
        if not isinstance(targets, np.ndarray):
            raise TypeError(
                "targets must be a numpy.ndarray."
            )

        if targets.ndim == 1:
            targets = targets.reshape(-1, 1)

        if targets.ndim != 2:
            raise ValueError(
                "targets must be a 1D or 2D array."
            )

        if targets.shape[0] != n_samples:
            raise ValueError(
                "targets sample count must match data."
            )

        if targets.shape[1] != self.config.n_outputs:
            raise ValueError(
                "targets output count does not match "
                "config.n_outputs."
            )

        if not np.issubdtype(
            targets.dtype,
            np.number,
        ):
            raise TypeError(
                "targets must contain numeric values."
            )

        if not np.all(np.isfinite(targets)):
            raise ValueError(
                "targets must contain only finite values."
            )

        return targets.astype(
            np.float64,
            copy=False,
        )

    def get_metadata(self) -> dict[str, Any]:
        """
        Return common Bayesian model metadata.
        """
        return {
            "model_name": self.config.model_name,
            "n_features": self.config.n_features,
            "n_outputs": self.config.n_outputs,
            "is_fitted": self.is_fitted,
        }