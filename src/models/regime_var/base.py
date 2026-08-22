from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .config import RegimeVARConfig


FloatArray = NDArray[np.float64]


class BaseRegimeVARModel(ABC):
    """
    Abstract base class for regime-specific VAR models.
    """

    def __init__(
        self,
        config: RegimeVARConfig | None = None,
    ) -> None:
        if config is None:
            config = RegimeVARConfig()

        if not isinstance(config, RegimeVARConfig):
            raise TypeError(
                "config must be a RegimeVARConfig instance."
            )

        self.config = config
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        """Return whether the model has been fitted."""
        return self._is_fitted

    def _validate_fitted(self) -> None:
        """Raise an error if the model has not been fitted."""
        if not self._is_fitted:
            raise RuntimeError(
                "Model must be fitted before use."
            )

    def _validate_history(
        self,
        history: Any,
    ) -> FloatArray:
        """
        Validate prediction history.

        Expected shape:
            (lag_order, n_features)
        """
        if not isinstance(history, np.ndarray):
            raise TypeError(
                "history must be a numpy array."
            )

        if history.ndim != 2:
            raise ValueError(
                "history must be two-dimensional."
            )

        expected_shape = (
            self.config.lag_order,
            self.config.n_features,
        )

        if history.shape != expected_shape:
            raise ValueError(
                "history must have shape "
                f"{expected_shape}."
            )

        if not np.issubdtype(
            history.dtype,
            np.number,
        ):
            raise TypeError(
                "history must contain numeric values."
            )

        values = history.astype(
            np.float64,
            copy=False,
        )

        if not np.all(np.isfinite(values)):
            raise ValueError(
                "history must contain only finite values."
            )

        return values

    def _validate_regime_index(
        self,
        regime: Any,
    ) -> int:
        """Validate and return a regime index."""
        if isinstance(regime, bool) or not isinstance(
            regime,
            (int, np.integer),
        ):
            raise TypeError(
                "regime must be an integer."
            )

        regime_index = int(regime)

        if not (
            0 <= regime_index
            < self.config.n_regimes
        ):
            raise ValueError(
                "regime must be within the configured "
                "regime range."
            )

        return regime_index

    @abstractmethod
    def fit(
        self,
        data: Any,
        regimes: Any,
        **kwargs: Any,
    ) -> "BaseRegimeVARModel":
        """Fit the model."""
        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        history: Any,
        regime: Any,
        **kwargs: Any,
    ) -> FloatArray:
        """Predict the next observation."""
        raise NotImplementedError

    @abstractmethod
    def get_metadata(self) -> dict[str, Any]:
        """Return model metadata."""
        raise NotImplementedError