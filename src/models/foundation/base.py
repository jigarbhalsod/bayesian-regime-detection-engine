from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from torch import Tensor

from .config import FoundationModelConfig


class BaseFoundationModel(ABC):
    """
    Abstract interface for time-series foundation models.

    All concrete implementations must provide:

    - fit()
    - predict()
    - predict_proba()
    - get_metadata()
    """

    def __init__(
        self,
        config: FoundationModelConfig | None = None,
    ) -> None:
        if config is None:
            config = FoundationModelConfig()

        if not isinstance(
            config,
            FoundationModelConfig,
        ):
            raise TypeError(
                "config must be a FoundationModelConfig instance."
            )

        self.config = config
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        """Return whether the model has been fitted."""
        return self._is_fitted

    def _validate_sequence_input(
        self,
        x: Tensor,
    ) -> Tensor:
        """
        Validate temporal input.

        Expected shape:
            (batch_size, context_length, n_features)
        """
        if not isinstance(x, Tensor):
            raise TypeError(
                "x must be a torch.Tensor."
            )

        if x.ndim != 3:
            raise ValueError(
                "x must be a 3-dimensional tensor with shape "
                "(batch_size, context_length, n_features)."
            )

        if x.shape[0] < 1:
            raise ValueError(
                "x must contain at least one sample."
            )

        if x.shape[1] != self.config.context_length:
            raise ValueError(
                f"Expected context_length="
                f"{self.config.context_length}, "
                f"but received {x.shape[1]}."
            )

        if x.shape[2] != self.config.n_features:
            raise ValueError(
                f"Expected n_features="
                f"{self.config.n_features}, "
                f"but received {x.shape[2]}."
            )

        return x.float()

    def _validate_targets(
        self,
        y: Tensor,
        batch_size: int,
    ) -> Tensor:
        """
        Validate forecast targets.

        Expected shape:
            (batch_size, forecast_horizon, output_dim)
        """
        if not isinstance(y, Tensor):
            raise TypeError(
                "y must be a torch.Tensor."
            )

        if y.ndim != 3:
            raise ValueError(
                "y must be a 3-dimensional tensor with shape "
                "(batch_size, forecast_horizon, output_dim)."
            )

        if y.shape[0] != batch_size:
            raise ValueError(
                "x and y must contain the same batch size."
            )

        if y.shape[1] != self.config.forecast_horizon:
            raise ValueError(
                f"Expected forecast_horizon="
                f"{self.config.forecast_horizon}, "
                f"but received {y.shape[1]}."
            )

        if y.shape[2] != self.config.output_dim:
            raise ValueError(
                f"Expected output_dim="
                f"{self.config.output_dim}, "
                f"but received {y.shape[2]}."
            )

        return y.float()

    @abstractmethod
    def fit(
        self,
        x: Tensor,
        y: Tensor,
        **kwargs: Any,
    ) -> "BaseFoundationModel":
        """Fit the foundation model."""
        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        x: Tensor,
        **kwargs: Any,
    ) -> Tensor:
        """
        Generate forecasts.

        Returns shape:
            (batch_size, forecast_horizon, output_dim)
        """
        raise NotImplementedError

    def predict_proba(
        self,
        x: Tensor,
        **kwargs: Any,
    ) -> Tensor:
        """
        Return probabilistic forecasts when supported.

        Concrete models may override this method.
        """
        raise NotImplementedError(
            "Probabilistic prediction is not implemented "
            "for this foundation model."
        )

    def get_metadata(self) -> dict[str, Any]:
        """Return model metadata."""
        return {
            "model_name": self.config.model_name,
            "context_length": self.config.context_length,
            "forecast_horizon": (
                self.config.forecast_horizon
            ),
            "n_features": self.config.n_features,
            "output_dim": self.config.output_dim,
            "is_fitted": self.is_fitted,
        }