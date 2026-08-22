from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from .base import BaseFoundationModel
from .config import FoundationModelConfig


class FoundationModelAdapter(BaseFoundationModel):
    """
    Lightweight concrete time-series forecasting adapter.

    The adapter provides a trainable implementation of the
    BaseFoundationModel interface.

    Input:
        (batch_size, context_length, n_features)

    Output:
        (batch_size, forecast_horizon, output_dim)
    """

    def __init__(
        self,
        config: FoundationModelConfig | None = None,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__(config=config)

        self.hidden_dim = self._validate_positive_int(
            hidden_dim,
            "hidden_dim",
        )

        input_size = (
            self.config.context_length
            * self.config.n_features
        )

        output_size = (
            self.config.forecast_horizon
            * self.config.output_dim
        )

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                input_size,
                self.hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                self.hidden_dim,
                output_size,
            ),
        )

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

    def fit(
        self,
        x: Tensor,
        y: Tensor,
        epochs: int = 100,
        learning_rate: float = 1e-3,
        **kwargs: Any,
    ) -> "FoundationModelAdapter":
        """
        Train the forecasting adapter using mean squared error.
        """
        x = self._validate_sequence_input(x)
        y = self._validate_targets(
            y,
            batch_size=x.shape[0],
        )

        epochs = self._validate_positive_int(
            epochs,
            "epochs",
        )

        learning_rate = self._validate_learning_rate(
            learning_rate
        )

        optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=learning_rate,
        )

        criterion = nn.MSELoss()

        self.network.train()

        for _ in range(epochs):
            optimizer.zero_grad()

            predictions = self._forward(x)

            loss = criterion(
                predictions,
                y,
            )

            loss.backward()

            optimizer.step()

        self._is_fitted = True

        return self

    def _forward(
        self,
        x: Tensor,
    ) -> Tensor:
        """
        Run the underlying network and restore temporal shape.
        """
        output = self.network(x)

        return output.reshape(
            x.shape[0],
            self.config.forecast_horizon,
            self.config.output_dim,
        )

    def predict(
        self,
        x: Tensor,
        **kwargs: Any,
    ) -> Tensor:
        """
        Generate deterministic forecasts.
        """
        x = self._validate_sequence_input(x)

        if not self.is_fitted:
            raise RuntimeError(
                "FoundationModelAdapter must be fitted "
                "before prediction."
            )

        self.network.eval()

        with torch.no_grad():
            predictions = self._forward(x)

        return predictions

    def get_metadata(self) -> dict[str, Any]:
        """
        Return adapter and foundation-model metadata.
        """
        metadata = super().get_metadata()

        metadata.update(
            {
                "hidden_dim": self.hidden_dim,
                "adapter_type": self.__class__.__name__,
            }
        )

        return metadata

    @staticmethod
    def _validate_learning_rate(
        value: Any,
    ) -> float:
        if isinstance(value, bool) or not isinstance(
            value,
            (int, float),
        ):
            raise TypeError(
                "learning_rate must be a number."
            )

        value = float(value)

        if value <= 0:
            raise ValueError(
                "learning_rate must be greater than 0."
            )

        return value