from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from .config import BayesianModelConfig


class BayesianRegimeNetwork(nn.Module):
    """
    Feed-forward neural network for regime classification.

    Dropout layers remain part of the architecture so they can later
    be activated during Monte Carlo Dropout inference.

    Parameters
    ----------
    input_dim:
        Number of input features.
    config:
        Bayesian neural model configuration.
    """

    def __init__(
        self,
        input_dim: int,
        config: BayesianModelConfig | None = None,
    ) -> None:
        super().__init__()

        if config is None:
            config = BayesianModelConfig()

        if not isinstance(config, BayesianModelConfig):
            raise TypeError(
                "config must be a BayesianModelConfig instance."
            )

        self.input_dim = self._validate_positive_int(
            input_dim,
            "input_dim",
        )

        self.config = config
        self.n_regimes = self._validate_positive_int(
            config.n_regimes,
            "n_regimes",
        )

        layers: list[nn.Module] = []
        previous_dim = self.input_dim

        for hidden_dim in config.hidden_dims:
            layers.extend(
                [
                    nn.Linear(
                        previous_dim,
                        hidden_dim,
                    ),
                    nn.ReLU(),
                    nn.Dropout(
                        p=config.dropout_rate
                    ),
                ]
            )

            previous_dim = hidden_dim

        self.hidden_layers = nn.Sequential(
            *layers
        )

        self.output_layer = nn.Linear(
            previous_dim,
            self.n_regimes,
        )

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

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        """
        Run a forward pass and return regime logits.

        Parameters
        ----------
        x:
            Input tensor with shape:
            (batch_size, input_dim)

        Returns
        -------
        Tensor
            Regime logits with shape:
            (batch_size, n_regimes)
        """
        if not isinstance(x, Tensor):
            raise TypeError(
                "x must be a torch.Tensor."
            )

        if x.ndim != 2:
            raise ValueError(
                "x must be a 2-dimensional tensor with shape "
                "(batch_size, input_dim)."
            )

        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected input_dim={self.input_dim}, "
                f"but received {x.shape[1]}."
            )

        x = x.float()

        hidden = self.hidden_layers(x)

        return self.output_layer(hidden)

    def predict_proba(
        self,
        x: Tensor,
    ) -> Tensor:
        """
        Return normalized regime probabilities.

        The network is temporarily evaluated in inference mode.
        The original training/evaluation mode is restored afterwards.
        """
        if not isinstance(x, Tensor):
            raise TypeError(
                "x must be a torch.Tensor."
            )

        was_training = self.training

        try:
            self.eval()

            with torch.no_grad():
                logits = self.forward(x)
                probabilities = torch.softmax(
                    logits,
                    dim=1,
                )

            return probabilities
        finally:
            self.train(was_training)

    def predict(
        self,
        x: Tensor,
    ) -> Tensor:
        """
        Return the most probable regime for each input sample.
        """
        probabilities = self.predict_proba(x)

        return torch.argmax(
            probabilities,
            dim=1,
        )