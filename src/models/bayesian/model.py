from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from .config import BayesianModelConfig
from .mc_dropout import MCDropoutInference
from .network import BayesianRegimeNetwork
from .uncertainty import (
    BayesianUncertaintyEstimator,
    UncertaintyResult,
)


@dataclass(frozen=True)
class BayesianPrediction:
    """
    Complete Bayesian prediction output for a batch of samples.
    """

    predictions: Tensor
    probabilities: Tensor
    confidence: Tensor
    uncertainty: UncertaintyResult


class BayesianRegimeModel:
    """
    High-level Bayesian regime classification model.

    Integrates:

    - BayesianRegimeNetwork
    - Monte Carlo Dropout inference
    - Predictive uncertainty estimation
    """

    def __init__(
        self,
        input_dim: int,
        config: BayesianModelConfig | None = None,
    ) -> None:
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

        self.network = BayesianRegimeNetwork(
            input_dim=self.input_dim,
            config=self.config,
        )

        self.inference = MCDropoutInference(
            network=self.network,
        )

        self.uncertainty_estimator = (
            BayesianUncertaintyEstimator()
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
    ) -> "BayesianRegimeModel":
        """
        Train the Bayesian regime network.

        Uses standard cross-entropy optimization. Dropout remains active
        during training through the underlying PyTorch network.
        """
        x = self._validate_features(x)
        y = self._validate_targets(y, x.shape[0])

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

        criterion = torch.nn.CrossEntropyLoss()

        self.network.train()

        for _ in range(epochs):
            optimizer.zero_grad()

            logits = self.network(x)

            loss = criterion(
                logits,
                y,
            )

            loss.backward()

            optimizer.step()

        return self

    def predict(
        self,
        x: Tensor,
    ) -> Tensor:
        """
        Return predicted regime labels.
        """
        x = self._validate_features(x)

        return self.inference.predict(x)

    def predict_proba(
        self,
        x: Tensor,
    ) -> Tensor:
        """
        Return mean predictive probabilities from MC Dropout.
        """
        x = self._validate_features(x)

        return self.inference.mean_probabilities(x)

    def predict_with_uncertainty(
        self,
        x: Tensor,
    ) -> BayesianPrediction:
        """
        Return predictions, probabilities, confidence, and uncertainty.
        """
        x = self._validate_features(x)

        samples = self.inference.sample_probabilities(
            x
        )

        probabilities = samples.mean(dim=0)

        predictions = torch.argmax(
            probabilities,
            dim=1,
        )

        uncertainty = (
            self.uncertainty_estimator.estimate(
                samples
            )
        )

        return BayesianPrediction(
            predictions=predictions,
            probabilities=probabilities,
            confidence=uncertainty.confidence,
            uncertainty=uncertainty,
        )

    def _validate_features(
        self,
        x: Tensor,
    ) -> Tensor:
        if not isinstance(x, Tensor):
            raise TypeError(
                "x must be a torch.Tensor."
            )

        if x.ndim != 2:
            raise ValueError(
                "x must be a 2-dimensional tensor with shape "
                "(batch_size, input_dim)."
            )

        if x.shape[0] < 1:
            raise ValueError(
                "x must contain at least one sample."
            )

        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected input_dim={self.input_dim}, "
                f"but received {x.shape[1]}."
            )

        return x.float()

    def _validate_targets(
        self,
        y: Tensor,
        batch_size: int,
    ) -> Tensor:
        if not isinstance(y, Tensor):
            raise TypeError(
                "y must be a torch.Tensor."
            )

        if y.ndim != 1:
            raise ValueError(
                "y must be a 1-dimensional tensor."
            )

        if y.shape[0] != batch_size:
            raise ValueError(
                "x and y must contain the same number of samples."
            )

        if y.shape[0] < 1:
            raise ValueError(
                "y must contain at least one target."
            )

        if torch.is_floating_point(y):
            raise TypeError(
                "y must contain integer regime labels."
            )

        y = y.long()

        if torch.any(y < 0):
            raise ValueError(
                "y contains an invalid negative regime label."
            )

        if torch.any(y >= self.config.n_regimes):
            raise ValueError(
                "y contains a regime label outside the valid range."
            )

        return y

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