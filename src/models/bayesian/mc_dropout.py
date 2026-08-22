from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from .network import BayesianRegimeNetwork


class MCDropoutInference:
    """
    Monte Carlo Dropout inference engine.

    Performs multiple stochastic forward passes with dropout layers
    active during inference. This allows the Bayesian regime model to
    estimate predictive uncertainty from the variation across samples.
    """

    def __init__(
        self,
        network: BayesianRegimeNetwork,
        mc_samples: int | None = None,
    ) -> None:
        if not isinstance(network, BayesianRegimeNetwork):
            raise TypeError(
                "network must be a BayesianRegimeNetwork instance."
            )

        if mc_samples is None:
            mc_samples = network.config.mc_samples

        self.network = network
        self.mc_samples = self._validate_positive_int(
            mc_samples,
            "mc_samples",
        )

    @staticmethod
    def _validate_positive_int(
        value: Any,
        field_name: str,
    ) -> int:
        """Validate a strictly positive integer."""
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(
                f"{field_name} must be an integer."
            )

        if value < 1:
            raise ValueError(
                f"{field_name} must be greater than or equal to 1."
            )

        return value

    def _validate_input(
        self,
        x: Tensor,
    ) -> Tensor:
        """Validate input against the underlying network."""
        if not isinstance(x, Tensor):
            raise TypeError(
                "x must be a torch.Tensor."
            )

        if x.ndim != 2:
            raise ValueError(
                "x must be a 2-dimensional tensor with shape "
                "(batch_size, input_dim)."
            )

        if x.shape[1] != self.network.input_dim:
            raise ValueError(
                f"Expected input_dim={self.network.input_dim}, "
                f"but received {x.shape[1]}."
            )

        return x.float()

    def _enable_dropout(self) -> None:
        """
        Activate dropout modules without globally switching the
        complete network into training mode.
        """
        for module in self.network.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def sample_probabilities(
        self,
        x: Tensor,
    ) -> Tensor:
        """
        Perform MC Dropout sampling.

        Returns
        -------
        Tensor
            Shape:
            (mc_samples, batch_size, n_regimes)
        """
        x = self._validate_input(x)

        was_training = self.network.training
        dropout_states = [
            (module, module.training)
            for module in self.network.modules()
            if isinstance(module, nn.Dropout)
        ]

        try:
            self.network.eval()
            self._enable_dropout()

            samples = []

            with torch.no_grad():
                for _ in range(self.mc_samples):
                    logits = self.network(x)

                    probabilities = torch.softmax(
                        logits,
                        dim=1,
                    )

                    samples.append(probabilities)

            return torch.stack(samples, dim=0)

        finally:
            self.network.train(was_training)

            for module, state in dropout_states:
                module.train(state)

    def mean_probabilities(
        self,
        x: Tensor,
    ) -> Tensor:
        """
        Return mean probabilities across MC samples.
        """
        samples = self.sample_probabilities(x)

        return samples.mean(dim=0)

    def predict(
        self,
        x: Tensor,
    ) -> Tensor:
        """
        Return the regime with the highest mean probability.
        """
        probabilities = self.mean_probabilities(x)

        return torch.argmax(
            probabilities,
            dim=1,
        )

    def confidence(
        self,
        x: Tensor,
    ) -> Tensor:
        """
        Return confidence of the final prediction.

        Confidence is the maximum mean regime probability.
        """
        probabilities = self.mean_probabilities(x)

        return torch.max(
            probabilities,
            dim=1,
        ).values

    def probability_std(
        self,
        x: Tensor,
    ) -> Tensor:
        """
        Return standard deviation of probabilities across MC samples.

        High values indicate greater epistemic uncertainty.
        """
        samples = self.sample_probabilities(x)

        return samples.std(
            dim=0,
            unbiased=False,
        )