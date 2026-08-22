from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


@dataclass(frozen=True)
class UncertaintyResult:
    """
    Container for Bayesian predictive uncertainty metrics.

    All tensors are per-sample and therefore have shape:
    (batch_size,)
    """

    predictive_entropy: Tensor
    expected_entropy: Tensor
    mutual_information: Tensor
    confidence: Tensor
    probability_std: Tensor
    normalized_predictive_entropy: Tensor
    epistemic_uncertainty: Tensor


class BayesianUncertaintyEstimator:
    """
    Estimate predictive and epistemic uncertainty from Monte Carlo
    probability samples.

    Expected input shape:
        (mc_samples, batch_size, n_regimes)
    """

    def __init__(
        self,
        epsilon: float = 1e-8,
    ) -> None:
        self.epsilon = self._validate_epsilon(
            epsilon
        )

    @staticmethod
    def _validate_epsilon(
        value: Any,
    ) -> float:
        """Validate numerical stability epsilon."""
        if isinstance(value, bool) or not isinstance(
            value,
            (int, float),
        ):
            raise TypeError(
                "epsilon must be a number."
            )

        value = float(value)

        if value <= 0:
            raise ValueError(
                "epsilon must be greater than 0."
            )

        return value

    def _validate_samples(
        self,
        samples: Tensor,
    ) -> Tensor:
        """
        Validate MC probability samples and normalize numerical dtype.

        Samples must have shape:
        (mc_samples, batch_size, n_regimes)
        """
        if not isinstance(samples, Tensor):
            raise TypeError(
                "samples must be a torch.Tensor."
            )

        if samples.ndim != 3:
            raise ValueError(
                "samples must be a 3-dimensional tensor with shape "
                "(mc_samples, batch_size, n_regimes)."
            )

        if samples.shape[0] < 1:
            raise ValueError(
                "samples must contain at least one MC sample."
            )

        if samples.shape[1] < 1:
            raise ValueError(
                "samples must contain at least one batch item."
            )

        if samples.shape[2] < 2:
            raise ValueError(
                "samples must contain at least two regimes."
            )

        if not torch.is_floating_point(samples):
            samples = samples.float()

        if not torch.isfinite(samples).all():
            raise ValueError(
                "samples must contain only finite values."
            )

        if torch.any(samples < 0):
            raise ValueError(
                "samples must not contain negative probabilities."
            )

        row_sums = samples.sum(dim=2)

        if not torch.allclose(
            row_sums,
            torch.ones_like(row_sums),
            atol=1e-5,
            rtol=1e-5,
        ):
            raise ValueError(
                "Each probability vector must sum to 1."
            )

        return samples

    def mean_probabilities(
        self,
        samples: Tensor,
    ) -> Tensor:
        """Return mean probability distribution across MC samples."""
        samples = self._validate_samples(samples)

        return samples.mean(dim=0)

    def predictive_entropy(
        self,
        samples: Tensor,
    ) -> Tensor:
        """
        Entropy of the mean predictive distribution.

        Higher entropy means greater overall predictive uncertainty.
        """
        probabilities = self.mean_probabilities(samples)

        safe_probabilities = probabilities.clamp_min(
            self.epsilon
        )

        return -torch.sum(
            probabilities
            * torch.log(safe_probabilities),
            dim=1,
        )

    def expected_entropy(
        self,
        samples: Tensor,
    ) -> Tensor:
        """
        Mean entropy of individual MC predictive distributions.
        """
        samples = self._validate_samples(samples)

        safe_samples = samples.clamp_min(
            self.epsilon
        )

        entropies = -torch.sum(
            samples * torch.log(safe_samples),
            dim=2,
        )

        return entropies.mean(dim=0)

    def mutual_information(
        self,
        samples: Tensor,
    ) -> Tensor:
        """
        Estimate epistemic uncertainty as mutual information.

        MI = predictive_entropy - expected_entropy
        """
        predictive = self.predictive_entropy(samples)
        expected = self.expected_entropy(samples)

        return torch.clamp(
            predictive - expected,
            min=0.0,
        )

    def confidence(
        self,
        samples: Tensor,
    ) -> Tensor:
        """Return maximum mean predictive probability."""
        probabilities = self.mean_probabilities(samples)

        return torch.max(
            probabilities,
            dim=1,
        ).values

    def probability_std(
        self,
        samples: Tensor,
    ) -> Tensor:
        """
        Mean standard deviation across regime probabilities.

        This provides a compact per-sample dispersion measure.
        """
        samples = self._validate_samples(samples)

        std = samples.std(
            dim=0,
            unbiased=False,
        )

        return std.mean(dim=1)

    def normalized_predictive_entropy(
        self,
        samples: Tensor,
    ) -> Tensor:
        """
        Predictive entropy normalized to [0, 1].

        0 = maximum certainty
        1 = maximum uncertainty
        """
        entropy = self.predictive_entropy(samples)

        n_regimes = samples.shape[2]

        maximum_entropy = torch.log(
            torch.tensor(
                float(n_regimes),
                dtype=entropy.dtype,
                device=entropy.device,
            )
        )

        return entropy / maximum_entropy

    def epistemic_uncertainty(
        self,
        samples: Tensor,
    ) -> Tensor:
        """
        Return normalized epistemic uncertainty.

        This is normalized mutual information in [0, 1].
        """
        mutual_information = self.mutual_information(
            samples
        )

        n_regimes = samples.shape[2]

        maximum_entropy = torch.log(
            torch.tensor(
                float(n_regimes),
                dtype=mutual_information.dtype,
                device=mutual_information.device,
            )
        )

        return mutual_information / maximum_entropy

    def estimate(
        self,
        samples: Tensor,
    ) -> UncertaintyResult:
        """
        Calculate all supported uncertainty metrics.
        """
        samples = self._validate_samples(samples)

        return UncertaintyResult(
            predictive_entropy=self.predictive_entropy(
                samples
            ),
            expected_entropy=self.expected_entropy(
                samples
            ),
            mutual_information=self.mutual_information(
                samples
            ),
            confidence=self.confidence(samples),
            probability_std=self.probability_std(
                samples
            ),
            normalized_predictive_entropy=(
                self.normalized_predictive_entropy(
                    samples
                )
            ),
            epistemic_uncertainty=(
                self.epistemic_uncertainty(
                    samples
                )
            ),
        )