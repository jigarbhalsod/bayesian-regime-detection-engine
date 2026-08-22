from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseBayesianModel, BayesianModelResult
from .config import BayesianModelConfig


class BayesianLinearRegression(BaseBayesianModel):
    """
    Bayesian linear regression model with Gaussian prior and
    Gaussian observation noise.
    """

    def __init__(
        self,
        config: BayesianModelConfig | None = None,
    ) -> None:
        super().__init__(config=config)

        self._posterior_mean: np.ndarray | None = None
        self._posterior_covariance: np.ndarray | None = None

    def fit(
        self,
        data: Any,
        targets: Any,
        **kwargs: Any,
    ) -> "BayesianLinearRegression":
        """
        Fit the Bayesian linear regression model.
        """
        features = self._validate_features(data)

        target_values = self._validate_targets(
            targets,
            n_samples=features.shape[0],
        )

        prior_variance = self.config.prior_std ** 2
        noise_variance = (
            self.config.observation_noise ** 2
        )

        prior_precision = 1.0 / prior_variance
        noise_precision = 1.0 / noise_variance

        identity = np.eye(
            self.config.n_features,
            dtype=np.float64,
        )

        posterior_precision = (
            prior_precision * identity
            + noise_precision * (features.T @ features)
        )

        posterior_covariance = np.linalg.pinv(
            posterior_precision
        )

        prior_mean = np.full(
            (
                self.config.n_features,
                self.config.n_outputs,
            ),
            self.config.prior_mean,
            dtype=np.float64,
        )

        posterior_mean = posterior_covariance @ (
            prior_precision * prior_mean
            + noise_precision
            * (features.T @ target_values)
        )

        self._posterior_mean = posterior_mean.astype(
            np.float64,
            copy=False,
        )

        self._posterior_covariance = (
            posterior_covariance.astype(
                np.float64,
                copy=False,
            )
        )

        self._is_fitted = True

        return self

    def predict(
        self,
        data: Any,
        **kwargs: Any,
    ) -> BayesianModelResult:
        """
        Predict posterior mean and predictive variance.
        """
        self._validate_fitted()

        features = self._validate_features(data)

        if (
            self._posterior_mean is None
            or self._posterior_covariance is None
        ):
            raise RuntimeError(
                "Posterior parameters are unavailable."
            )

        mean = features @ self._posterior_mean

        model_variance = np.sum(
            (features @ self._posterior_covariance)
            * features,
            axis=1,
            keepdims=True,
        )

        noise_variance = (
            self.config.observation_noise ** 2
        )

        variance = model_variance + noise_variance

        if self.config.n_outputs > 1:
            variance = np.repeat(
                variance,
                self.config.n_outputs,
                axis=1,
            )

        return BayesianModelResult(
            mean=mean.astype(
                np.float64,
                copy=False,
            ),
            variance=variance.astype(
                np.float64,
                copy=False,
            ),
        )

    def get_posterior_parameters(
        self,
    ) -> dict[str, np.ndarray]:
        """
        Return safe copies of the fitted posterior parameters.
        """
        self._validate_fitted()

        if (
            self._posterior_mean is None
            or self._posterior_covariance is None
        ):
            raise RuntimeError(
                "Posterior parameters are unavailable."
            )

        return {
            "mean": self._posterior_mean.copy(),
            "covariance": (
                self._posterior_covariance.copy()
            ),
        }

    def get_metadata(self) -> dict[str, Any]:
        """
        Return metadata describing the model state and configuration.
        """
        metadata = super().get_metadata()

        metadata.update(
            {
                "model_class": self.__class__.__name__,
                "n_features": self.config.n_features,
                "n_outputs": self.config.n_outputs,
                "prior_mean": self.config.prior_mean,
                "prior_std": self.config.prior_std,
                "observation_noise": (
                    self.config.observation_noise
                ),
                "has_posterior": (
                    self._posterior_mean is not None
                    and self._posterior_covariance is not None
                ),
            }
        )

        return metadata

    def get_model_summary(self) -> dict[str, Any]:
        """
        Return a summary of the fitted Bayesian model.
        """
        self._validate_fitted()

        if (
            self._posterior_mean is None
            or self._posterior_covariance is None
        ):
            raise RuntimeError(
                "Posterior parameters are unavailable."
            )

        return {
            "metadata": self.get_metadata(),
            "posterior_mean_shape": (
                self._posterior_mean.shape
            ),
            "posterior_covariance_shape": (
                self._posterior_covariance.shape
            ),
            "posterior_mean_finite": bool(
                np.all(np.isfinite(self._posterior_mean))
            ),
            "posterior_covariance_finite": bool(
                np.all(
                    np.isfinite(
                        self._posterior_covariance
                    )
                )
            ),
        }