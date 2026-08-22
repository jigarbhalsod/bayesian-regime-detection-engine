from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from hmmlearn.hmm import GMMHMM

from ..base import BaseAdvancedRegimeModel
from ..result import AdvancedModelResult
from .config import HMMConfig


class GMMHMMRegimeModel(BaseAdvancedRegimeModel):
    """
    Gaussian Mixture Hidden Markov Model for market regime detection.

    Unlike a standard Gaussian HMM, each hidden regime can be modeled
    using multiple Gaussian mixture components.
    """

    def __init__(
        self,
        config: HMMConfig | None = None,
    ) -> None:
        if config is None:
            config = HMMConfig(
                model_name="gmm_hmm",
            )

        if not isinstance(config, HMMConfig):
            raise TypeError(
                "config must be an HMMConfig instance."
            )

        super().__init__(config)

        self._model: GMMHMM | None = None
        self._n_features: int | None = None

    @property
    def gmm_config(self) -> HMMConfig:
        """Return the GMM-HMM configuration."""
        return self.config

    @property
    def model(self) -> GMMHMM | None:
        """Return the underlying hmmlearn GMMHMM."""
        return self._model

    @property
    def n_features(self) -> int | None:
        """Return the number of features used during fitting."""
        return self._n_features

    def validate_input(
        self,
        data: Any,
    ) -> np.ndarray:
        """
        Validate input and convert it to a finite 2D float array.
        """
        super().validate_input(data)

        try:
            array = np.asarray(data, dtype=float)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "Input data must contain numeric values."
            ) from exc

        if array.ndim != 2:
            raise ValueError(
                "Input data must be a 2-dimensional array "
                "with shape (n_samples, n_features)."
            )

        if array.shape[0] == 0:
            raise ValueError(
                "Input data must contain at least one sample."
            )

        if array.shape[1] == 0:
            raise ValueError(
                "Input data must contain at least one feature."
            )

        if not np.isfinite(array).all():
            raise ValueError(
                "Input data must contain only finite values."
            )

        return array

    def _validate_feature_columns(
        self,
        data: np.ndarray,
    ) -> None:
        """
        Ensure configured feature columns match the input width.
        """
        feature_columns = self.gmm_config.feature_columns

        if (
            feature_columns is not None
            and len(feature_columns) != data.shape[1]
        ):
            raise ValueError(
                "Number of feature_columns must match the number "
                "of columns in input data."
            )

    def _validate_prediction_features(
        self,
        data: np.ndarray,
    ) -> None:
        """
        Ensure prediction data has the same feature count as training.
        """
        if self._n_features is None:
            raise RuntimeError(
                "Model feature count is unavailable. "
                "Fit the model before prediction."
            )

        if data.shape[1] != self._n_features:
            raise ValueError(
                "Prediction data must contain the same number "
                "of features used during fitting."
            )

    def _create_model(self) -> GMMHMM:
        """
        Create a configured hmmlearn GMMHMM instance.

        GMMHMM uses mixture weights internally, so we do not alter
        the existing Gaussian-HMM params/init_params contract.
        """
        return GMMHMM(
            n_components=self.gmm_config.n_regimes,
            n_mix=self.gmm_config.n_mix,
            covariance_type=self.gmm_config.covariance_type,
            min_covar=self.gmm_config.min_covar,
            n_iter=self.gmm_config.n_iter,
            tol=self.gmm_config.tol,
            algorithm=self.gmm_config.algorithm,
            random_state=self.gmm_config.random_state,
            verbose=self.gmm_config.verbose,
        )

    def fit(
        self,
        data: Any,
    ) -> "GMMHMMRegimeModel":
        """
        Fit the GMM-HMM to sequential market features.
        """
        array = self.validate_input(data)

        self._validate_feature_columns(array)

        if array.shape[0] < self.n_regimes:
            raise ValueError(
                "Number of samples must be greater than or equal "
                "to n_regimes."
            )

        self._mark_unfitted()

        model = self._create_model()

        try:
            model.fit(array)
        except Exception:
            self._model = None
            self._n_features = None
            raise

        self._model = model
        self._n_features = int(array.shape[1])

        self._mark_fitted()

        return self

    def predict(
        self,
        data: Any,
    ) -> AdvancedModelResult:
        """
        Predict hidden regimes and return standardized model results.
        """
        self._require_fitted()

        array = self.validate_input(data)
        self._validate_prediction_features(array)

        if self._model is None:
            raise RuntimeError(
                "Underlying GMMHMM is unavailable."
            )

        regimes = self._model.predict(array)
        probabilities = self._model.predict_proba(array)
        confidence = probabilities.max(axis=1)

        return AdvancedModelResult(
            regimes=regimes.astype(int).tolist(),
            probabilities=probabilities.astype(float).tolist(),
            confidence=confidence.astype(float).tolist(),
            model_name=self.model_name,
            metadata={
                "n_regimes": self.n_regimes,
                "n_mix": self.gmm_config.n_mix,
                "n_features": self.n_features,
                "covariance_type": self.gmm_config.covariance_type,
                "algorithm": self.gmm_config.algorithm,
            },
            diagnostics=self.get_diagnostics(),
        )

    def predict_proba(
        self,
        data: Any,
    ) -> List[List[float]]:
        """
        Return posterior regime probabilities.
        """
        self._require_fitted()

        array = self.validate_input(data)
        self._validate_prediction_features(array)

        if self._model is None:
            raise RuntimeError(
                "Underlying GMMHMM is unavailable."
            )

        probabilities = self._model.predict_proba(array)

        return probabilities.astype(float).tolist()

    def score(
        self,
        data: Any,
    ) -> float:
        """
        Return sequence log likelihood.
        """
        self._require_fitted()

        array = self.validate_input(data)
        self._validate_prediction_features(array)

        if self._model is None:
            raise RuntimeError(
                "Underlying GMMHMM is unavailable."
            )

        return float(
            self._model.score(array)
        )

    def get_transition_matrix(self) -> List[List[float]]:
        """
        Return the learned regime transition matrix.
        """
        self._require_fitted()

        if self._model is None:
            raise RuntimeError(
                "Underlying GMMHMM is unavailable."
            )

        return (
            np.asarray(
                self._model.transmat_,
                dtype=float,
            )
            .tolist()
        )

    def get_state_counts(
        self,
        data: Any,
    ) -> List[int]:
        """
        Return predicted observation counts for every regime.
        """
        self._require_fitted()

        array = self.validate_input(data)
        self._validate_prediction_features(array)

        if self._model is None:
            raise RuntimeError(
                "Underlying GMMHMM is unavailable."
            )

        regimes = self._model.predict(array)

        counts = np.bincount(
            regimes,
            minlength=self.n_regimes,
        )

        return counts.astype(int).tolist()

    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Return training and convergence diagnostics.
        """
        self._require_fitted()

        if self._model is None:
            raise RuntimeError(
                "Underlying GMMHMM is unavailable."
            )

        monitor = self._model.monitor_

        return {
            "converged": bool(monitor.converged),
            "iterations": int(monitor.iter),
            "log_likelihood_history": [
                float(value)
                for value in monitor.history
            ],
            "n_mix": self.gmm_config.n_mix,
        }