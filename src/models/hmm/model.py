from __future__ import annotations

from typing import Any, List

import numpy as np
from hmmlearn.hmm import GaussianHMM

from ..base import BaseAdvancedRegimeModel
from ..result import AdvancedModelResult
from .config import HMMConfig


class GaussianHMMRegimeModel(BaseAdvancedRegimeModel):
    """
    Gaussian Hidden Markov Model for market regime detection.

    The model learns latent market states from sequential numerical
    feature data and returns standardized AdvancedModelResult objects.
    """

    def __init__(
        self,
        config: HMMConfig | None = None,
    ) -> None:
        if config is None:
            config = HMMConfig()

        if not isinstance(config, HMMConfig):
            raise TypeError(
                "config must be an HMMConfig instance."
            )

        super().__init__(config)

        self._model: GaussianHMM | None = None
        self._n_features: int | None = None

    @property
    def hmm_config(self) -> HMMConfig:
        """
        Return the model configuration with HMM-specific typing.
        """
        return self.config

    @property
    def model(self) -> GaussianHMM | None:
        """
        Return the underlying hmmlearn GaussianHMM instance.
        """
        return self._model

    @property
    def n_features(self) -> int | None:
        """
        Return the number of features observed during fitting.
        """
        return self._n_features

    def validate_input(self, data: Any) -> np.ndarray:
        """
        Validate and convert input into a finite 2-dimensional array.
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
        Validate configured feature columns against input width.
        """
        feature_columns = self.hmm_config.feature_columns

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

    def _create_model(self) -> GaussianHMM:
        """
        Create a configured hmmlearn GaussianHMM instance.
        """
        return GaussianHMM(
            n_components=self.hmm_config.n_regimes,
            covariance_type=self.hmm_config.covariance_type,
            min_covar=self.hmm_config.min_covar,
            n_iter=self.hmm_config.n_iter,
            tol=self.hmm_config.tol,
            params=self.hmm_config.params,
            init_params=self.hmm_config.init_params,
            random_state=self.hmm_config.random_state,
            verbose=self.hmm_config.verbose,
            **self.hmm_config.model_params,
        )

    def fit(
        self,
        data: Any,
    ) -> "GaussianHMMRegimeModel":
        """
        Fit the Gaussian HMM to sequential market features.
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
        self._n_features = array.shape[1]

        self._mark_fitted()

        return self

    def predict(
        self,
        data: Any,
    ) -> AdvancedModelResult:
        """
        Predict hidden market regimes and state probabilities.
        """
        self._require_fitted()

        array = self.validate_input(data)
        self._validate_prediction_features(array)

        if self._model is None:
            raise RuntimeError(
                "Underlying GaussianHMM is unavailable."
            )

        regimes = self._model.predict(array)
        probabilities = self._model.predict_proba(array)

        confidence = probabilities.max(axis=1)

        return AdvancedModelResult(
            regimes=regimes.tolist(),
            probabilities=probabilities.tolist(),
            confidence=confidence.tolist(),
            model_name=self.model_name,
            metadata={
                "n_regimes": self.n_regimes,
                "n_features": self.n_features,
                "covariance_type": (
                    self.hmm_config.covariance_type
                ),
            },
            diagnostics=self.get_diagnostics(),
        )

    def predict_proba(
        self,
        data: Any,
    ) -> List[List[float]]:
        """
        Predict posterior probabilities for each hidden regime.
        """
        self._require_fitted()

        array = self.validate_input(data)
        self._validate_prediction_features(array)

        if self._model is None:
            raise RuntimeError(
                "Underlying GaussianHMM is unavailable."
            )

        probabilities = self._model.predict_proba(array)

        return probabilities.tolist()

    def score(
        self,
        data: Any,
    ) -> float:
        """
        Return the log likelihood of the input sequence.
        """
        self._require_fitted()

        array = self.validate_input(data)
        self._validate_prediction_features(array)

        if self._model is None:
            raise RuntimeError(
                "Underlying GaussianHMM is unavailable."
            )

        return float(self._model.score(array))

    def get_transition_matrix(self) -> List[List[float]]:
        """
        Return the learned hidden-state transition matrix.
        """
        self._require_fitted()

        if self._model is None:
            raise RuntimeError(
                "Underlying GaussianHMM is unavailable."
            )

        return self._model.transmat_.tolist()

    def get_state_counts(
        self,
        data: Any,
    ) -> List[int]:
        """
        Return the number of predictions assigned to each regime.
        """
        self._require_fitted()

        array = self.validate_input(data)
        self._validate_prediction_features(array)

        if self._model is None:
            raise RuntimeError(
                "Underlying GaussianHMM is unavailable."
            )

        regimes = self._model.predict(array)

        counts = np.bincount(
            regimes,
            minlength=self.n_regimes,
        )

        return counts.astype(int).tolist()

    def get_diagnostics(self) -> dict:
        """
        Return currently available HMM training diagnostics.
        """
        self._require_fitted()

        if self._model is None:
            raise RuntimeError(
                "Underlying GaussianHMM is unavailable."
            )

        monitor = self._model.monitor_

        history = [
            float(value)
            for value in monitor.history
        ]

        return {
            "converged": bool(monitor.converged),
            "iterations": int(monitor.iter),
            "log_likelihood_history": history,
        }