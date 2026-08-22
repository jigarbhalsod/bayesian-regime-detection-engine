from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseRegimeVARModel, FloatArray
from .config import RegimeVARConfig
from .preparation import RegimeVARPreparer


class RegimeSwitchingVAR(BaseRegimeVARModel):
    """
    Regime-specific Vector Autoregression model.

    A separate multivariate linear VAR model is fitted for
    every configured regime.

    The model uses externally supplied regime labels during
    training and requires the desired regime during prediction.
    """

    def __init__(
        self,
        config: RegimeVARConfig | None = None,
    ) -> None:
        super().__init__(config=config)

        self._preparer = RegimeVARPreparer(
            lag_order=self.config.lag_order,
            n_features=self.config.n_features,
        )

        self._coefficients: dict[int, FloatArray] = {}
        self._intercepts: dict[int, FloatArray] = {}
        self._sample_counts: dict[int, int] = {}

    def fit(
        self,
        data: Any,
        regimes: Any,
        **kwargs: Any,
    ) -> "RegimeSwitchingVAR":
        """
        Fit a separate VAR equation for each regime.
        """
        prepared = self._preparer.prepare(
            data=data,
            regimes=regimes,
        )

        if prepared.regimes is None:
            raise ValueError(
                "regimes are required for model fitting."
            )

        self._coefficients = {}
        self._intercepts = {}
        self._sample_counts = {}

        for regime in range(self.config.n_regimes):
            mask = prepared.regimes == regime

            x = prepared.features[mask]
            y = prepared.targets[mask]

            if x.shape[0] == 0:
                continue

            coefficients, intercept = self._fit_regime(
                x=x,
                y=y,
            )

            self._coefficients[regime] = coefficients
            self._intercepts[regime] = intercept
            self._sample_counts[regime] = int(
                x.shape[0]
            )

        if not self._coefficients:
            raise ValueError(
                "No training samples matched the configured "
                "regimes."
            )

        self._is_fitted = True

        return self

    def _fit_regime(
        self,
        x: FloatArray,
        y: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:
        """
        Fit one regime-specific multivariate linear VAR model.
        """
        if self.config.include_intercept:
            design = np.column_stack(
                [
                    np.ones(
                        x.shape[0],
                        dtype=np.float64,
                    ),
                    x,
                ]
            )
        else:
            design = x

        if self.config.ridge_alpha > 0:
            penalty = (
                self.config.ridge_alpha
                * np.eye(
                    design.shape[1],
                    dtype=np.float64,
                )
            )

            if self.config.include_intercept:
                penalty[0, 0] = 0.0

            parameters = np.linalg.pinv(
                design.T @ design + penalty
            ) @ design.T @ y
        else:
            parameters = np.linalg.pinv(
                design
            ) @ y

        if self.config.include_intercept:
            intercept = parameters[0]
            coefficients = parameters[1:]
        else:
            intercept = np.zeros(
                self.config.n_features,
                dtype=np.float64,
            )
            coefficients = parameters

        return (
            coefficients.astype(
                np.float64,
                copy=False,
            ),
            intercept.astype(
                np.float64,
                copy=False,
            ),
        )

    def predict(
        self,
        history: Any,
        regime: Any,
        **kwargs: Any,
    ) -> FloatArray:
        """
        Predict the next multivariate observation for one regime.
        """
        self._validate_fitted()

        history_values = self._validate_history(history)
        regime_index = self._validate_regime_index(
            regime
        )

        if regime_index not in self._coefficients:
            raise ValueError(
                f"Regime {regime_index} has no fitted parameters."
            )

        x = history_values.reshape(-1)

        coefficients = self._coefficients[regime_index]
        intercept = self._intercepts[regime_index]

        prediction = x @ coefficients + intercept

        return prediction.astype(
            np.float64,
            copy=False,
        )

    def get_regime_parameters(
        self,
        regime: Any,
    ) -> dict[str, Any]:
        """
        Return parameters fitted for one regime.
        """
        self._validate_fitted()

        regime_index = self._validate_regime_index(
            regime
        )

        if regime_index not in self._coefficients:
            raise ValueError(
                f"Regime {regime_index} has no fitted parameters."
            )

        return {
            "regime": regime_index,
            "coefficients": self._coefficients[
                regime_index
            ].copy(),
            "intercept": self._intercepts[
                regime_index
            ].copy(),
            "sample_count": self._sample_counts[
                regime_index
            ],
        }

    def get_metadata(self) -> dict[str, Any]:
        """
        Return model configuration and fitting diagnostics.
        """
        return {
            "model_name": self.config.model_name,
            "n_regimes": self.config.n_regimes,
            "lag_order": self.config.lag_order,
            "n_features": self.config.n_features,
            "include_intercept": (
                self.config.include_intercept
            ),
            "ridge_alpha": self.config.ridge_alpha,
            "is_fitted": self.is_fitted,
            "fitted_regimes": sorted(
                self._coefficients.keys()
            ),
            "sample_counts": self._sample_counts.copy(),
        }