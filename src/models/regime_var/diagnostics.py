from __future__ import annotations

from typing import Any

import numpy as np

from .model import RegimeSwitchingVAR


class RegimeVARDiagnostics:
    """
    Evaluation and diagnostic utilities for RegimeSwitchingVAR.
    """

    def __init__(
        self,
        model: RegimeSwitchingVAR,
    ) -> None:
        if not isinstance(model, RegimeSwitchingVAR):
            raise TypeError(
                "model must be a RegimeSwitchingVAR instance."
            )

        self.model = model

    def evaluate(
        self,
        data: Any,
        regimes: Any,
    ) -> dict[str, Any]:
        """
        Evaluate fitted regime-specific predictions.

        Returns overall, per-feature, and per-regime error metrics.
        """
        if not self.model.is_fitted:
            raise RuntimeError(
                "Model must be fitted before evaluation."
            )

        prepared = self.model._preparer.prepare(
            data=data,
            regimes=regimes,
        )

        if prepared.regimes is None:
            raise ValueError(
                "regimes are required for evaluation."
            )

        predictions: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        evaluated_regimes: list[int] = []

        for index, regime in enumerate(prepared.regimes):
            regime_index = int(regime)

            try:
                prediction = self.model.predict(
                    history=prepared.features[index].reshape(
                        self.model.config.lag_order,
                        self.model.config.n_features,
                    ),
                    regime=regime_index,
                )
            except ValueError:
                # Skip samples belonging to regimes
                # with no fitted parameters.
                continue

            predictions.append(prediction)
            targets.append(prepared.targets[index])
            evaluated_regimes.append(regime_index)

        if not predictions:
            raise ValueError(
                "No evaluable samples were available."
            )

        prediction_array = np.asarray(
            predictions,
            dtype=np.float64,
        )
        target_array = np.asarray(
            targets,
            dtype=np.float64,
        )
        regime_array = np.asarray(
            evaluated_regimes,
            dtype=np.int64,
        )

        errors = prediction_array - target_array

        mse = float(np.mean(errors**2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(errors)))

        per_feature_mse = np.mean(
            errors**2,
            axis=0,
        )
        per_feature_rmse = np.sqrt(
            per_feature_mse
        )
        per_feature_mae = np.mean(
            np.abs(errors),
            axis=0,
        )

        per_feature = {
            index: {
                "mse": float(per_feature_mse[index]),
                "rmse": float(per_feature_rmse[index]),
                "mae": float(per_feature_mae[index]),
            }
            for index in range(
                self.model.config.n_features
            )
        }

        per_regime: dict[int, dict[str, float | int]] = {}

        for regime in sorted(np.unique(regime_array)):
            mask = regime_array == regime
            regime_errors = errors[mask]

            regime_mse = float(
                np.mean(regime_errors**2)
            )

            per_regime[int(regime)] = {
                "sample_count": int(mask.sum()),
                "mse": regime_mse,
                "rmse": float(np.sqrt(regime_mse)),
                "mae": float(
                    np.mean(np.abs(regime_errors))
                ),
            }

        return {
            "n_samples": int(target_array.shape[0]),
            "n_features": int(target_array.shape[1]),
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "per_feature": per_feature,
            "per_regime": per_regime,
        }

    def model_summary(self) -> dict[str, Any]:
        """
        Return fitted model configuration and parameter diagnostics.
        """
        metadata = self.model.get_metadata()

        summary: dict[str, Any] = {
            **metadata,
            "regime_parameters": {},
        }

        if not self.model.is_fitted:
            return summary

        regime_parameters: dict[int, dict[str, Any]] = {}

        for regime in metadata["fitted_regimes"]:
            parameters = self.model.get_regime_parameters(
                regime
            )

            regime_parameters[regime] = {
                "coefficient_shape": tuple(
                    parameters["coefficients"].shape
                ),
                "intercept_shape": tuple(
                    parameters["intercept"].shape
                ),
                "sample_count": parameters[
                    "sample_count"
                ],
            }

        summary["regime_parameters"] = regime_parameters

        return summary