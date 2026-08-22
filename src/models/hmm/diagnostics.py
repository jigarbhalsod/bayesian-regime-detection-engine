from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from .model import GaussianHMMRegimeModel


class HMMRegimeDiagnostics:
    """
    Provides interpretation and diagnostic utilities for a fitted
    GaussianHMMRegimeModel.

    Hidden-state labels produced by an HMM are numerical identifiers.
    This class derives descriptive statistics and transition-based
    metrics to make those states easier to analyse.
    """

    def __init__(
        self,
        model: GaussianHMMRegimeModel,
    ) -> None:
        if not isinstance(model, GaussianHMMRegimeModel):
            raise TypeError(
                "model must be a GaussianHMMRegimeModel instance."
            )

        if not model.is_fitted:
            raise RuntimeError(
                "The HMM model must be fitted before diagnostics "
                "can be generated."
            )

        self._model = model

    @property
    def model(self) -> GaussianHMMRegimeModel:
        """
        Return the fitted HMM model.
        """
        return self._model

    @property
    def n_regimes(self) -> int:
        """
        Return the number of hidden regimes.
        """
        return self._model.n_regimes

    def _validate_data(
        self,
        data: Any,
    ) -> np.ndarray:
        """
        Validate diagnostic input using the model's own validation rules.
        """
        array = self._model.validate_input(data)
        self._model._validate_prediction_features(array)

        return array

    def get_state_statistics(
        self,
        data: Any,
    ) -> List[Dict[str, Any]]:
        """
        Calculate descriptive statistics for each inferred hidden state.

        Each state's statistics contain:
        - state index
        - observation count
        - proportion of observations
        - feature-wise mean
        - feature-wise standard deviation
        """
        array = self._validate_data(data)

        if self._model.model is None:
            raise RuntimeError(
                "Underlying GaussianHMM is unavailable."
            )

        states = self._model.model.predict(array)

        results: List[Dict[str, Any]] = []

        for state in range(self.n_regimes):
            mask = states == state
            observations = array[mask]
            count = int(mask.sum())

            if count == 0:
                means = np.full(
                    array.shape[1],
                    np.nan,
                )
                stds = np.full(
                    array.shape[1],
                    np.nan,
                )
            else:
                means = observations.mean(axis=0)
                stds = observations.std(axis=0)

            results.append(
                {
                    "state": state,
                    "count": count,
                    "proportion": float(
                        count / len(array)
                    ),
                    "mean": means.astype(float).tolist(),
                    "std": stds.astype(float).tolist(),
                }
            )

        return results

    def get_transition_diagnostics(
        self,
    ) -> List[Dict[str, Any]]:
        """
        Calculate transition and persistence metrics for every state.

        Expected duration is derived as:

            1 / (1 - P(state -> same state))

        Infinite duration is returned when the self-transition
        probability is numerically equal to 1.
        """
        matrix = np.asarray(
            self._model.get_transition_matrix(),
            dtype=float,
        )

        results: List[Dict[str, Any]] = []

        for state in range(self.n_regimes):
            self_probability = float(
                matrix[state, state]
            )

            if np.isclose(
                self_probability,
                1.0,
            ):
                expected_duration = float("inf")
            else:
                expected_duration = float(
                    1.0 / (1.0 - self_probability)
                )

            transition_probability = float(
                1.0 - self_probability
            )

            results.append(
                {
                    "state": state,
                    "self_transition_probability": (
                        self_probability
                    ),
                    "exit_probability": (
                        transition_probability
                    ),
                    "expected_duration": (
                        expected_duration
                    ),
                }
            )

        return results

    def get_state_ordering(
        self,
        data: Any,
        feature_index: int = 0,
        ascending: bool = True,
    ) -> List[int]:
        """
        Order states according to their mean value for one feature.

        States with no assigned observations are placed after states
        with valid means.
        """
        if (
            isinstance(feature_index, bool)
            or not isinstance(feature_index, int)
        ):
            raise TypeError(
                "feature_index must be an integer."
            )

        if feature_index < 0:
            raise ValueError(
                "feature_index must be greater than or equal to 0."
            )

        if not isinstance(ascending, bool):
            raise TypeError(
                "ascending must be a boolean."
            )

        statistics = self.get_state_statistics(data)

        n_features = self._model.n_features

        if (
            n_features is None
            or feature_index >= n_features
        ):
            raise ValueError(
                "feature_index is outside the available "
                "feature range."
            )

        valid_states = []
        empty_states = []

        for item in statistics:
            mean = item["mean"][feature_index]

            if np.isnan(mean):
                empty_states.append(item["state"])
            else:
                valid_states.append(
                    (
                        item["state"],
                        float(mean),
                    )
                )

        valid_states.sort(
            key=lambda item: item[1],
            reverse=not ascending,
        )

        return (
            [state for state, _ in valid_states]
            + empty_states
        )

    def get_summary(
        self,
        data: Any,
    ) -> Dict[str, Any]:
        """
        Return a combined regime diagnostics summary.
        """
        array = self._validate_data(data)

        state_statistics = self.get_state_statistics(array)
        transition_diagnostics = (
            self.get_transition_diagnostics()
        )

        state_counts = [
            item["count"]
            for item in state_statistics
        ]

        return {
            "n_regimes": self.n_regimes,
            "n_features": int(array.shape[1]),
            "n_samples": int(array.shape[0]),
            "state_counts": state_counts,
            "state_statistics": state_statistics,
            "transition_diagnostics": (
                transition_diagnostics
            ),
        }