from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .base import FloatArray


IntArray = np.ndarray


@dataclass(frozen=True)
class RegimeVARPreparedData:
    """
    Prepared supervised samples for a regime-specific VAR model.

    features shape:
        (n_samples, lag_order * n_features)

    targets shape:
        (n_samples, n_features)

    regimes shape:
        (n_samples,)
    """

    features: FloatArray
    targets: FloatArray
    regimes: IntArray | None


class RegimeVARPreparer:
    """
    Convert multivariate time-series data into lagged supervised
    samples suitable for regime-specific VAR fitting.
    """

    def __init__(
        self,
        lag_order: int,
        n_features: int,
    ) -> None:
        self.lag_order = self._validate_positive_integer(
            lag_order,
            "lag_order",
        )
        self.n_features = self._validate_positive_integer(
            n_features,
            "n_features",
        )

    def prepare(
        self,
        data: Any,
        regimes: Any = None,
    ) -> RegimeVARPreparedData:
        """
        Prepare lagged features, targets, and aligned regime labels.

        Input data shape:
            (n_observations, n_features)

        Output:
            Each sample uses the previous `lag_order` observations
            to predict the next observation.
        """
        values = self._validate_data(data)

        n_observations = values.shape[0]

        if n_observations <= self.lag_order:
            raise ValueError(
                "data must contain more observations than "
                "lag_order."
            )

        aligned_regimes = self._validate_regimes(
            regimes=regimes,
            n_observations=n_observations,
        )

        n_samples = n_observations - self.lag_order

        features = np.empty(
            (
                n_samples,
                self.lag_order * self.n_features,
            ),
            dtype=np.float64,
        )

        targets = np.empty(
            (
                n_samples,
                self.n_features,
            ),
            dtype=np.float64,
        )

        for index in range(n_samples):
            history = values[
                index : index + self.lag_order
            ]

            features[index] = history.reshape(-1)
            targets[index] = values[
                index + self.lag_order
            ]

        if aligned_regimes is not None:
            aligned_regimes = aligned_regimes[
                self.lag_order :
            ].astype(np.int64, copy=False)

        return RegimeVARPreparedData(
            features=features,
            targets=targets,
            regimes=aligned_regimes,
        )

    def _validate_data(
        self,
        data: Any,
    ) -> FloatArray:
        """Validate the multivariate time-series input."""
        if not isinstance(data, np.ndarray):
            raise TypeError(
                "data must be a numpy array."
            )

        if data.ndim != 2:
            raise ValueError(
                "data must be two-dimensional."
            )

        if data.shape[1] != self.n_features:
            raise ValueError(
                "data feature count does not match "
                "n_features."
            )

        if not np.issubdtype(
            data.dtype,
            np.number,
        ):
            raise TypeError(
                "data must contain numeric values."
            )

        values = data.astype(
            np.float64,
            copy=False,
        )

        if not np.all(np.isfinite(values)):
            raise ValueError(
                "data must contain only finite values."
            )

        return values

    def _validate_regimes(
        self,
        regimes: Any,
        n_observations: int,
    ) -> IntArray | None:
        """Validate optional observation-level regime labels."""
        if regimes is None:
            return None

        if not isinstance(regimes, np.ndarray):
            raise TypeError(
                "regimes must be a numpy array."
            )

        if regimes.ndim != 1:
            raise ValueError(
                "regimes must be one-dimensional."
            )

        if regimes.shape[0] != n_observations:
            raise ValueError(
                "regimes length must match the number of "
                "observations."
            )

        if not np.issubdtype(
            regimes.dtype,
            np.integer,
        ):
            raise TypeError(
                "regimes must contain integer values."
            )

        return regimes.astype(
            np.int64,
            copy=False,
        )

    @staticmethod
    def _validate_positive_integer(
        value: Any,
        name: str,
    ) -> int:
        if isinstance(value, bool) or not isinstance(
            value,
            (int, np.integer),
        ):
            raise TypeError(
                f"{name} must be an integer."
            )

        value = int(value)

        if value <= 0:
            raise ValueError(
                f"{name} must be greater than zero."
            )

        return value