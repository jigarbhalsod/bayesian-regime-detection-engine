from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.models.particle_filter.config import (
    ParticleFilterConfig,
)


@dataclass(frozen=True)
class ParticleFilterResult:
    """
    Result returned by particle-filter estimation.
    """

    state_mean: np.ndarray
    state_covariance: np.ndarray
    effective_sample_size: float
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseParticleFilter(ABC):
    """
    Abstract base class for particle-filter implementations.
    """

    def __init__(
        self,
        config: ParticleFilterConfig | None = None,
    ) -> None:
        self.config = config or ParticleFilterConfig()

        if not isinstance(
            self.config,
            ParticleFilterConfig,
        ):
            raise TypeError(
                "config must be a ParticleFilterConfig."
            )

        self._is_fitted = False
        self._n_samples_seen = 0

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def _validate_observations(
        self,
        observations: np.ndarray,
    ) -> np.ndarray:
        if not isinstance(observations, np.ndarray):
            raise TypeError(
                "observations must be a numpy array."
            )

        if observations.ndim != 2:
            raise ValueError(
                "observations must be a 2-dimensional array."
            )

        if observations.shape[0] == 0:
            raise ValueError(
                "observations must not be empty."
            )

        if (
            observations.shape[1]
            != self.config.observation_dim
        ):
            raise ValueError(
                "observations feature dimension does not "
                "match config.observation_dim."
            )

        if not np.issubdtype(
            observations.dtype,
            np.number,
        ):
            raise TypeError(
                "observations must contain numeric values."
            )

        result = observations.astype(
            np.float64,
            copy=False,
        )

        if not np.all(np.isfinite(result)):
            raise ValueError(
                "observations must contain only finite values."
            )

        return result

    def _validate_states(
        self,
        states: np.ndarray,
    ) -> np.ndarray:
        if not isinstance(states, np.ndarray):
            raise TypeError(
                "states must be a numpy array."
            )

        if states.ndim != 2:
            raise ValueError(
                "states must be a 2-dimensional array."
            )

        if states.shape[0] == 0:
            raise ValueError(
                "states must not be empty."
            )

        if states.shape[1] != self.config.state_dim:
            raise ValueError(
                "states feature dimension does not match "
                "config.state_dim."
            )

        if not np.issubdtype(
            states.dtype,
            np.number,
        ):
            raise TypeError(
                "states must contain numeric values."
            )

        result = states.astype(
            np.float64,
            copy=False,
        )

        if not np.all(np.isfinite(result)):
            raise ValueError(
                "states must contain only finite values."
            )

        return result

    def _validate_weights(
        self,
        weights: np.ndarray,
        n_particles: int | None = None,
    ) -> np.ndarray:
        if not isinstance(weights, np.ndarray):
            raise TypeError(
                "weights must be a numpy array."
            )

        if weights.ndim != 1:
            raise ValueError(
                "weights must be a 1-dimensional array."
            )

        expected_particles = (
            n_particles
            if n_particles is not None
            else self.config.n_particles
        )

        if weights.shape[0] != expected_particles:
            raise ValueError(
                "weights length must match the number "
                "of particles."
            )

        if not np.issubdtype(
            weights.dtype,
            np.number,
        ):
            raise TypeError(
                "weights must contain numeric values."
            )

        result = weights.astype(
            np.float64,
            copy=False,
        )

        if not np.all(np.isfinite(result)):
            raise ValueError(
                "weights must contain only finite values."
            )

        if np.any(result < 0):
            raise ValueError(
                "weights must not contain negative values."
            )

        total = float(result.sum())

        if total <= 0:
            raise ValueError(
                "weights must have a positive sum."
            )

        return result / total

    @abstractmethod
    def fit(
        self,
        observations: np.ndarray,
    ) -> BaseParticleFilter:
        """
        Fit or initialize the filter from observations.
        """

    @abstractmethod
    def predict(
        self,
        observations: np.ndarray,
    ) -> ParticleFilterResult:
        """
        Estimate latent states from observations.
        """

    def get_metadata(self) -> dict[str, Any]:
        """
        Return model metadata.
        """
        return {
            "model_name": self.config.model_name,
            "n_particles": self.config.n_particles,
            "state_dim": self.config.state_dim,
            "observation_dim": self.config.observation_dim,
            "is_fitted": self.is_fitted,
            "n_samples_seen": self._n_samples_seen,
        }