from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.models.particle_filter.config import ParticleFilterConfig


@dataclass
class ParticleState:
    """
    Container and manager for particle states and their weights.
    """

    config: ParticleFilterConfig
    particles: np.ndarray | None = None
    weights: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.config, ParticleFilterConfig):
            raise TypeError(
                "config must be a ParticleFilterConfig."
            )

        if self.particles is not None:
            self.particles = self._validate_particles(
                self.particles
            )

        if self.weights is not None:
            self.weights = self._validate_weights(
                self.weights
            )

        if (
            self.particles is not None
            and self.weights is not None
            and self.particles.shape[0]
            != self.weights.shape[0]
        ):
            raise ValueError(
                "particles and weights must have the same "
                "number of entries."
            )

    @property
    def is_initialized(self) -> bool:
        """
        Return whether particles and weights are initialized.
        """
        return (
            self.particles is not None
            and self.weights is not None
        )

    @property
    def n_particles(self) -> int:
        """
        Return the configured number of particles.
        """
        return self.config.n_particles

    @property
    def state_dim(self) -> int:
        """
        Return the configured state dimension.
        """
        return self.config.state_dim

    def _validate_particles(
        self,
        particles: np.ndarray,
    ) -> np.ndarray:
        if not isinstance(particles, np.ndarray):
            raise TypeError(
                "particles must be a numpy array."
            )

        if particles.ndim != 2:
            raise ValueError(
                "particles must be a 2-dimensional array."
            )

        expected_shape = (
            self.config.n_particles,
            self.config.state_dim,
        )

        if particles.shape != expected_shape:
            raise ValueError(
                "particles shape must match "
                "(n_particles, state_dim)."
            )

        if not np.issubdtype(
            particles.dtype,
            np.number,
        ):
            raise TypeError(
                "particles must contain numeric values."
            )

        result = particles.astype(
            np.float64,
            copy=False,
        )

        if not np.all(np.isfinite(result)):
            raise ValueError(
                "particles must contain only finite values."
            )

        return result.copy()

    def _validate_weights(
        self,
        weights: np.ndarray,
    ) -> np.ndarray:
        if not isinstance(weights, np.ndarray):
            raise TypeError(
                "weights must be a numpy array."
            )

        if weights.ndim != 1:
            raise ValueError(
                "weights must be a 1-dimensional array."
            )

        if weights.shape[0] != self.config.n_particles:
            raise ValueError(
                "weights length must match n_particles."
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

        return (result / total).copy()

    def initialize(
        self,
        particles: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> ParticleState:
        """
        Initialize the particle state.

        If weights are omitted, uniform weights are assigned.
        """
        validated_particles = self._validate_particles(
            particles
        )

        if weights is None:
            validated_weights = np.full(
                self.config.n_particles,
                1.0 / self.config.n_particles,
                dtype=np.float64,
            )
        else:
            validated_weights = self._validate_weights(
                weights
            )

        self.particles = validated_particles
        self.weights = validated_weights

        return self

    def set_particles(
        self,
        particles: np.ndarray,
    ) -> ParticleState:
        """
        Replace the current particle matrix.
        """
        self.particles = self._validate_particles(
            particles
        )
        return self

    def set_weights(
        self,
        weights: np.ndarray,
    ) -> ParticleState:
        """
        Replace and normalize the current particle weights.
        """
        self.weights = self._validate_weights(
            weights
        )
        return self

    def weighted_mean(self) -> np.ndarray:
        """
        Compute the weighted mean of the particle states.
        """
        self._require_initialized()

        return np.average(
            self.particles,
            axis=0,
            weights=self.weights,
        )

    def weighted_covariance(self) -> np.ndarray:
        """
        Compute the weighted covariance matrix of particle states.
        """
        self._require_initialized()

        mean = self.weighted_mean()
        centered = self.particles - mean

        covariance = (
            centered.T
            @ (centered * self.weights[:, None])
        )

        return covariance

    def effective_sample_size(self) -> float:
        """
        Compute the effective sample size (ESS).
        """
        self._require_initialized()

        return float(
            1.0 / np.sum(np.square(self.weights))
        )

    def reset(self) -> ParticleState:
        """
        Clear particles and weights.
        """
        self.particles = None
        self.weights = None

        return self

    def get_particles(self) -> np.ndarray:
        """
        Return a defensive copy of the particles.
        """
        self._require_initialized()
        return self.particles.copy()

    def get_weights(self) -> np.ndarray:
        """
        Return a defensive copy of the weights.
        """
        self._require_initialized()
        return self.weights.copy()

    def _require_initialized(self) -> None:
        if not self.is_initialized:
            raise RuntimeError(
                "particle state has not been initialized."
            )