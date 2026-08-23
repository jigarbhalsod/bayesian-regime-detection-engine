from __future__ import annotations

from typing import Literal

import numpy as np

from src.models.particle_filter.config import ParticleFilterConfig


ResamplingMethod = Literal[
    "multinomial",
    "systematic",
    "stratified",
]


class ParticleResampler:
    """
    Resampling engine for particle filter state indices.

    Supports multinomial, systematic, and stratified resampling.
    """

    SUPPORTED_METHODS = (
        "multinomial",
        "systematic",
        "stratified",
    )

    def __init__(
        self,
        config: ParticleFilterConfig,
        method: ResamplingMethod = "systematic",
    ) -> None:
        if not isinstance(config, ParticleFilterConfig):
            raise TypeError(
                "config must be a ParticleFilterConfig."
            )

        if method not in self.SUPPORTED_METHODS:
            raise ValueError(
                "method must be one of: "
                "multinomial, systematic, stratified."
            )

        self.config = config
        self.method = method

        self._rng = np.random.default_rng(
            config.random_seed
        )

    @property
    def n_particles(self) -> int:
        """Return the configured number of particles."""
        return self.config.n_particles

    def validate_weights(
        self,
        weights: np.ndarray,
    ) -> np.ndarray:
        """
        Validate and normalize particle weights.
        """
        if not isinstance(weights, np.ndarray):
            raise TypeError(
                "weights must be a numpy array."
            )

        if weights.ndim != 1:
            raise ValueError(
                "weights must be a 1-dimensional array."
            )

        if weights.shape[0] != self.n_particles:
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

    def effective_sample_size(
        self,
        weights: np.ndarray,
    ) -> float:
        """
        Calculate the effective sample size.

        ESS = 1 / sum(weights ** 2)
        """
        normalized = self.validate_weights(weights)

        return float(
            1.0 / np.sum(np.square(normalized))
        )

    def should_resample(
        self,
        weights: np.ndarray,
        threshold: float | None = None,
    ) -> bool:
        """
        Determine whether resampling should occur.

        If threshold is None, the configuration's
        resample_threshold is used.
        """
        if threshold is None:
            threshold = self.config.resample_threshold

        if isinstance(threshold, bool) or not isinstance(
            threshold,
            (int, float),
        ):
            raise TypeError(
                "threshold must be numeric."
            )

        threshold = float(threshold)

        if not np.isfinite(threshold):
            raise ValueError(
                "threshold must be finite."
            )

        if threshold <= 0:
            raise ValueError(
                "threshold must be greater than 0."
            )

        ess = self.effective_sample_size(weights)

        return ess <= threshold * self.n_particles

    def resample(
        self,
        weights: np.ndarray,
        method: ResamplingMethod | None = None,
    ) -> np.ndarray:
        """
        Resample particle indices according to weights.
        """
        normalized = self.validate_weights(weights)

        selected_method = (
            self.method if method is None else method
        )

        if selected_method not in self.SUPPORTED_METHODS:
            raise ValueError(
                "method must be one of: "
                "multinomial, systematic, stratified."
            )

        if selected_method == "multinomial":
            indices = self.multinomial(normalized)

        elif selected_method == "systematic":
            indices = self.systematic(normalized)

        else:
            indices = self.stratified(normalized)

        return indices

    def multinomial(
        self,
        weights: np.ndarray,
    ) -> np.ndarray:
        """
        Perform multinomial resampling.
        """
        normalized = self.validate_weights(weights)

        return self._rng.choice(
            self.n_particles,
            size=self.n_particles,
            replace=True,
            p=normalized,
        ).astype(np.int64)

    def systematic(
        self,
        weights: np.ndarray,
    ) -> np.ndarray:
        """
        Perform systematic resampling.
        """
        normalized = self.validate_weights(weights)

        positions = (
            self._rng.random() + np.arange(
                self.n_particles
            )
        ) / self.n_particles

        cumulative = np.cumsum(normalized)
        cumulative[-1] = 1.0

        indices = np.searchsorted(
            cumulative,
            positions,
            side="right",
        )

        return np.clip(
            indices,
            0,
            self.n_particles - 1,
        ).astype(np.int64)

    def stratified(
        self,
        weights: np.ndarray,
    ) -> np.ndarray:
        """
        Perform stratified resampling.
        """
        normalized = self.validate_weights(weights)

        positions = (
            np.arange(self.n_particles)
            + self._rng.random(self.n_particles)
        ) / self.n_particles

        cumulative = np.cumsum(normalized)
        cumulative[-1] = 1.0

        indices = np.searchsorted(
            cumulative,
            positions,
            side="right",
        )

        return np.clip(
            indices,
            0,
            self.n_particles - 1,
        ).astype(np.int64)