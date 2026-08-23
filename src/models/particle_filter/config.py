from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


_UNSET = object()


@dataclass(frozen=True, init=False)
class ParticleFilterConfig:
    """
    Configuration for Sequential Monte Carlo / Particle Filter models.

    `resampling_threshold` is the canonical configuration name.

    `resample_threshold` is accepted as a backward-compatible alias
    because earlier resampling components use that parameter name.
    """

    model_name: str = "particle_filter"

    n_particles: int = 100
    state_dim: int = 1
    observation_dim: int = 1

    process_noise: float = 1.0
    observation_noise: float = 1.0

    resampling_threshold: float = 0.5

    random_seed: int | None = None

    def __init__(
        self,
        model_name: str = "particle_filter",
        n_particles: int = 100,
        state_dim: int = 1,
        observation_dim: int = 1,
        process_noise: float = 1.0,
        observation_noise: float = 1.0,
        resampling_threshold: Any = _UNSET,
        random_seed: int | None = None,
        *,
        resample_threshold: Any = _UNSET,
    ) -> None:
        """
        Create and validate the particle filter configuration.

        Both `resampling_threshold` and `resample_threshold` are
        supported. Supplying both with different values is rejected.
        """
        threshold = self._resolve_threshold(
            resampling_threshold=resampling_threshold,
            resample_threshold=resample_threshold,
        )

        self._validate_model_name(
            model_name
        )

        self._validate_positive_int(
            n_particles,
            "n_particles",
        )

        self._validate_positive_int(
            state_dim,
            "state_dim",
        )

        self._validate_positive_int(
            observation_dim,
            "observation_dim",
        )

        self._validate_positive_number(
            process_noise,
            "process_noise",
        )

        self._validate_positive_number(
            observation_noise,
            "observation_noise",
        )

        self._validate_resampling_threshold(
            threshold
        )

        self._validate_random_seed(
            random_seed
        )

        object.__setattr__(
            self,
            "model_name",
            model_name,
        )

        object.__setattr__(
            self,
            "n_particles",
            n_particles,
        )

        object.__setattr__(
            self,
            "state_dim",
            state_dim,
        )

        object.__setattr__(
            self,
            "observation_dim",
            observation_dim,
        )

        object.__setattr__(
            self,
            "process_noise",
            float(process_noise),
        )

        object.__setattr__(
            self,
            "observation_noise",
            float(observation_noise),
        )

        object.__setattr__(
            self,
            "resampling_threshold",
            float(threshold),
        )

        object.__setattr__(
            self,
            "random_seed",
            random_seed,
        )

    @staticmethod
    def _resolve_threshold(
        resampling_threshold: Any,
        resample_threshold: Any,
    ) -> Any:
        """
        Resolve the canonical threshold and its compatibility alias.
        """
        canonical_provided = (
            resampling_threshold is not _UNSET
        )

        alias_provided = (
            resample_threshold is not _UNSET
        )

        if canonical_provided and alias_provided:
            if (
                resampling_threshold
                != resample_threshold
            ):
                raise ValueError(
                    "resampling_threshold and "
                    "resample_threshold must match when both "
                    "are provided."
                )

            return resampling_threshold

        if canonical_provided:
            return resampling_threshold

        if alias_provided:
            return resample_threshold

        return 0.5

    @staticmethod
    def _validate_model_name(
        value: Any,
    ) -> None:
        if not isinstance(value, str):
            raise TypeError(
                "model_name must be a string."
            )

        if not value.strip():
            raise ValueError(
                "model_name must not be empty."
            )

    @staticmethod
    def _validate_positive_int(
        value: Any,
        field_name: str,
    ) -> None:
        if isinstance(value, bool) or not isinstance(
            value,
            int,
        ):
            raise TypeError(
                f"{field_name} must be an integer."
            )

        if value <= 0:
            raise ValueError(
                f"{field_name} must be greater than 0."
            )

    @staticmethod
    def _validate_positive_number(
        value: Any,
        field_name: str,
    ) -> None:
        if isinstance(value, bool) or not isinstance(
            value,
            (int, float),
        ):
            raise TypeError(
                f"{field_name} must be numeric."
            )

        value = float(value)

        if not math.isfinite(value):
            raise ValueError(
                f"{field_name} must be finite."
            )

        if value <= 0:
            raise ValueError(
                f"{field_name} must be greater than 0."
            )

    @staticmethod
    def _validate_resampling_threshold(
        value: Any,
    ) -> None:
        if isinstance(value, bool) or not isinstance(
            value,
            (int, float),
        ):
            raise TypeError(
                "resampling_threshold must be numeric."
            )

        value = float(value)

        if not math.isfinite(value):
            raise ValueError(
                "resampling_threshold must be finite."
            )

        if value <= 0 or value > 1:
            raise ValueError(
                "resampling_threshold must be greater than 0 "
                "and less than or equal to 1."
            )

    @staticmethod
    def _validate_random_seed(
        value: Any,
    ) -> None:
        if value is None:
            return

        if isinstance(value, bool) or not isinstance(
            value,
            int,
        ):
            raise TypeError(
                "random_seed must be an integer or None."
            )

    @property
    def resample_threshold(self) -> float:
        """
        Backward-compatible alias for resampling_threshold.
        """
        return self.resampling_threshold

    def to_dict(self) -> dict[str, Any]:
        """
        Return the canonical configuration as a dictionary.
        """
        return asdict(self)