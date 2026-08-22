from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class BayesianModelConfig:
    """
    Shared configuration for Bayesian models.

    Supports both:
    - Bayesian linear regression
    - Bayesian regime classification using neural networks
      and Monte Carlo Dropout
    """

    # Common model configuration
    model_name: str = "bayesian_model"
    n_features: int = 1
    n_outputs: int = 1

    # Bayesian linear regression configuration
    prior_mean: float = 0.0
    prior_std: float = 1.0
    observation_noise: float = 1.0

    # Bayesian regime neural network configuration
    n_regimes: int = 3
    hidden_dims: tuple[int, ...] = (32, 16)
    dropout_rate: float = 0.2
    mc_samples: int = 20

    # Reproducibility
    random_seed: int | None = None

    def __post_init__(self) -> None:
        """Validate configuration values."""

        self._validate_model_name()
        self._validate_positive_int(
            self.n_features,
            "n_features",
        )
        self._validate_positive_int(
            self.n_outputs,
            "n_outputs",
        )

        self._validate_numeric(
            self.prior_mean,
            "prior_mean",
        )
        self._validate_positive_numeric(
            self.prior_std,
            "prior_std",
        )
        self._validate_positive_numeric(
            self.observation_noise,
            "observation_noise",
        )

        self._validate_positive_int(
            self.n_regimes,
            "n_regimes",
        )
        self._validate_hidden_dims()
        self._validate_dropout_rate()
        self._validate_positive_int(
            self.mc_samples,
            "mc_samples",
        )

        self._validate_random_seed()

    def _validate_model_name(self) -> None:
        if not isinstance(self.model_name, str):
            raise TypeError(
                "model_name must be a string."
            )

        if not self.model_name.strip():
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
    def _validate_numeric(
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

    @classmethod
    def _validate_positive_numeric(
        cls,
        value: Any,
        field_name: str,
    ) -> None:
        cls._validate_numeric(
            value,
            field_name,
        )

        if value <= 0:
            raise ValueError(
                f"{field_name} must be greater than 0."
            )

    def _validate_hidden_dims(self) -> None:
        if not isinstance(
            self.hidden_dims,
            tuple,
        ):
            raise TypeError(
                "hidden_dims must be a tuple."
            )

        if not self.hidden_dims:
            raise ValueError(
                "hidden_dims must contain at least one value."
            )

        for hidden_dim in self.hidden_dims:
            self._validate_positive_int(
                hidden_dim,
                "hidden_dims values",
            )

    def _validate_dropout_rate(self) -> None:
        self._validate_numeric(
            self.dropout_rate,
            "dropout_rate",
        )

        if not 0 <= self.dropout_rate < 1:
            raise ValueError(
                "dropout_rate must be greater than or "
                "equal to 0 and less than 1."
            )

    def _validate_random_seed(self) -> None:
        if (
            self.random_seed is not None
            and (
                isinstance(self.random_seed, bool)
                or not isinstance(
                    self.random_seed,
                    int,
                )
            )
        ):
            raise TypeError(
                "random_seed must be an integer or None."
            )

    def to_dict(self) -> dict[str, Any]:
        """
        Return the complete configuration as a dictionary.
        """
        return asdict(self)