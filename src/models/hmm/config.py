from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from ..config import AdvancedModelConfig


@dataclass
class HMMConfig(AdvancedModelConfig):
    """
    Configuration for Hidden Markov Model based regime models.

    Supports:
    - Gaussian HMM
    - Gaussian Mixture Model HMM (GMM-HMM)
    """

    # Override the inherited default for HMM models.
    model_name: str = "gaussian_hmm"

    covariance_type: str = "diag"
    n_iter: int = 100
    tol: float = 1e-3
    min_covar: float = 1e-3
    init_params: str = "stmc"
    params: str = "stmc"
    verbose: bool = False

    # GMM-HMM specific configuration.
    n_mix: int = 2
    algorithm: str = "viterbi"

    def __post_init__(self) -> None:
        """
        Validate and normalize HMM configuration values.
        """
        super().__post_init__()

        self.covariance_type = self._validate_covariance_type(
            self.covariance_type
        )

        self.n_iter = self._validate_n_iter(
            self.n_iter
        )

        self.tol = self._validate_positive_float(
            self.tol,
            "tol",
        )

        self.min_covar = self._validate_non_negative_float(
            self.min_covar,
            "min_covar",
        )

        self.init_params = self._validate_parameter_string(
            self.init_params,
            "init_params",
        )

        self.params = self._validate_parameter_string(
            self.params,
            "params",
        )

        self.verbose = self._validate_verbose(
            self.verbose
        )

        self.n_mix = self._validate_n_mix(
            self.n_mix
        )

        self.algorithm = self._validate_algorithm(
            self.algorithm
        )

    @staticmethod
    def _validate_covariance_type(
        value: Any,
    ) -> str:
        """
        Validate HMM covariance type.
        """
        if not isinstance(value, str):
            raise TypeError(
                "covariance_type must be a string."
            )

        value = value.strip().lower()

        allowed_types = {
            "spherical",
            "diag",
            "full",
            "tied",
        }

        if value not in allowed_types:
            raise ValueError(
                "covariance_type must be one of: "
                "'spherical', 'diag', 'full', or 'tied'."
            )

        return value

    @staticmethod
    def _validate_n_iter(
        value: Any,
    ) -> int:
        """
        Validate maximum number of EM iterations.
        """
        if isinstance(value, bool) or not isinstance(
            value,
            int,
        ):
            raise TypeError(
                "n_iter must be an integer."
            )

        if value < 1:
            raise ValueError(
                "n_iter must be greater than or equal to 1."
            )

        return value

    @staticmethod
    def _validate_positive_float(
        value: Any,
        field_name: str,
    ) -> float:
        """
        Validate a strictly positive numeric value.
        """
        if isinstance(value, bool) or not isinstance(
            value,
            (int, float),
        ):
            raise TypeError(
                f"{field_name} must be a number."
            )

        value = float(value)

        if value <= 0:
            raise ValueError(
                f"{field_name} must be greater than 0."
            )

        return value

    @staticmethod
    def _validate_non_negative_float(
        value: Any,
        field_name: str,
    ) -> float:
        """
        Validate a numeric value greater than or equal to zero.
        """
        if isinstance(value, bool) or not isinstance(
            value,
            (int, float),
        ):
            raise TypeError(
                f"{field_name} must be a number."
            )

        value = float(value)

        if value < 0:
            raise ValueError(
                f"{field_name} must be greater than or equal to 0."
            )

        return value

    @staticmethod
    def _validate_parameter_string(
        value: Any,
        field_name: str,
    ) -> str:
        """
        Validate hmmlearn parameter control strings.

        The existing Gaussian HMM configuration accepts:
        - s: start probabilities
        - t: transition matrix
        - m: means
        - c: covariances

        Each parameter code may appear at most once.
        """
        if not isinstance(value, str):
            raise TypeError(
                f"{field_name} must be a string."
            )

        value = value.strip().lower()

        if not value:
            raise ValueError(
                f"{field_name} cannot be empty."
            )

        allowed_characters = {
            "s",
            "t",
            "m",
            "c",
        }

        invalid_characters = (
            set(value) - allowed_characters
        )

        if invalid_characters:
            invalid = ", ".join(
                sorted(invalid_characters)
            )

            raise ValueError(
                f"{field_name} contains invalid parameter "
                f"characters: {invalid}."
            )

        if len(set(value)) != len(value):
            raise ValueError(
                f"{field_name} cannot contain duplicate "
                "parameter characters."
            )

        return value

    @staticmethod
    def _validate_verbose(
        value: Any,
    ) -> bool:
        """
        Validate verbose configuration.
        """
        if not isinstance(value, bool):
            raise TypeError(
                "verbose must be a boolean."
            )

        return value

    @staticmethod
    def _validate_n_mix(
        value: Any,
    ) -> int:
        """
        Validate the number of Gaussian mixture components
        per hidden regime.
        """
        if isinstance(value, bool) or not isinstance(
            value,
            int,
        ):
            raise TypeError(
                "n_mix must be an integer."
            )

        if value < 1:
            raise ValueError(
                "n_mix must be greater than or equal to 1."
            )

        return value

    @staticmethod
    def _validate_algorithm(
        value: Any,
    ) -> str:
        """
        Validate the decoding algorithm.
        """
        if not isinstance(value, str):
            raise TypeError(
                "algorithm must be a string."
            )

        value = value.strip().lower()

        allowed_algorithms = {
            "viterbi",
            "map",
        }

        if value not in allowed_algorithms:
            raise ValueError(
                "algorithm must be either "
                "'viterbi' or 'map'."
            )

        return value

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the configuration into a dictionary.
        """
        data = super().to_dict()

        data.update(
            {
                "covariance_type": self.covariance_type,
                "n_iter": self.n_iter,
                "tol": self.tol,
                "min_covar": self.min_covar,
                "init_params": self.init_params,
                "params": self.params,
                "verbose": self.verbose,
                "n_mix": self.n_mix,
                "algorithm": self.algorithm,
            }
        )

        return data