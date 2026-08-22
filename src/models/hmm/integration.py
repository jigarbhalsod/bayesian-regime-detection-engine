from __future__ import annotations

from typing import Any, Dict, Type

from ..result import AdvancedModelResult
from .config import HMMConfig
from .gmm import GMMHMMRegimeModel
from .model import GaussianHMMRegimeModel


class HMMRegimeIntegration:
    """
    Unified integration layer for HMM-based regime models.

    Supports:
    - Gaussian HMM
    - GMM-HMM

    Provides a single interface for model creation, fitting,
    prediction, scoring, diagnostics, and model metadata.
    """

    _MODEL_TYPES: Dict[str, Type[Any]] = {
        "gaussian_hmm": GaussianHMMRegimeModel,
        "gmm_hmm": GMMHMMRegimeModel,
    }

    def __init__(
        self,
        config: HMMConfig | None = None,
    ) -> None:
        """
        Initialize the integration layer and create the configured
        underlying HMM model.
        """
        if config is None:
            config = HMMConfig()

        if not isinstance(config, HMMConfig):
            raise TypeError(
                "config must be an HMMConfig instance."
            )

        self.config = config

        # Normalize the configured name so values such as
        # " GAUSSIAN_HMM " and "gMm_HmM" resolve correctly.
        self._model_type = self._normalize_model_type(
            config.model_name
        )

        self._model = self._create_model(config)

    @property
    def model(self) -> Any:
        """
        Return the underlying regime model implementation.
        """
        return self._model

    @property
    def model_type(self) -> str:
        """
        Return the normalized HMM model type.
        """
        return self._model_type

    @property
    def is_fitted(self) -> bool:
        """
        Return whether the underlying model has been fitted.
        """
        return bool(self._model.is_fitted)

    @classmethod
    def supported_model_types(cls) -> list[str]:
        """
        Return all supported HMM model identifiers.
        """
        return sorted(
            cls._MODEL_TYPES.keys()
        )

    @classmethod
    def _normalize_model_type(
        cls,
        model_type: Any,
    ) -> str:
        """
        Validate and normalize a supported model identifier.

        Normalization:
        - Removes surrounding whitespace
        - Converts to lowercase

        Supported values:
        - gaussian_hmm
        - gmm_hmm
        """
        if not isinstance(model_type, str):
            raise TypeError(
                "model_name must be a string."
            )

        normalized = model_type.strip().lower()

        if not normalized:
            raise ValueError(
                "model_name cannot be empty."
            )

        if normalized not in cls._MODEL_TYPES:
            supported = ", ".join(
                cls.supported_model_types()
            )

            raise ValueError(
                f"Unsupported model_name '{normalized}'. "
                f"Supported values are: {supported}."
            )

        return normalized

    @classmethod
    def _create_model(
        cls,
        config: HMMConfig,
    ) -> Any:
        """
        Create the correct HMM model implementation based on
        the configured model name.
        """
        model_type = cls._normalize_model_type(
            config.model_name
        )

        model_class = cls._MODEL_TYPES[
            model_type
        ]

        return model_class(config)

    def fit(
        self,
        data: Any,
    ) -> "HMMRegimeIntegration":
        """
        Fit the configured HMM model.

        Returns
        -------
        HMMRegimeIntegration
            The integration instance for method chaining.
        """
        self._model.fit(data)

        return self

    def predict(
        self,
        data: Any,
    ) -> AdvancedModelResult:
        """
        Predict market regimes using the configured model.
        """
        return self._model.predict(data)

    def predict_proba(
        self,
        data: Any,
    ) -> list[list[float]]:
        """
        Return posterior regime probabilities.
        """
        return self._model.predict_proba(data)

    def score(
        self,
        data: Any,
    ) -> float:
        """
        Return the sequence log likelihood.
        """
        return float(
            self._model.score(data)
        )

    def get_transition_matrix(
        self,
    ) -> list[list[float]]:
        """
        Return the learned regime transition matrix.
        """
        return self._model.get_transition_matrix()

    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Return diagnostics from the underlying HMM model.

        Raises
        ------
        RuntimeError
            If the underlying model has not been fitted.
        """
        return self._model.get_diagnostics()

    def get_model_metadata(self) -> Dict[str, Any]:
        """
        Return integration and underlying model metadata.
        """
        metadata = {
            "model_type": self.model_type,
            "model_name": self.config.model_name,
            "is_fitted": self.is_fitted,
            "n_regimes": self.config.n_regimes,
            "n_mix": self.config.n_mix,
            "covariance_type": self.config.covariance_type,
            "algorithm": self.config.algorithm,
        }

        if hasattr(self._model, "n_features"):
            metadata["n_features"] = (
                self._model.n_features
            )

        return metadata