from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.explainability.config import ExplainabilityConfig
from src.explainability.models import ExplanationResult


class BaseExplainer(ABC):
    """
    Abstract base class for explainability components.

    All explainers must validate their inputs and return a standardized
    ExplanationResult.
    """

    def __init__(
        self,
        config: ExplainabilityConfig | None = None,
    ) -> None:
        if config is None:
            config = ExplainabilityConfig()

        if not isinstance(config, ExplainabilityConfig):
            raise TypeError(
                "config must be an ExplainabilityConfig instance or None."
            )

        self.config = config

    def validate_inputs(
        self,
        features: Any,
        **kwargs: Any,
    ) -> None:
        """
        Validate explainer inputs.

        Subclasses may override this method when they require additional
        validation.
        """
        if features is None:
            raise ValueError("features must not be None.")

    @abstractmethod
    def explain(
        self,
        features: Any,
        **kwargs: Any,
    ) -> ExplanationResult:
        """
        Generate an explanation for the provided features.
        """

    def normalize(
        self,
        values: list[float],
    ) -> list[float]:
        """
        Normalize numeric values by their absolute-value total.

        When normalization is disabled, values are returned unchanged.
        When all values are zero, a zero-filled result is returned.
        """
        if not isinstance(values, list):
            raise TypeError("values must be a list.")

        validated_values: list[float] = []

        for value in values:
            if isinstance(value, bool) or not isinstance(
                value,
                (int, float),
            ):
                raise TypeError(
                    "each value must be a numeric value."
                )

            validated_values.append(float(value))

        if not self.config.normalize:
            return validated_values

        total = sum(abs(value) for value in validated_values)

        if total == 0:
            return [0.0 for _ in validated_values]

        return [
            value / total
            for value in validated_values
        ]

    def build_result(
        self,
        **kwargs: Any,
    ) -> ExplanationResult:
        """
        Build a standardized ExplanationResult.
        """
        return ExplanationResult(**kwargs)