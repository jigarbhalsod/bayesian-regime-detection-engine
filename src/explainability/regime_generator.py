from __future__ import annotations

from typing import Any

from src.explainability.config import ExplainabilityConfig
from src.explainability.regime_models import (
    RegimeContribution,
    RegimeExplanation,
)


class RegimeExplanationGenerator:
    """
    Builds structured, human-readable explanations for
    market regime predictions.
    """

    def __init__(
        self,
        config: ExplainabilityConfig | None = None,
    ) -> None:
        if config is None:
            config = ExplainabilityConfig()

        if not isinstance(config, ExplainabilityConfig):
            raise TypeError(
                "config must be an ExplainabilityConfig instance."
            )

        self.config = config

    def generate(
        self,
        regime: str,
        probability: float | None = None,
        contributions: list[RegimeContribution] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RegimeExplanation:
        """
        Generate a regime explanation from prediction details.
        """

        normalized_regime = self._validate_regime(regime)
        validated_probability = self._validate_probability(
            probability
        )
        validated_contributions = (
            self._validate_contributions(contributions)
        )
        validated_metadata = self._validate_metadata(metadata)

        key_drivers = [
            item.feature_name
            for item in validated_contributions
        ]

        summary = self._build_summary(
            regime=normalized_regime,
            probability=validated_probability,
            key_drivers=key_drivers,
        )

        return RegimeExplanation(
            regime=normalized_regime,
            probability=validated_probability,
            contributions=validated_contributions,
            key_drivers=key_drivers,
            summary=summary,
            metadata=validated_metadata,
        )

    @staticmethod
    def _validate_regime(value: Any) -> str:
        if not isinstance(value, str):
            raise TypeError("regime must be a string.")

        value = value.strip()

        if not value:
            raise ValueError("regime must not be empty.")

        return value

    @staticmethod
    def _validate_probability(value: Any) -> float | None:
        if value is None:
            return None

        if isinstance(value, bool):
            raise TypeError("probability must be numeric.")

        if not isinstance(value, (int, float)):
            raise TypeError("probability must be numeric.")

        value = float(value)

        if value < 0 or value > 1:
            raise ValueError(
                "probability must be between 0 and 1."
            )

        return value

    @staticmethod
    def _validate_contributions(
        value: Any,
    ) -> list[RegimeContribution]:
        if value is None:
            return []

        if not isinstance(value, list):
            raise TypeError("contributions must be a list.")

        for item in value:
            if not isinstance(item, RegimeContribution):
                raise TypeError(
                    "each contribution must be a "
                    "RegimeContribution instance."
                )

        return value

    @staticmethod
    def _validate_metadata(value: Any) -> dict[str, Any]:
        if value is None:
            return {}

        if not isinstance(value, dict):
            raise TypeError("metadata must be a dictionary.")

        return value

    @staticmethod
    def _build_summary(
        regime: str,
        probability: float | None,
        key_drivers: list[str],
    ) -> str:
        if probability is None:
            probability_text = "with no probability available"
        else:
            probability_text = (
                f"with probability {probability:.2f}"
            )

        if not key_drivers:
            return (
                f"The market is classified as {regime} "
                f"{probability_text}."
            )

        if len(key_drivers) == 1:
            drivers_text = key_drivers[0]
        else:
            drivers_text = ", ".join(key_drivers[:-1])
            drivers_text += f" and {key_drivers[-1]}"

        return (
            f"The market is classified as {regime} "
            f"{probability_text}, driven by {drivers_text}."
        )