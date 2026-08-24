from __future__ import annotations

import math
from typing import Any

from src.explainability.config import ExplainabilityConfig
from src.explainability.regime_models import RegimeContribution


class RegimeContributionAnalyzer:
    """
    Converts raw feature contribution scores into ranked
    RegimeContribution objects.
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

    def analyze(
        self,
        contributions: dict[str, float],
    ) -> list[RegimeContribution]:
        """
        Validate, filter, sort, and rank regime contributions.
        """

        if contributions is None:
            raise ValueError("contributions must not be None.")

        if not isinstance(contributions, dict):
            raise TypeError("contributions must be a dictionary.")

        validated: list[tuple[str, float, int]] = []

        for index, (feature_name, contribution) in enumerate(
            contributions.items()
        ):
            normalized_name = self._validate_feature_name(feature_name)
            numeric_value = self._validate_contribution(contribution)

            if (
                abs(numeric_value)
                < self.config.attribution_threshold
            ):
                continue

            validated.append(
                (
                    normalized_name,
                    numeric_value,
                    index,
                )
            )

        validated.sort(
            key=lambda item: (-abs(item[1]), item[2])
        )

        if self.config.top_k is not None:
            validated = validated[: self.config.top_k]

        return [
            RegimeContribution(
                feature_name=feature_name,
                contribution=contribution,
                rank=index + 1,
            )
            for index, (
                feature_name,
                contribution,
                _,
            ) in enumerate(validated)
        ]

    @staticmethod
    def _validate_feature_name(value: Any) -> str:
        if not isinstance(value, str):
            raise TypeError("feature names must be strings.")

        value = value.strip()

        if not value:
            raise ValueError("feature names must not be empty.")

        return value

    @staticmethod
    def _validate_contribution(value: Any) -> float:
        if isinstance(value, bool):
            raise TypeError("contribution values must be numeric.")

        if not isinstance(value, (int, float)):
            raise TypeError("contribution values must be numeric.")

        value = float(value)

        if not math.isfinite(value):
            raise ValueError(
                "contribution values must be finite."
            )

        return value