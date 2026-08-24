from __future__ import annotations

import math
from typing import Any

from src.explainability.config import ExplainabilityConfig
from src.explainability.regime_contribution import (
    RegimeContributionAnalyzer,
)
from src.explainability.regime_generator import (
    RegimeExplanationGenerator,
)
from src.explainability.regime_models import RegimeExplanation


class RegimeExplanationPipeline:
    """
    End-to-end pipeline for generating explainable
    market regime classifications.
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
        self.contribution_analyzer = (
            RegimeContributionAnalyzer(config)
        )
        self.generator = RegimeExplanationGenerator(config)

    def explain(
        self,
        regime: str,
        raw_contributions: dict[str, float],
        probability: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RegimeExplanation:
        """
        Generate a complete regime explanation.
        """

        self._validate_regime(regime)
        self._validate_probability(probability)
        self._validate_contributions(raw_contributions)
        self._validate_metadata(metadata)

        contributions = self.contribution_analyzer.analyze(
            raw_contributions
        )

        return self.generator.generate(
            regime=regime,
            probability=probability,
            contributions=contributions,
            metadata=metadata,
        )

    @staticmethod
    def _validate_regime(value: Any) -> None:
        if not isinstance(value, str):
            raise TypeError("regime must be a string.")

        if not value.strip():
            raise ValueError("regime must not be empty.")

    @staticmethod
    def _validate_probability(value: Any) -> None:
        if value is None:
            return

        if isinstance(value, bool):
            raise TypeError("probability must be numeric.")

        if not isinstance(value, (int, float)):
            raise TypeError("probability must be numeric.")

        numeric_value = float(value)

        if not math.isfinite(numeric_value):
            raise ValueError("probability must be finite.")

        if numeric_value < 0 or numeric_value > 1:
            raise ValueError(
                "probability must be between 0 and 1."
            )

    @staticmethod
    def _validate_contributions(value: Any) -> None:
        if value is None:
            raise ValueError(
                "raw_contributions must not be None."
            )

        if not isinstance(value, dict):
            raise TypeError(
                "raw_contributions must be a dictionary."
            )

    @staticmethod
    def _validate_metadata(value: Any) -> None:
        if value is None:
            return

        if not isinstance(value, dict):
            raise TypeError("metadata must be a dictionary.")