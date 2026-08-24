from __future__ import annotations

from typing import Any

from src.explainability.models import ExplanationResult
from src.explainability.regime_models import RegimeExplanation


class ExplanationSummaryEngine:
    """
    Builds a unified structured summary from explainability outputs.
    """

    def summarize(
        self,
        explanations: list[ExplanationResult | RegimeExplanation],
    ) -> dict[str, Any]:
        """
        Summarize prediction and regime explanation results.
        """

        self._validate_explanations(explanations)

        if not explanations:
            return {
                "predictions": [],
                "confidences": [],
                "feature_drivers": [],
                "regimes": [],
                "metadata": {},
            }

        predictions: list[Any] = []
        confidences: list[float] = []
        feature_drivers: list[str] = []
        regimes: list[str] = []
        metadata: dict[str, Any] = {}

        for explanation in explanations:
            if isinstance(explanation, ExplanationResult):
                if explanation.prediction is not None:
                    predictions.append(explanation.prediction)

                if explanation.confidence is not None:
                    confidences.append(explanation.confidence)

                for attribution in explanation.attributions:
                    feature_drivers.append(attribution.feature_name)

                for importance in explanation.importances:
                    feature_drivers.append(importance.feature_name)

                metadata.update(explanation.metadata)

            elif isinstance(explanation, RegimeExplanation):
                regimes.append(explanation.regime)

                if explanation.probability is not None:
                    confidences.append(explanation.probability)

                feature_drivers.extend(explanation.key_drivers)

                metadata.update(explanation.metadata)

        return {
            "predictions": predictions,
            "confidences": confidences,
            "feature_drivers": feature_drivers,
            "regimes": regimes,
            "metadata": metadata,
        }

    @staticmethod
    def _validate_explanations(value: Any) -> None:
        if value is None:
            raise ValueError("explanations must not be None.")

        if not isinstance(value, list):
            raise TypeError("explanations must be a list.")

        for explanation in value:
            if not isinstance(
                explanation,
                (ExplanationResult, RegimeExplanation),
            ):
                raise TypeError(
                    "Each explanation must be an ExplanationResult "
                    "or RegimeExplanation."
                )