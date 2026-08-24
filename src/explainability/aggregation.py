from __future__ import annotations

from typing import Any

from src.explainability.models import (
    ExplanationResult,
    FeatureAttribution,
    FeatureImportance,
)


class ExplanationAggregator:
    """
    Combine multiple ExplanationResult objects into one
    standardized aggregated explanation.
    """

    def aggregate(
        self,
        results: Any,
    ) -> ExplanationResult:
        if results is None:
            raise ValueError("results must not be None.")

        if not isinstance(results, list):
            raise TypeError("results must be a list.")

        if not results:
            return ExplanationResult()

        for result in results:
            if not isinstance(result, ExplanationResult):
                raise TypeError(
                    "each result must be an ExplanationResult instance."
                )

        attributions: list[FeatureAttribution] = []
        importances: list[FeatureImportance] = []
        warnings: list[str] = []
        metadata: dict[str, Any] = {}

        prediction = None
        confidence = None

        for result in results:
            attributions.extend(result.attributions)
            importances.extend(result.importances)
            warnings.extend(result.warnings)
            metadata.update(result.metadata)

            if result.prediction is not None:
                prediction = result.prediction

            if result.confidence is not None:
                confidence = result.confidence

        return ExplanationResult(
            attributions=attributions,
            importances=importances,
            prediction=prediction,
            confidence=confidence,
            warnings=warnings,
            metadata=metadata,
        )