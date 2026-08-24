from __future__ import annotations

from typing import Any

from src.explainability.base import BaseExplainer
from src.explainability.models import (
    ExplanationResult,
    FeatureAttribution,
    FeatureImportance,
)


class PredictionExplainer(BaseExplainer):
    """
    Build a standardized explanation for a single model prediction.
    """

    def explain(
        self,
        features: Any,
        **kwargs: Any,
    ) -> ExplanationResult:
        self.validate_inputs(features)

        if not isinstance(features, list):
            raise TypeError("features must be a list of FeatureAttribution objects.")

        for feature in features:
            if not isinstance(feature, FeatureAttribution):
                raise TypeError(
                    "each feature must be a FeatureAttribution instance."
                )

        prediction = kwargs.get("prediction")

        if prediction is None:
            raise ValueError("prediction must not be None.")

        confidence = kwargs.get("confidence")

        importances = kwargs.get("importances", [])

        if not isinstance(importances, list):
            raise TypeError("importances must be a list.")

        for importance in importances:
            if not isinstance(importance, FeatureImportance):
                raise TypeError(
                    "each importance must be a FeatureImportance instance."
                )

        warnings = kwargs.get("warnings", [])
        metadata = kwargs.get("metadata", {})

        return self.build_result(
            attributions=list(features),
            importances=list(importances),
            prediction=prediction,
            confidence=confidence,
            warnings=warnings,
            metadata=metadata,
        )