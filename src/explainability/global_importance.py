from __future__ import annotations

from typing import Any

from src.explainability.base import BaseExplainer
from src.explainability.models import FeatureImportance


class GlobalFeatureImportanceExplainer(BaseExplainer):
    """
    Process global feature importance scores into standardized,
    ranked FeatureImportance objects.
    """

    def explain(
        self,
        features: Any,
        **kwargs: Any,
    ) -> list[FeatureImportance]:
        self.validate_inputs(features)

        if not isinstance(features, dict):
            raise TypeError("features must be a dictionary.")

        validated_features: list[tuple[str, float]] = []

        for feature_name, importance in features.items():
            item = FeatureImportance(
                feature_name=feature_name,
                importance=importance,
            )

            if item.importance < self.config.importance_threshold:
                continue

            validated_features.append(
                (item.feature_name, item.importance)
            )

        validated_features.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        validated_features = validated_features[: self.config.top_k]

        importance_values = [
            importance
            for _, importance in validated_features
        ]

        normalized_values = self.normalize(importance_values)

        return [
            FeatureImportance(
                feature_name=feature_name,
                importance=importance,
                rank=index + 1,
            )
            for index, (
                (feature_name, _),
                importance,
            ) in enumerate(
                zip(validated_features, normalized_values)
            )
        ]