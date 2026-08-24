from __future__ import annotations

from typing import Any

from src.explainability.base import BaseExplainer
from src.explainability.models import FeatureAttribution


class LocalFeatureAttributionExplainer(BaseExplainer):
    """
    Process local feature attribution scores into standardized,
    ranked FeatureAttribution objects.

    Features are ranked by absolute attribution magnitude while
    preserving the original positive or negative contribution sign.
    """

    def explain(
        self,
        features: Any,
        **kwargs: Any,
    ) -> list[FeatureAttribution]:
        self.validate_inputs(features)

        if not isinstance(features, dict):
            raise TypeError("features must be a dictionary.")

        validated_features: list[tuple[str, float]] = []

        for feature_name, attribution in features.items():
            item = FeatureAttribution(
                feature_name=feature_name,
                attribution=attribution,
            )

            if (
                abs(item.attribution)
                < self.config.attribution_threshold
            ):
                continue

            validated_features.append(
                (item.feature_name, item.attribution)
            )

        validated_features.sort(
            key=lambda item: abs(item[1]),
            reverse=True,
        )

        validated_features = validated_features[: self.config.top_k]

        attribution_values = [
            attribution
            for _, attribution in validated_features
        ]

        normalized_values = self.normalize(attribution_values)

        return [
            FeatureAttribution(
                feature_name=feature_name,
                attribution=attribution,
                rank=index + 1,
            )
            for index, (
                (feature_name, _),
                attribution,
            ) in enumerate(
                zip(validated_features, normalized_values)
            )
        ]