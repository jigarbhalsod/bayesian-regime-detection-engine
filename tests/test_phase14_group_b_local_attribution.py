import math

import pytest

from src.explainability.config import ExplainabilityConfig
from src.explainability.local_attribution import (
    LocalFeatureAttributionExplainer,
)
from src.explainability.models import FeatureAttribution


class TestLocalFeatureAttributionExplainer:
    def test_returns_ranked_feature_attribution_objects(self):
        explainer = LocalFeatureAttributionExplainer(
            ExplainabilityConfig(normalize=False)
        )

        result = explainer.explain(
            {
                "momentum": -0.4,
                "volatility": 0.8,
                "volume": -0.6,
            }
        )

        assert len(result) == 3
        assert all(
            isinstance(item, FeatureAttribution)
            for item in result
        )

        assert [item.feature_name for item in result] == [
            "volatility",
            "volume",
            "momentum",
        ]

        assert [item.attribution for item in result] == [
            0.8,
            -0.6,
            -0.4,
        ]

        assert [item.rank for item in result] == [1, 2, 3]

    def test_preserves_positive_and_negative_signs(self):
        explainer = LocalFeatureAttributionExplainer(
            ExplainabilityConfig(normalize=False)
        )

        result = explainer.explain(
            {
                "positive": 0.7,
                "negative": -0.5,
            }
        )

        assert result[0].attribution == 0.7
        assert result[1].attribution == -0.5

    def test_applies_attribution_threshold_by_absolute_value(self):
        config = ExplainabilityConfig(
            attribution_threshold=0.5,
            normalize=False,
        )
        explainer = LocalFeatureAttributionExplainer(config)

        result = explainer.explain(
            {
                "feature_a": 0.2,
                "feature_b": -0.5,
                "feature_c": 0.8,
                "feature_d": -0.3,
            }
        )

        assert [item.feature_name for item in result] == [
            "feature_c",
            "feature_b",
        ]

    def test_applies_top_k_by_absolute_magnitude(self):
        config = ExplainabilityConfig(
            top_k=2,
            normalize=False,
        )
        explainer = LocalFeatureAttributionExplainer(config)

        result = explainer.explain(
            {
                "feature_a": 0.1,
                "feature_b": -0.9,
                "feature_c": 0.5,
                "feature_d": -0.7,
            }
        )

        assert len(result) == 2
        assert [item.feature_name for item in result] == [
            "feature_b",
            "feature_d",
        ]

        assert [item.attribution for item in result] == [
            -0.9,
            -0.7,
        ]

        assert [item.rank for item in result] == [1, 2]

    def test_normalizes_using_absolute_total_and_preserves_sign(self):
        explainer = LocalFeatureAttributionExplainer()

        result = explainer.explain(
            {
                "feature_a": 2.0,
                "feature_b": -3.0,
                "feature_c": 5.0,
            }
        )

        assert [
            item.attribution
            for item in result
        ] == pytest.approx(
            [0.5, -0.3, 0.2]
        )

    def test_normalization_can_be_disabled(self):
        config = ExplainabilityConfig(normalize=False)
        explainer = LocalFeatureAttributionExplainer(config)

        result = explainer.explain(
            {
                "feature_a": 2.0,
                "feature_b": -3.0,
            }
        )

        assert [
            item.attribution
            for item in result
        ] == [-3.0, 2.0]

    def test_equal_absolute_attributions_preserve_input_order(self):
        config = ExplainabilityConfig(normalize=False)
        explainer = LocalFeatureAttributionExplainer(config)

        result = explainer.explain(
            {
                "feature_a": 0.5,
                "feature_b": -0.5,
                "feature_c": 0.5,
            }
        )

        assert [item.feature_name for item in result] == [
            "feature_a",
            "feature_b",
            "feature_c",
        ]

    def test_zero_attributions_are_allowed(self):
        config = ExplainabilityConfig(normalize=False)
        explainer = LocalFeatureAttributionExplainer(config)

        result = explainer.explain(
            {
                "feature_a": 0.0,
                "feature_b": 0.0,
            }
        )

        assert [
            item.attribution
            for item in result
        ] == [0.0, 0.0]

    def test_all_features_filtered_by_threshold(self):
        config = ExplainabilityConfig(
            attribution_threshold=1.0,
        )
        explainer = LocalFeatureAttributionExplainer(config)

        result = explainer.explain(
            {
                "feature_a": 0.2,
                "feature_b": -0.5,
            }
        )

        assert result == []

    def test_empty_dictionary_returns_empty_list(self):
        explainer = LocalFeatureAttributionExplainer()

        assert explainer.explain({}) == []

    def test_none_features_rejected(self):
        explainer = LocalFeatureAttributionExplainer()

        with pytest.raises(
            ValueError,
            match="features must not be None",
        ):
            explainer.explain(None)

    @pytest.mark.parametrize(
        "features",
        [
            [],
            (),
            "invalid",
            123,
        ],
    )
    def test_non_dictionary_features_rejected(self, features):
        explainer = LocalFeatureAttributionExplainer()

        with pytest.raises(
            TypeError,
            match="features must be a dictionary",
        ):
            explainer.explain(features)

    @pytest.mark.parametrize(
        "attribution",
        [
            math.inf,
            -math.inf,
            math.nan,
            "invalid",
            True,
            None,
        ],
    )
    def test_invalid_attribution_rejected(self, attribution):
        explainer = LocalFeatureAttributionExplainer()

        with pytest.raises((TypeError, ValueError)):
            explainer.explain(
                {"feature_a": attribution}
            )

    def test_empty_feature_name_rejected(self):
        explainer = LocalFeatureAttributionExplainer()

        with pytest.raises(
            ValueError,
            match="feature_name must not be empty",
        ):
            explainer.explain(
                {"   ": 0.5}
            )

    def test_feature_name_is_normalized(self):
        config = ExplainabilityConfig(normalize=False)
        explainer = LocalFeatureAttributionExplainer(config)

        result = explainer.explain(
            {"  volatility  ": -0.8}
        )

        assert result[0].feature_name == "volatility"