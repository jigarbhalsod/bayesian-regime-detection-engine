import math

import pytest

from src.explainability.config import ExplainabilityConfig
from src.explainability.global_importance import (
    GlobalFeatureImportanceExplainer,
)
from src.explainability.models import FeatureImportance


class TestGlobalFeatureImportanceExplainer:
    def test_returns_ranked_feature_importance_objects(self):
        explainer = GlobalFeatureImportanceExplainer(
            ExplainabilityConfig(normalize=False)
        )

        result = explainer.explain(
            {
                "momentum": 0.4,
                "volatility": 0.8,
                "volume": 0.6,
            }
        )

        assert len(result) == 3
        assert all(
            isinstance(item, FeatureImportance)
            for item in result
        )

        assert [item.feature_name for item in result] == [
            "volatility",
            "volume",
            "momentum",
        ]

        assert [item.importance for item in result] == [
            0.8,
            0.6,
            0.4,
        ]

        assert [item.rank for item in result] == [1, 2, 3]

    def test_applies_importance_threshold(self):
        config = ExplainabilityConfig(
            importance_threshold=0.5,
            normalize=False,
        )
        explainer = GlobalFeatureImportanceExplainer(config)

        result = explainer.explain(
            {
                "feature_a": 0.2,
                "feature_b": 0.5,
                "feature_c": 0.8,
            }
        )

        assert [item.feature_name for item in result] == [
            "feature_c",
            "feature_b",
        ]

    def test_applies_top_k(self):
        config = ExplainabilityConfig(
            top_k=2,
            normalize=False,
        )
        explainer = GlobalFeatureImportanceExplainer(config)

        result = explainer.explain(
            {
                "feature_a": 0.1,
                "feature_b": 0.9,
                "feature_c": 0.5,
                "feature_d": 0.7,
            }
        )

        assert len(result) == 2
        assert [item.feature_name for item in result] == [
            "feature_b",
            "feature_d",
        ]
        assert [item.rank for item in result] == [1, 2]

    def test_normalizes_importance_values_by_absolute_total(self):
        explainer = GlobalFeatureImportanceExplainer()

        result = explainer.explain(
            {
                "feature_a": 2.0,
                "feature_b": 3.0,
                "feature_c": 5.0,
            }
        )

        assert [
            item.importance
            for item in result
        ] == pytest.approx([0.5, 0.3, 0.2])

    def test_normalization_can_be_disabled(self):
        config = ExplainabilityConfig(normalize=False)
        explainer = GlobalFeatureImportanceExplainer(config)

        result = explainer.explain(
            {
                "feature_a": 2.0,
                "feature_b": 3.0,
            }
        )

        assert [
            item.importance
            for item in result
        ] == [3.0, 2.0]

    def test_equal_importance_preserves_input_order(self):
        config = ExplainabilityConfig(normalize=False)
        explainer = GlobalFeatureImportanceExplainer(config)

        result = explainer.explain(
            {
                "feature_a": 0.5,
                "feature_b": 0.5,
                "feature_c": 0.5,
            }
        )

        assert [item.feature_name for item in result] == [
            "feature_a",
            "feature_b",
            "feature_c",
        ]

    def test_empty_dictionary_returns_empty_list(self):
        explainer = GlobalFeatureImportanceExplainer()

        assert explainer.explain({}) == []

    def test_none_features_rejected(self):
        explainer = GlobalFeatureImportanceExplainer()

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
        explainer = GlobalFeatureImportanceExplainer()

        with pytest.raises(
            TypeError,
            match="features must be a dictionary",
        ):
            explainer.explain(features)

    @pytest.mark.parametrize(
        "importance",
        [
            math.inf,
            -math.inf,
            math.nan,
            "invalid",
            True,
            None,
        ],
    )
    def test_invalid_importance_rejected(self, importance):
        explainer = GlobalFeatureImportanceExplainer()

        with pytest.raises((TypeError, ValueError)):
            explainer.explain(
                {"feature_a": importance}
            )

    def test_empty_feature_name_rejected(self):
        explainer = GlobalFeatureImportanceExplainer()

        with pytest.raises(
            ValueError,
            match="feature_name must not be empty",
        ):
            explainer.explain(
                {"   ": 0.5}
            )

    def test_feature_name_is_normalized(self):
        config = ExplainabilityConfig(normalize=False)
        explainer = GlobalFeatureImportanceExplainer(config)

        result = explainer.explain(
            {"  volatility  ": 0.8}
        )

        assert result[0].feature_name == "volatility"

    def test_all_features_filtered_by_threshold(self):
        config = ExplainabilityConfig(
            importance_threshold=1.0,
        )
        explainer = GlobalFeatureImportanceExplainer(config)

        result = explainer.explain(
            {
                "feature_a": 0.2,
                "feature_b": 0.5,
            }
        )

        assert result == []