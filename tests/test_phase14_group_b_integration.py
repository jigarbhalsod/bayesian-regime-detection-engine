import pytest

from src.explainability.aggregation import ExplanationAggregator
from src.explainability.config import ExplainabilityConfig
from src.explainability.global_importance import (
    GlobalFeatureImportanceExplainer,
)
from src.explainability.local_attribution import (
    LocalFeatureAttributionExplainer,
)
from src.explainability.models import ExplanationResult
from src.explainability.prediction import PredictionExplainer


class TestPhase14GroupBIntegration:
    def test_complete_explainability_pipeline(self):
        config = ExplainabilityConfig(
            normalize=False,
            top_k=3,
        )

        global_explainer = GlobalFeatureImportanceExplainer(config)
        local_explainer = LocalFeatureAttributionExplainer(config)
        prediction_explainer = PredictionExplainer(config)
        aggregator = ExplanationAggregator()

        global_scores = {
            "volatility": 0.9,
            "momentum": 0.6,
            "volume": 0.4,
        }

        local_scores = {
            "volatility": 0.7,
            "momentum": -0.5,
            "volume": 0.2,
        }

        importances = global_explainer.explain(global_scores)
        attributions = local_explainer.explain(local_scores)

        prediction_result = prediction_explainer.explain(
            attributions,
            prediction="RISK_OFF",
            confidence=0.87,
            importances=importances,
            warnings=["Elevated volatility"],
            metadata={"model": "regime_engine"},
        )

        aggregated = aggregator.aggregate([prediction_result])

        assert isinstance(aggregated, ExplanationResult)
        assert aggregated.prediction == "RISK_OFF"
        assert aggregated.confidence == 0.87
        assert len(aggregated.attributions) == 3
        assert len(aggregated.importances) == 3
        assert aggregated.warnings == ["Elevated volatility"]
        assert aggregated.metadata == {
            "model": "regime_engine"
        }

    def test_global_and_local_rankings_work_together(self):
        config = ExplainabilityConfig(normalize=False)

        global_explainer = GlobalFeatureImportanceExplainer(config)
        local_explainer = LocalFeatureAttributionExplainer(config)

        importances = global_explainer.explain(
            {
                "volatility": 0.9,
                "momentum": 0.7,
                "volume": 0.3,
            }
        )

        attributions = local_explainer.explain(
            {
                "volatility": -0.8,
                "momentum": 0.5,
                "volume": 0.2,
            }
        )

        assert [item.feature_name for item in importances] == [
            "volatility",
            "momentum",
            "volume",
        ]

        assert [item.feature_name for item in attributions] == [
            "volatility",
            "momentum",
            "volume",
        ]

        assert [item.rank for item in importances] == [1, 2, 3]
        assert [item.rank for item in attributions] == [1, 2, 3]

        assert attributions[0].attribution == -0.8

    def test_prediction_explanation_flows_into_aggregation(self):
        config = ExplainabilityConfig(normalize=False)

        local_explainer = LocalFeatureAttributionExplainer(config)
        prediction_explainer = PredictionExplainer(config)
        aggregator = ExplanationAggregator()

        attributions = local_explainer.explain(
            {
                "vix_change": 0.6,
                "market_return": -0.4,
            }
        )

        result = prediction_explainer.explain(
            attributions,
            prediction="RISK_OFF",
            confidence=0.82,
        )

        final_result = aggregator.aggregate([result])

        assert final_result.prediction == "RISK_OFF"
        assert final_result.confidence == 0.82
        assert len(final_result.attributions) == 2
        assert final_result.importances == []

    def test_multiple_prediction_results_are_aggregated(self):
        config = ExplainabilityConfig(normalize=False)

        local_explainer = LocalFeatureAttributionExplainer(config)
        prediction_explainer = PredictionExplainer(config)
        aggregator = ExplanationAggregator()

        first_attributions = local_explainer.explain(
            {"volatility": 0.7}
        )

        second_attributions = local_explainer.explain(
            {"momentum": -0.6}
        )

        first_result = prediction_explainer.explain(
            first_attributions,
            prediction="RISK_ON",
            confidence=0.65,
            warnings=["First model warning"],
            metadata={"model_a": True},
        )

        second_result = prediction_explainer.explain(
            second_attributions,
            prediction="RISK_OFF",
            confidence=0.91,
            warnings=["Second model warning"],
            metadata={"model_b": True},
        )

        final_result = aggregator.aggregate(
            [first_result, second_result]
        )

        assert final_result.prediction == "RISK_OFF"
        assert final_result.confidence == 0.91

        assert [
            item.feature_name
            for item in final_result.attributions
        ] == [
            "volatility",
            "momentum",
        ]

        assert final_result.warnings == [
            "First model warning",
            "Second model warning",
        ]

        assert final_result.metadata == {
            "model_a": True,
            "model_b": True,
        }

    def test_normalized_pipeline_preserves_expected_totals(self):
        config = ExplainabilityConfig(
            normalize=True,
            top_k=3,
        )

        global_explainer = GlobalFeatureImportanceExplainer(config)
        local_explainer = LocalFeatureAttributionExplainer(config)

        importances = global_explainer.explain(
            {
                "feature_a": 2.0,
                "feature_b": 3.0,
                "feature_c": 5.0,
            }
        )

        attributions = local_explainer.explain(
            {
                "feature_a": 2.0,
                "feature_b": -3.0,
                "feature_c": 5.0,
            }
        )

        assert sum(
            item.importance
            for item in importances
        ) == pytest.approx(1.0)

        assert sum(
            abs(item.attribution)
            for item in attributions
        ) == pytest.approx(1.0)

    def test_top_k_is_consistent_across_pipeline(self):
        config = ExplainabilityConfig(
            top_k=2,
            normalize=False,
        )

        global_explainer = GlobalFeatureImportanceExplainer(config)
        local_explainer = LocalFeatureAttributionExplainer(config)
        prediction_explainer = PredictionExplainer(config)

        scores = {
            "feature_a": 0.9,
            "feature_b": 0.8,
            "feature_c": 0.7,
            "feature_d": 0.6,
        }

        importances = global_explainer.explain(scores)

        attributions = local_explainer.explain(
            {
                "feature_a": 0.9,
                "feature_b": -0.8,
                "feature_c": 0.7,
                "feature_d": -0.6,
            }
        )

        result = prediction_explainer.explain(
            attributions,
            prediction="TRANSITIONAL",
            confidence=0.75,
            importances=importances,
        )

        assert len(result.importances) == 2
        assert len(result.attributions) == 2

        assert [
            item.feature_name
            for item in result.importances
        ] == ["feature_a", "feature_b"]

        assert [
            item.feature_name
            for item in result.attributions
        ] == ["feature_a", "feature_b"]

    def test_threshold_filtering_flows_through_pipeline(self):
        config = ExplainabilityConfig(
            importance_threshold=0.5,
            attribution_threshold=0.5,
            normalize=False,
        )

        global_explainer = GlobalFeatureImportanceExplainer(config)
        local_explainer = LocalFeatureAttributionExplainer(config)
        prediction_explainer = PredictionExplainer(config)

        importances = global_explainer.explain(
            {
                "feature_a": 0.8,
                "feature_b": 0.4,
            }
        )

        attributions = local_explainer.explain(
            {
                "feature_a": 0.7,
                "feature_b": -0.3,
            }
        )

        result = prediction_explainer.explain(
            attributions,
            prediction="RISK_ON",
            importances=importances,
        )

        assert len(result.importances) == 1
        assert result.importances[0].feature_name == "feature_a"

        assert len(result.attributions) == 1
        assert result.attributions[0].feature_name == "feature_a"

    def test_empty_explanations_can_flow_through_pipeline(self):
        config = ExplainabilityConfig(
            importance_threshold=1.0,
            attribution_threshold=1.0,
        )

        global_explainer = GlobalFeatureImportanceExplainer(config)
        local_explainer = LocalFeatureAttributionExplainer(config)
        prediction_explainer = PredictionExplainer(config)
        aggregator = ExplanationAggregator()

        importances = global_explainer.explain(
            {"feature_a": 0.5}
        )

        attributions = local_explainer.explain(
            {"feature_a": -0.5}
        )

        prediction_result = prediction_explainer.explain(
            attributions,
            prediction="TRANSITIONAL",
            importances=importances,
        )

        final_result = aggregator.aggregate([prediction_result])

        assert final_result.attributions == []
        assert final_result.importances == []
        assert final_result.prediction == "TRANSITIONAL"