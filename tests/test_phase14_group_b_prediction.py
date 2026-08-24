import pytest

from src.explainability.config import ExplainabilityConfig
from src.explainability.models import (
    ExplanationResult,
    FeatureAttribution,
    FeatureImportance,
)
from src.explainability.prediction import PredictionExplainer


class TestPredictionExplainer:
    def test_builds_complete_prediction_explanation(self):
        explainer = PredictionExplainer()

        attributions = [
            FeatureAttribution(
                feature_name="volatility",
                attribution=0.6,
                rank=1,
            ),
            FeatureAttribution(
                feature_name="momentum",
                attribution=-0.4,
                rank=2,
            ),
        ]

        importances = [
            FeatureImportance(
                feature_name="volatility",
                importance=0.8,
                rank=1,
            )
        ]

        result = explainer.explain(
            attributions,
            prediction="RISK_OFF",
            confidence=0.85,
            importances=importances,
            warnings=["Elevated market volatility"],
            metadata={"model": "bayesian_regime_engine"},
        )

        assert isinstance(result, ExplanationResult)
        assert result.prediction == "RISK_OFF"
        assert result.confidence == 0.85
        assert result.attributions == attributions
        assert result.importances == importances
        assert result.warnings == ["Elevated market volatility"]
        assert result.metadata == {
            "model": "bayesian_regime_engine"
        }

    def test_accepts_prediction_without_confidence(self):
        explainer = PredictionExplainer()

        result = explainer.explain(
            [],
            prediction="RISK_ON",
        )

        assert result.prediction == "RISK_ON"
        assert result.confidence is None
        assert result.attributions == []
        assert result.importances == []

    def test_accepts_prediction_with_only_attributions(self):
        explainer = PredictionExplainer()

        attributions = [
            FeatureAttribution(
                feature_name="volume",
                attribution=0.5,
            )
        ]

        result = explainer.explain(
            attributions,
            prediction="TRANSITIONAL",
        )

        assert len(result.attributions) == 1
        assert result.importances == []

    def test_preserves_attribution_order(self):
        explainer = PredictionExplainer()

        attributions = [
            FeatureAttribution(
                feature_name="feature_a",
                attribution=0.3,
                rank=1,
            ),
            FeatureAttribution(
                feature_name="feature_b",
                attribution=-0.2,
                rank=2,
            ),
        ]

        result = explainer.explain(
            attributions,
            prediction="RISK_ON",
        )

        assert [
            item.feature_name
            for item in result.attributions
        ] == ["feature_a", "feature_b"]

    def test_preserves_importance_order(self):
        explainer = PredictionExplainer()

        importances = [
            FeatureImportance(
                feature_name="feature_a",
                importance=0.8,
                rank=1,
            ),
            FeatureImportance(
                feature_name="feature_b",
                importance=0.4,
                rank=2,
            ),
        ]

        result = explainer.explain(
            [],
            prediction="RISK_ON",
            importances=importances,
        )

        assert [
            item.feature_name
            for item in result.importances
        ] == ["feature_a", "feature_b"]

    def test_none_features_rejected(self):
        explainer = PredictionExplainer()

        with pytest.raises(
            ValueError,
            match="features must not be None",
        ):
            explainer.explain(
                None,
                prediction="RISK_ON",
            )

    @pytest.mark.parametrize(
        "features",
        [
            {},
            (),
            "invalid",
            123,
        ],
    )
    def test_non_list_features_rejected(self, features):
        explainer = PredictionExplainer()

        with pytest.raises(
            TypeError,
            match="features must be a list of FeatureAttribution objects",
        ):
            explainer.explain(
                features,
                prediction="RISK_ON",
            )

    @pytest.mark.parametrize(
        "features",
        [
            ["invalid"],
            [123],
            [{}],
            [FeatureImportance("feature", 0.5)],
        ],
    )
    def test_invalid_feature_instances_rejected(self, features):
        explainer = PredictionExplainer()

        with pytest.raises(
            TypeError,
            match="each feature must be a FeatureAttribution instance",
        ):
            explainer.explain(
                features,
                prediction="RISK_ON",
            )

    def test_prediction_is_required(self):
        explainer = PredictionExplainer()

        with pytest.raises(
            ValueError,
            match="prediction must not be None",
        ):
            explainer.explain([])

    def test_none_prediction_is_rejected(self):
        explainer = PredictionExplainer()

        with pytest.raises(ValueError):
            explainer.explain(
                [],
                prediction=None,
            )

    @pytest.mark.parametrize(
        "importances",
        [
            "invalid",
            {},
            (),
            123,
        ],
    )
    def test_non_list_importances_rejected(self, importances):
        explainer = PredictionExplainer()

        with pytest.raises(
            TypeError,
            match="importances must be a list",
        ):
            explainer.explain(
                [],
                prediction="RISK_ON",
                importances=importances,
            )

    @pytest.mark.parametrize(
        "importances",
        [
            ["invalid"],
            [123],
            [{}],
            [FeatureAttribution("feature", 0.5)],
        ],
    )
    def test_invalid_importance_instances_rejected(self, importances):
        explainer = PredictionExplainer()

        with pytest.raises(
            TypeError,
            match="each importance must be a FeatureImportance instance",
        ):
            explainer.explain(
                [],
                prediction="RISK_ON",
                importances=importances,
            )

    @pytest.mark.parametrize(
        "confidence",
        [-0.1, 1.1],
    )
    def test_invalid_confidence_propagates_validation_error(
        self,
        confidence,
    ):
        explainer = PredictionExplainer()

        with pytest.raises(ValueError):
            explainer.explain(
                [],
                prediction="RISK_ON",
                confidence=confidence,
            )

    @pytest.mark.parametrize(
        "confidence",
        ["invalid", True],
    )
    def test_invalid_confidence_type_propagates_validation_error(
        self,
        confidence,
    ):
        explainer = PredictionExplainer()

        with pytest.raises(TypeError):
            explainer.explain(
                [],
                prediction="RISK_ON",
                confidence=confidence,
            )

    def test_invalid_warnings_propagate_validation_error(self):
        explainer = PredictionExplainer()

        with pytest.raises(TypeError):
            explainer.explain(
                [],
                prediction="RISK_ON",
                warnings="invalid",
            )

    def test_invalid_metadata_propagate_validation_error(self):
        explainer = PredictionExplainer()

        with pytest.raises(TypeError):
            explainer.explain(
                [],
                prediction="RISK_ON",
                metadata=["invalid"],
            )

    def test_custom_config_is_accepted(self):
        config = ExplainabilityConfig(
            method="prediction_test",
        )

        explainer = PredictionExplainer(config)

        result = explainer.explain(
            [],
            prediction="RISK_ON",
        )

        assert explainer.config is config
        assert result.prediction == "RISK_ON"