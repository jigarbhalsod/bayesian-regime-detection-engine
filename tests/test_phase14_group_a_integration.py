import pytest

from src.explainability.base import BaseExplainer
from src.explainability.config import ExplainabilityConfig
from src.explainability.models import (
    ExplanationResult,
    FeatureAttribution,
    FeatureImportance,
)


class IntegrationExplainer(BaseExplainer):
    """Concrete explainer used for end-to-end Group A integration tests."""

    def explain(self, features, **kwargs):
        self.validate_inputs(features)

        attributions = [
            FeatureAttribution(
                feature_name=name,
                attribution=value,
                rank=index + 1,
                metadata={"type": "local"},
            )
            for index, (name, value) in enumerate(features.items())
        ]

        importances = [
            FeatureImportance(
                feature_name=attribution.feature_name,
                importance=abs(attribution.attribution),
                rank=attribution.rank,
                metadata={"type": "global"},
            )
            for attribution in attributions
        ]

        normalized_values = self.normalize(
            [item.attribution for item in attributions]
        )

        normalized_attributions = [
            FeatureAttribution(
                feature_name=item.feature_name,
                attribution=normalized_value,
                rank=item.rank,
                metadata=item.metadata,
            )
            for item, normalized_value in zip(
                attributions,
                normalized_values,
            )
        ]

        return self.build_result(
            attributions=normalized_attributions,
            importances=importances,
            prediction=kwargs.get("prediction"),
            confidence=kwargs.get("confidence"),
            warnings=kwargs.get("warnings", []),
            metadata=kwargs.get("metadata", {}),
        )


class TestPhase14GroupAIntegration:
    def test_complete_default_explainability_flow(self):
        explainer = IntegrationExplainer()

        result = explainer.explain(
            {
                "volatility_20d": 3.0,
                "momentum_10d": -2.0,
                "volume_ratio": 1.0,
            },
            prediction="RISK_OFF",
            confidence=0.85,
            warnings=["Market volatility elevated"],
            metadata={"model": "integration_test"},
        )

        assert isinstance(result, ExplanationResult)
        assert result.prediction == "RISK_OFF"
        assert result.confidence == 0.85

        assert len(result.attributions) == 3
        assert len(result.importances) == 3

        assert result.attributions[0].feature_name == "volatility_20d"
        assert result.attributions[0].rank == 1

        assert result.importances[1].feature_name == "momentum_10d"
        assert result.importances[1].importance == 2.0

        assert result.warnings == ["Market volatility elevated"]
        assert result.metadata == {"model": "integration_test"}

    def test_normalized_attributions_preserve_sign(self):
        explainer = IntegrationExplainer()

        result = explainer.explain(
            {
                "positive": 2.0,
                "negative": -1.0,
                "larger_positive": 3.0,
            }
        )

        values = [
            item.attribution
            for item in result.attributions
        ]

        assert values == pytest.approx(
            [2 / 6, -1 / 6, 3 / 6]
        )

    def test_normalized_absolute_values_sum_to_one(self):
        explainer = IntegrationExplainer()

        result = explainer.explain(
            {
                "feature_a": 4.0,
                "feature_b": -3.0,
                "feature_c": 1.0,
            }
        )

        total = sum(
            abs(item.attribution)
            for item in result.attributions
        )

        assert total == pytest.approx(1.0)

    def test_normalization_can_be_disabled(self):
        config = ExplainabilityConfig(normalize=False)
        explainer = IntegrationExplainer(config=config)

        result = explainer.explain(
            {
                "feature_a": 2.0,
                "feature_b": -3.0,
            }
        )

        assert [
            item.attribution
            for item in result.attributions
        ] == [2.0, -3.0]

    def test_zero_attributions_flow_through_safely(self):
        explainer = IntegrationExplainer()

        result = explainer.explain(
            {
                "feature_a": 0.0,
                "feature_b": 0.0,
            }
        )

        assert [
            item.attribution
            for item in result.attributions
        ] == [0.0, 0.0]

        assert [
            item.importance
            for item in result.importances
        ] == [0.0, 0.0]

    def test_default_result_fields_are_preserved(self):
        explainer = IntegrationExplainer()

        result = explainer.explain(
            {"feature_a": 1.0}
        )

        assert result.prediction is None
        assert result.confidence is None
        assert result.warnings == []
        assert result.metadata == {}

    def test_none_features_rejected_through_full_flow(self):
        explainer = IntegrationExplainer()

        with pytest.raises(
            ValueError,
            match="features must not be None",
        ):
            explainer.explain(None)

    def test_invalid_confidence_propagates_through_integration(self):
        explainer = IntegrationExplainer()

        with pytest.raises(
            ValueError,
            match="confidence must be between 0.0 and 1.0",
        ):
            explainer.explain(
                {"feature_a": 1.0},
                confidence=1.5,
            )

    def test_feature_ranks_are_assigned_consistently(self):
        explainer = IntegrationExplainer()

        result = explainer.explain(
            {
                "feature_a": 1.0,
                "feature_b": 2.0,
                "feature_c": 3.0,
            }
        )

        assert [
            item.rank
            for item in result.attributions
        ] == [1, 2, 3]

        assert [
            item.rank
            for item in result.importances
        ] == [1, 2, 3]

    def test_feature_metadata_flows_into_result_models(self):
        explainer = IntegrationExplainer()

        result = explainer.explain(
            {"feature_a": 1.0}
        )

        assert result.attributions[0].metadata == {
            "type": "local"
        }

        assert result.importances[0].metadata == {
            "type": "global"
        }