import math

import pytest

from src.explainability.models import (
    ExplanationResult,
    FeatureAttribution,
    FeatureImportance,
)


class TestFeatureAttribution:
    def test_valid_feature_attribution(self):
        attribution = FeatureAttribution(
            feature_name="volatility_20d",
            attribution=0.42,
            rank=1,
            metadata={"source": "local"},
        )

        assert attribution.feature_name == "volatility_20d"
        assert attribution.attribution == 0.42
        assert attribution.rank == 1
        assert attribution.metadata == {"source": "local"}

    def test_feature_name_is_stripped(self):
        attribution = FeatureAttribution(
            feature_name="  volatility_20d  ",
            attribution=0.42,
        )

        assert attribution.feature_name == "volatility_20d"

    def test_empty_feature_name_rejected(self):
        with pytest.raises(ValueError, match="feature_name must not be empty"):
            FeatureAttribution(
                feature_name="   ",
                attribution=0.42,
            )

    def test_non_string_feature_name_rejected(self):
        with pytest.raises(TypeError, match="feature_name must be a string"):
            FeatureAttribution(
                feature_name=123,
                attribution=0.42,
            )

    @pytest.mark.parametrize(
        "value",
        [
            "invalid",
            True,
            None,
            math.inf,
            -math.inf,
            math.nan,
        ],
    )
    def test_invalid_attribution_rejected(self, value):
        with pytest.raises((TypeError, ValueError)):
            FeatureAttribution(
                feature_name="volatility_20d",
                attribution=value,
            )

    @pytest.mark.parametrize(
        "rank",
        [
            0,
            -1,
            1.5,
            True,
            "1",
        ],
    )
    def test_invalid_rank_rejected(self, rank):
        with pytest.raises((TypeError, ValueError)):
            FeatureAttribution(
                feature_name="volatility_20d",
                attribution=0.42,
                rank=rank,
            )

    def test_none_rank_is_allowed(self):
        attribution = FeatureAttribution(
            feature_name="volatility_20d",
            attribution=0.42,
            rank=None,
        )

        assert attribution.rank is None

    def test_invalid_metadata_rejected(self):
        with pytest.raises(TypeError, match="metadata must be a dictionary"):
            FeatureAttribution(
                feature_name="volatility_20d",
                attribution=0.42,
                metadata=["invalid"],
            )


class TestFeatureImportance:
    def test_valid_feature_importance(self):
        importance = FeatureImportance(
            feature_name="momentum_10d",
            importance=0.81,
            rank=1,
            metadata={"scope": "global"},
        )

        assert importance.feature_name == "momentum_10d"
        assert importance.importance == 0.81
        assert importance.rank == 1
        assert importance.metadata == {"scope": "global"}

    def test_empty_feature_name_rejected(self):
        with pytest.raises(ValueError):
            FeatureImportance(
                feature_name="",
                importance=0.81,
            )

    @pytest.mark.parametrize(
        "value",
        [
            "invalid",
            True,
            None,
            math.inf,
            -math.inf,
            math.nan,
        ],
    )
    def test_invalid_importance_rejected(self, value):
        with pytest.raises((TypeError, ValueError)):
            FeatureImportance(
                feature_name="momentum_10d",
                importance=value,
            )

    def test_invalid_rank_rejected(self):
        with pytest.raises(ValueError):
            FeatureImportance(
                feature_name="momentum_10d",
                importance=0.81,
                rank=0,
            )


class TestExplanationResult:
    def test_valid_explanation_result(self):
        attribution = FeatureAttribution(
            feature_name="volatility_20d",
            attribution=0.42,
            rank=1,
        )

        importance = FeatureImportance(
            feature_name="momentum_10d",
            importance=0.81,
            rank=1,
        )

        result = ExplanationResult(
            attributions=[attribution],
            importances=[importance],
            prediction="RISK_OFF",
            confidence=0.85,
            warnings=["High volatility detected"],
            metadata={"model": "bayesian_regime_engine"},
        )

        assert result.attributions == [attribution]
        assert result.importances == [importance]
        assert result.prediction == "RISK_OFF"
        assert result.confidence == 0.85
        assert result.warnings == ["High volatility detected"]
        assert result.metadata == {
            "model": "bayesian_regime_engine"
        }

    def test_default_collections_are_independent(self):
        result_one = ExplanationResult()
        result_two = ExplanationResult()

        result_one.warnings.append("warning")

        assert result_one.warnings == ["warning"]
        assert result_two.warnings == []

    def test_invalid_attributions_collection_rejected(self):
        with pytest.raises(TypeError):
            ExplanationResult(
                attributions="invalid",
            )

    def test_invalid_attribution_instance_rejected(self):
        with pytest.raises(TypeError):
            ExplanationResult(
                attributions=["invalid"],
            )

    def test_invalid_importances_collection_rejected(self):
        with pytest.raises(TypeError):
            ExplanationResult(
                importances="invalid",
            )

    def test_invalid_importance_instance_rejected(self):
        with pytest.raises(TypeError):
            ExplanationResult(
                importances=["invalid"],
            )

    @pytest.mark.parametrize(
        "confidence",
        [
            -0.01,
            1.01,
            math.inf,
            -math.inf,
            math.nan,
        ],
    )
    def test_invalid_confidence_rejected(self, confidence):
        with pytest.raises(ValueError):
            ExplanationResult(
                confidence=confidence,
            )

    @pytest.mark.parametrize(
        "confidence",
        [
            "invalid",
            True,
        ],
    )
    def test_invalid_confidence_type_rejected(self, confidence):
        with pytest.raises(TypeError):
            ExplanationResult(
                confidence=confidence,
            )

    def test_confidence_boundaries_allowed(self):
        low = ExplanationResult(confidence=0.0)
        high = ExplanationResult(confidence=1.0)

        assert low.confidence == 0.0
        assert high.confidence == 1.0

    def test_invalid_warnings_collection_rejected(self):
        with pytest.raises(TypeError):
            ExplanationResult(
                warnings="invalid",
            )

    def test_non_string_warning_rejected(self):
        with pytest.raises(TypeError):
            ExplanationResult(
                warnings=["valid warning", 123],
            )

    def test_invalid_metadata_rejected(self):
        with pytest.raises(TypeError):
            ExplanationResult(
                metadata=["invalid"],
            )