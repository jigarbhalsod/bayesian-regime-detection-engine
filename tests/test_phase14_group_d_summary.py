import pytest

from src.explainability.models import (
    ExplanationResult,
    FeatureAttribution,
    FeatureImportance,
)
from src.explainability.regime_models import RegimeExplanation
from src.explainability.summary import ExplanationSummaryEngine


class TestExplanationSummaryEngine:
    def test_summarizes_empty_list(self):
        engine = ExplanationSummaryEngine()

        result = engine.summarize([])

        assert result == {
            "predictions": [],
            "confidences": [],
            "feature_drivers": [],
            "regimes": [],
            "metadata": {},
        }

    def test_extracts_prediction(self):
        engine = ExplanationSummaryEngine()

        explanation = ExplanationResult(
            prediction="UP",
        )

        result = engine.summarize([explanation])

        assert result["predictions"] == ["UP"]

    def test_ignores_none_prediction(self):
        engine = ExplanationSummaryEngine()

        explanation = ExplanationResult()

        result = engine.summarize([explanation])

        assert result["predictions"] == []

    def test_extracts_confidence(self):
        engine = ExplanationSummaryEngine()

        explanation = ExplanationResult(
            confidence=0.85,
        )

        result = engine.summarize([explanation])

        assert result["confidences"] == [0.85]

    def test_extracts_regime_probability_as_confidence(self):
        engine = ExplanationSummaryEngine()

        explanation = RegimeExplanation(
            regime="RISK_ON",
            probability=0.72,
        )

        result = engine.summarize([explanation])

        assert result["confidences"] == [0.72]

    def test_ignores_none_confidences(self):
        engine = ExplanationSummaryEngine()

        prediction_explanation = ExplanationResult()
        regime_explanation = RegimeExplanation(
            regime="RISK_OFF",
        )

        result = engine.summarize(
            [
                prediction_explanation,
                regime_explanation,
            ]
        )

        assert result["confidences"] == []

    def test_extracts_attribution_feature_drivers(self):
        engine = ExplanationSummaryEngine()

        explanation = ExplanationResult(
            attributions=[
                FeatureAttribution(
                    feature_name="volatility",
                    attribution=-0.8,
                ),
                FeatureAttribution(
                    feature_name="momentum",
                    attribution=0.5,
                ),
            ]
        )

        result = engine.summarize([explanation])

        assert result["feature_drivers"] == [
            "volatility",
            "momentum",
        ]

    def test_extracts_importance_feature_drivers(self):
        engine = ExplanationSummaryEngine()

        explanation = ExplanationResult(
            importances=[
                FeatureImportance(
                    feature_name="volume",
                    importance=0.7,
                ),
                FeatureImportance(
                    feature_name="vix",
                    importance=0.4,
                ),
            ]
        )

        result = engine.summarize([explanation])

        assert result["feature_drivers"] == [
            "volume",
            "vix",
        ]

    def test_preserves_attribution_then_importance_order(self):
        engine = ExplanationSummaryEngine()

        explanation = ExplanationResult(
            attributions=[
                FeatureAttribution(
                    feature_name="momentum",
                    attribution=0.8,
                ),
            ],
            importances=[
                FeatureImportance(
                    feature_name="volatility",
                    importance=0.9,
                ),
            ],
        )

        result = engine.summarize([explanation])

        assert result["feature_drivers"] == [
            "momentum",
            "volatility",
        ]

    def test_extracts_regime_and_key_drivers(self):
        engine = ExplanationSummaryEngine()

        explanation = RegimeExplanation(
            regime="RISK_OFF",
            key_drivers=[
                "volatility",
                "liquidity",
            ],
        )

        result = engine.summarize([explanation])

        assert result["regimes"] == ["RISK_OFF"]
        assert result["feature_drivers"] == [
            "volatility",
            "liquidity",
        ]

    def test_preserves_duplicate_feature_drivers(self):
        engine = ExplanationSummaryEngine()

        explanation = ExplanationResult(
            attributions=[
                FeatureAttribution(
                    feature_name="volatility",
                    attribution=0.5,
                ),
            ],
            importances=[
                FeatureImportance(
                    feature_name="volatility",
                    importance=0.8,
                ),
            ],
        )

        result = engine.summarize([explanation])

        assert result["feature_drivers"] == [
            "volatility",
            "volatility",
        ]

    def test_merges_metadata_with_later_values_overriding(self):
        engine = ExplanationSummaryEngine()

        first = ExplanationResult(
            metadata={
                "model": "hmm",
                "version": 1,
            }
        )

        second = RegimeExplanation(
            regime="RISK_ON",
            metadata={
                "model": "bayesian",
                "source": "regime_engine",
            },
        )

        result = engine.summarize([first, second])

        assert result["metadata"] == {
            "model": "bayesian",
            "version": 1,
            "source": "regime_engine",
        }

    def test_preserves_multiple_explanation_order(self):
        engine = ExplanationSummaryEngine()

        first = ExplanationResult(
            prediction="UP",
            confidence=0.6,
        )

        second = ExplanationResult(
            prediction="DOWN",
            confidence=0.8,
        )

        result = engine.summarize([first, second])

        assert result["predictions"] == [
            "UP",
            "DOWN",
        ]
        assert result["confidences"] == [
            0.6,
            0.8,
        ]

    def test_handles_mixed_explanation_flow(self):
        engine = ExplanationSummaryEngine()

        prediction = ExplanationResult(
            prediction="UP",
            confidence=0.82,
            attributions=[
                FeatureAttribution(
                    feature_name="momentum",
                    attribution=0.7,
                ),
            ],
            importances=[
                FeatureImportance(
                    feature_name="trend",
                    importance=0.9,
                ),
            ],
            metadata={"prediction_model": "ensemble"},
        )

        regime = RegimeExplanation(
            regime="RISK_ON",
            probability=0.75,
            key_drivers=[
                "volatility",
                "liquidity",
            ],
            metadata={"regime_model": "hmm"},
        )

        result = engine.summarize(
            [
                prediction,
                regime,
            ]
        )

        assert result["predictions"] == ["UP"]
        assert result["confidences"] == [
            0.82,
            0.75,
        ]
        assert result["feature_drivers"] == [
            "momentum",
            "trend",
            "volatility",
            "liquidity",
        ]
        assert result["regimes"] == ["RISK_ON"]
        assert result["metadata"] == {
            "prediction_model": "ensemble",
            "regime_model": "hmm",
        }

    def test_none_explanations_rejected(self):
        engine = ExplanationSummaryEngine()

        with pytest.raises(ValueError):
            engine.summarize(None)

    @pytest.mark.parametrize(
        "explanations",
        [
            "invalid",
            123,
            True,
            {},
            (),
        ],
    )
    def test_non_list_explanations_rejected(
        self,
        explanations,
    ):
        engine = ExplanationSummaryEngine()

        with pytest.raises(TypeError):
            engine.summarize(explanations)

    @pytest.mark.parametrize(
        "explanations",
        [
            [None],
            ["invalid"],
            [123],
            [True],
            [{}],
        ],
    )
    def test_invalid_explanation_instances_rejected(
        self,
        explanations,
    ):
        engine = ExplanationSummaryEngine()

        with pytest.raises(TypeError):
            engine.summarize(explanations)

    def test_mixed_valid_and_invalid_explanations_rejected(self):
        engine = ExplanationSummaryEngine()

        valid = ExplanationResult(prediction="UP")

        with pytest.raises(TypeError):
            engine.summarize(
                [
                    valid,
                    "invalid",
                ]
            )