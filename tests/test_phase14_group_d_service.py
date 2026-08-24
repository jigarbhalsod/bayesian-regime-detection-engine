import pytest

from src.explainability.service import ExplainabilityService


class TestExplainabilityService:
    def test_default_dependencies_are_created(self):
        service = ExplainabilityService()

        assert service.prediction_explainer is not None
        assert service.regime_generator is not None
        assert service.aggregator is not None
        assert service.summary_engine is not None
        assert service.consistency_validator is not None

    def test_empty_flow_returns_complete_structure(self):
        service = ExplainabilityService()

        result = service.explain()

        assert set(result.keys()) == {
            "explanation",
            "summary",
            "consistency",
        }

        assert result["summary"] == {
            "predictions": [],
            "confidences": [],
            "feature_drivers": [],
            "regimes": [],
            "metadata": {},
        }

        assert result["consistency"]["is_consistent"] is True
        assert result["consistency"]["issues"] == []

    def test_prediction_only_flow(self):
        service = ExplainabilityService()

        result = service.explain(
            prediction="UP",
            confidence=0.85,
            attributions={
                "momentum": 0.7,
                "volatility": -0.3,
            },
            importances={
                "momentum": 0.8,
                "volatility": 0.4,
            },
            metadata={"model": "ensemble"},
        )

        assert result["explanation"] is not None
        assert result["summary"]["predictions"] == ["UP"]
        assert result["summary"]["confidences"] == [0.85]
        assert "momentum" in result["summary"]["feature_drivers"]
        assert "volatility" in result["summary"]["feature_drivers"]
        assert result["summary"]["regimes"] == []
        assert result["summary"]["metadata"] == {
            "model": "ensemble",
        }
        assert result["consistency"]["is_consistent"] is True

    def test_prediction_without_optional_values(self):
        service = ExplainabilityService()

        result = service.explain(
            prediction="DOWN",
        )

        assert result["explanation"] is not None
        assert result["summary"]["predictions"] == ["DOWN"]
        assert result["summary"]["confidences"] == []
        assert result["summary"]["feature_drivers"] == []
        assert result["summary"]["regimes"] == []
        assert result["consistency"]["is_consistent"] is True

    def test_regime_only_flow(self):
        service = ExplainabilityService()

        result = service.explain(
            regime="RISK_ON",
            probability=0.78,
            contributions={
                "momentum": 0.6,
                "volatility": -0.2,
            },
            metadata={"source": "regime_model"},
        )

        assert result["explanation"] is not None
        assert result["summary"]["predictions"] == []
        assert result["summary"]["confidences"] == [0.78]
        assert "RISK_ON" in result["summary"]["regimes"]
        assert "momentum" in result["summary"]["feature_drivers"]
        assert "volatility" in result["summary"]["feature_drivers"]
        assert result["summary"]["metadata"] == {
            "source": "regime_model",
        }
        assert result["consistency"]["is_consistent"] is True

    def test_regime_without_optional_values(self):
        service = ExplainabilityService()

        result = service.explain(
            regime="RISK_OFF",
        )

        assert result["explanation"] is not None
        assert result["summary"]["predictions"] == []
        assert result["summary"]["confidences"] == []
        assert result["summary"]["feature_drivers"] == []
        assert result["summary"]["regimes"] == ["RISK_OFF"]
        assert result["consistency"]["is_consistent"] is True

    def test_complete_prediction_and_regime_flow(self):
        service = ExplainabilityService()

        result = service.explain(
            prediction="UP",
            confidence=0.91,
            attributions={
                "momentum": 0.8,
                "volatility": -0.4,
            },
            importances={
                "momentum": 0.9,
                "volume": 0.5,
            },
            regime="RISK_ON",
            probability=0.82,
            contributions={
                "macro_growth": 0.7,
                "volatility": -0.3,
            },
            metadata={"model": "combined_ensemble"},
        )

        assert result["explanation"] is not None

        assert result["summary"]["predictions"] == ["UP"]
        assert result["summary"]["confidences"] == [
            0.91,
            0.82,
        ]

        assert "momentum" in result["summary"]["feature_drivers"]
        assert "volatility" in result["summary"]["feature_drivers"]
        assert "volume" in result["summary"]["feature_drivers"]
        assert "macro_growth" in result["summary"]["feature_drivers"]

        assert result["summary"]["regimes"] == ["RISK_ON"]

        assert result["summary"]["metadata"] == {
            "model": "combined_ensemble",
        }

        assert result["consistency"]["is_consistent"] is True
        assert result["consistency"]["issues"] == []

    def test_metadata_flows_through_complete_pipeline(self):
        service = ExplainabilityService()

        metadata = {
            "model": "bayesian_ensemble",
            "version": "1.0",
            "source": "integration_test",
        }

        result = service.explain(
            prediction="UP",
            confidence=0.8,
            regime="RISK_ON",
            probability=0.7,
            metadata=metadata,
        )

        assert result["summary"]["metadata"] == metadata

    def test_duplicate_feature_drivers_are_preserved(self):
        service = ExplainabilityService()

        result = service.explain(
            prediction="UP",
            attributions={
                "momentum": 0.7,
            },
            importances={
                "momentum": 0.8,
            },
            regime="RISK_ON",
            contributions={
                "momentum": 0.6,
            },
        )

        drivers = result["summary"]["feature_drivers"]

        assert drivers.count("momentum") == 3

    @pytest.mark.parametrize(
        ("prediction", "confidence"),
        [
            ("UP", 0.0),
            ("UP", 1.0),
            ("DOWN", 0.5),
        ],
    )
    def test_valid_prediction_confidence_boundaries(
        self,
        prediction,
        confidence,
    ):
        service = ExplainabilityService()

        result = service.explain(
            prediction=prediction,
            confidence=confidence,
        )

        assert result["summary"]["confidences"] == [confidence]
        assert result["consistency"]["is_consistent"] is True

    @pytest.mark.parametrize(
        ("regime", "probability"),
        [
            ("RISK_ON", 0.0),
            ("RISK_ON", 1.0),
            ("TRANSITIONAL", 0.5),
        ],
    )
    def test_valid_regime_probability_boundaries(
        self,
        regime,
        probability,
    ):
        service = ExplainabilityService()

        result = service.explain(
            regime=regime,
            probability=probability,
        )

        assert result["summary"]["confidences"] == [probability]
        assert result["consistency"]["is_consistent"] is True

    def test_prediction_and_regime_metadata_merge_consistently(self):
        service = ExplainabilityService()

        result = service.explain(
            prediction="UP",
            confidence=0.8,
            regime="RISK_ON",
            probability=0.75,
            metadata={"model": "shared_model"},
        )

        assert result["summary"]["metadata"] == {
            "model": "shared_model",
        }

    def test_explanation_result_is_aggregated(self):
        service = ExplainabilityService()

        result = service.explain(
            prediction="UP",
            confidence=0.85,
            regime="RISK_ON",
            probability=0.75,
        )

        explanation = result["explanation"]

        assert explanation is not None
        assert explanation.prediction == "UP"
        assert explanation.confidence == 0.75

    def test_final_summary_is_validated(self):
        service = ExplainabilityService()

        result = service.explain(
            prediction="UP",
            confidence=0.9,
            attributions={"momentum": 0.6},
            regime="RISK_ON",
            probability=0.8,
            contributions={"macro": 0.4},
        )

        assert result["consistency"] == {
            "is_consistent": True,
            "issues": [],
        }