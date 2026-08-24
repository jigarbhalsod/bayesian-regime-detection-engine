import pytest

from src.explainability.base import BaseExplainer
from src.explainability.config import ExplainabilityConfig
from src.explainability.models import ExplanationResult


class DummyExplainer(BaseExplainer):
    """Concrete implementation for BaseExplainer tests."""

    def explain(
        self,
        features,
        **kwargs,
    ):
        self.validate_inputs(features)

        return self.build_result(
            prediction=kwargs.get("prediction"),
        )


class TestBaseExplainer:
    def test_cannot_instantiate_abstract_base_class(self):
        with pytest.raises(TypeError):
            BaseExplainer()

    def test_default_config_is_created(self):
        explainer = DummyExplainer()

        assert isinstance(
            explainer.config,
            ExplainabilityConfig,
        )

        assert explainer.config == ExplainabilityConfig()

    def test_custom_config_is_used(self):
        config = ExplainabilityConfig(
            top_k=5,
            normalize=False,
            method="test",
        )

        explainer = DummyExplainer(config=config)

        assert explainer.config is config
        assert explainer.config.top_k == 5
        assert explainer.config.normalize is False

    @pytest.mark.parametrize(
        "config",
        [
            "invalid",
            123,
            True,
            {},
            [],
        ],
    )
    def test_invalid_config_rejected(self, config):
        with pytest.raises(
            TypeError,
            match="config must be an ExplainabilityConfig instance or None",
        ):
            DummyExplainer(config=config)

    def test_validate_inputs_accepts_valid_features(self):
        explainer = DummyExplainer()

        explainer.validate_inputs(
            {"volatility": 0.25},
        )

    def test_validate_inputs_rejects_none(self):
        explainer = DummyExplainer()

        with pytest.raises(
            ValueError,
            match="features must not be None",
        ):
            explainer.validate_inputs(None)

    def test_explain_returns_explanation_result(self):
        explainer = DummyExplainer()

        result = explainer.explain(
            {"volatility": 0.25},
            prediction="RISK_OFF",
        )

        assert isinstance(result, ExplanationResult)
        assert result.prediction == "RISK_OFF"

    def test_explain_rejects_none_features(self):
        explainer = DummyExplainer()

        with pytest.raises(ValueError):
            explainer.explain(
                None,
                prediction="RISK_OFF",
            )

    def test_normalize_values(self):
        explainer = DummyExplainer()

        result = explainer.normalize([1.0, 2.0, 3.0])

        assert result == pytest.approx(
            [1 / 6, 2 / 6, 3 / 6],
        )

    def test_normalize_negative_values(self):
        explainer = DummyExplainer()

        result = explainer.normalize([-1.0, 2.0, -3.0])

        assert result == pytest.approx(
            [-1 / 6, 2 / 6, -3 / 6],
        )

    def test_normalize_zero_values(self):
        explainer = DummyExplainer()

        result = explainer.normalize([0.0, 0.0, 0.0])

        assert result == [0.0, 0.0, 0.0]

    def test_normalize_empty_values(self):
        explainer = DummyExplainer()

        assert explainer.normalize([]) == []

    def test_normalize_disabled_returns_float_values(self):
        config = ExplainabilityConfig(normalize=False)
        explainer = DummyExplainer(config=config)

        result = explainer.normalize([1, 2, -3])

        assert result == [1.0, 2.0, -3.0]

    def test_normalize_rejects_non_list(self):
        explainer = DummyExplainer()

        with pytest.raises(
            TypeError,
            match="values must be a list",
        ):
            explainer.normalize((1.0, 2.0))

    @pytest.mark.parametrize(
        "values",
        [
            [1.0, "invalid"],
            [True, 1.0],
            [None, 1.0],
            [{}, 1.0],
        ],
    )
    def test_normalize_rejects_invalid_values(self, values):
        explainer = DummyExplainer()

        with pytest.raises(
            TypeError,
            match="each value must be a numeric value",
        ):
            explainer.normalize(values)

    def test_build_result_returns_explanation_result(self):
        explainer = DummyExplainer()

        result = explainer.build_result(
            prediction="RISK_ON",
            confidence=0.9,
            warnings=["test warning"],
            metadata={"source": "test"},
        )

        assert isinstance(result, ExplanationResult)
        assert result.prediction == "RISK_ON"
        assert result.confidence == 0.9
        assert result.warnings == ["test warning"]
        assert result.metadata == {"source": "test"}

    def test_build_result_propagates_validation_errors(self):
        explainer = DummyExplainer()

        with pytest.raises(ValueError):
            explainer.build_result(
                confidence=1.5,
            )