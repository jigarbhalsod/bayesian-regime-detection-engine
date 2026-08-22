import pytest

from src.models.base import BaseAdvancedRegimeModel
from src.models.config import AdvancedModelConfig
from src.models.result import AdvancedModelResult


class DummyAdvancedModel(BaseAdvancedRegimeModel):
    """
    Minimal concrete implementation for testing the base class.
    """

    def fit(self, data):
        self.validate_input(data)
        self._mark_fitted()
        return self

    def predict(self, data):
        self.validate_input(data)
        self._require_fitted()

        return AdvancedModelResult(
            regimes=["RISK_ON"] * len(data),
            model_name=self.model_name,
        )


class ProbabilityDummyModel(DummyAdvancedModel):

    def predict_proba(self, data):
        self.validate_input(data)
        self._require_fitted()

        return [
            [1.0, 0.0, 0.0]
            for _ in range(len(data))
        ]


def test_default_configuration_is_created():
    model = DummyAdvancedModel()

    assert isinstance(
        model.config,
        AdvancedModelConfig,
    )
    assert model.model_name == "advanced_model"
    assert model.n_regimes == 3
    assert model.is_fitted is False


def test_custom_configuration():
    config = AdvancedModelConfig(
        model_name="dummy",
        n_regimes=4,
    )

    model = DummyAdvancedModel(config)

    assert model.model_name == "dummy"
    assert model.n_regimes == 4


def test_invalid_config_rejected():
    with pytest.raises(TypeError):
        DummyAdvancedModel(config={})


def test_validate_input_rejects_none():
    model = DummyAdvancedModel()

    with pytest.raises(ValueError):
        model.validate_input(None)


def test_validate_input_rejects_string():
    model = DummyAdvancedModel()

    with pytest.raises(TypeError):
        model.validate_input("invalid")


def test_validate_input_rejects_empty_list():
    model = DummyAdvancedModel()

    with pytest.raises(ValueError):
        model.validate_input([])


def test_validate_input_accepts_non_empty_list():
    model = DummyAdvancedModel()

    data = [1, 2, 3]

    assert model.validate_input(data) == data


def test_model_is_unfitted_initially():
    model = DummyAdvancedModel()

    assert model.is_fitted is False


def test_fit_marks_model_as_fitted():
    model = DummyAdvancedModel()

    returned = model.fit([1, 2, 3])

    assert returned is model
    assert model.is_fitted is True


def test_mark_unfitted():
    model = DummyAdvancedModel()

    model.fit([1])
    model._mark_unfitted()

    assert model.is_fitted is False


def test_predict_before_fit_rejected():
    model = DummyAdvancedModel()

    with pytest.raises(RuntimeError):
        model.predict([1, 2])


def test_predict_after_fit_returns_standard_result():
    model = DummyAdvancedModel()

    model.fit([1, 2])

    result = model.predict([3, 4])

    assert isinstance(result, AdvancedModelResult)
    assert result.regimes == [
        "RISK_ON",
        "RISK_ON",
    ]
    assert result.model_name == model.model_name


def test_default_predict_proba_not_supported():
    model = DummyAdvancedModel()

    with pytest.raises(NotImplementedError):
        model.predict_proba([1, 2])


def test_custom_predict_proba():
    model = ProbabilityDummyModel()

    model.fit([1, 2])

    probabilities = model.predict_proba([3, 4])

    assert probabilities == [
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ]


def test_require_fitted_after_fit():
    model = DummyAdvancedModel()

    model.fit([1])

    model._require_fitted()


def test_repr_contains_key_information():
    config = AdvancedModelConfig(
        model_name="test_model",
        n_regimes=5,
    )

    model = DummyAdvancedModel(config)

    representation = repr(model)

    assert "DummyAdvancedModel" in representation
    assert "test_model" in representation
    assert "5" in representation
    assert "False" in representation