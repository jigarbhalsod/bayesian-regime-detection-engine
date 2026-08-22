import numpy as np
import pytest

from src.models.regime_var.base import (
    BaseRegimeVARModel,
)
from src.models.regime_var.config import (
    RegimeVARConfig,
)


class DummyRegimeVARModel(BaseRegimeVARModel):
    def fit(
        self,
        data,
        regimes,
        **kwargs,
    ):
        self._is_fitted = True
        return self

    def predict(
        self,
        history,
        regime,
        **kwargs,
    ):
        self._validate_fitted()
        history_values = self._validate_history(history)
        self._validate_regime_index(regime)

        return history_values[-1].copy()

    def get_metadata(self):
        return {
            "model_name": self.config.model_name,
            "is_fitted": self.is_fitted,
        }


def test_default_creation():
    model = DummyRegimeVARModel()

    assert isinstance(
        model.config,
        RegimeVARConfig,
    )
    assert model.is_fitted is False


def test_custom_config():
    config = RegimeVARConfig(
        n_regimes=4,
        lag_order=2,
        n_features=3,
    )

    model = DummyRegimeVARModel(config=config)

    assert model.config is config
    assert model.is_fitted is False


def test_invalid_config_rejected():
    with pytest.raises(TypeError):
        DummyRegimeVARModel(config="invalid")


def test_validate_fitted_rejects_unfitted_model():
    model = DummyRegimeVARModel()

    with pytest.raises(RuntimeError):
        model._validate_fitted()


def test_fit_marks_model_as_fitted():
    model = DummyRegimeVARModel()

    result = model.fit(
        data=np.zeros((3, 1)),
        regimes=np.array([0, 1, 2]),
    )

    assert result is model
    assert model.is_fitted is True


def test_validate_history_accepts_valid_values():
    model = DummyRegimeVARModel(
        RegimeVARConfig(
            lag_order=2,
            n_features=3,
        )
    )

    history = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )

    result = model._validate_history(history)

    assert result.shape == (2, 3)
    assert result.dtype == np.float64
    assert np.allclose(result, history)


def test_validate_history_rejects_non_array():
    model = DummyRegimeVARModel()

    with pytest.raises(TypeError):
        model._validate_history(
            [[1.0], [2.0]]
        )


@pytest.mark.parametrize(
    "history",
    [
        np.array([1.0, 2.0]),
        np.array([[[1.0]]]),
    ],
)
def test_validate_history_rejects_invalid_dimensions(
    history,
):
    model = DummyRegimeVARModel()

    with pytest.raises(ValueError):
        model._validate_history(history)


def test_validate_history_rejects_wrong_shape():
    model = DummyRegimeVARModel(
        RegimeVARConfig(
            lag_order=2,
            n_features=2,
        )
    )

    history = np.array(
        [
            [1.0],
            [2.0],
        ]
    )

    with pytest.raises(ValueError):
        model._validate_history(history)


def test_validate_history_rejects_non_numeric_values():
    model = DummyRegimeVARModel()

    history = np.array(
        [["a"]],
        dtype=object,
    )

    with pytest.raises(TypeError):
        model._validate_history(history)


@pytest.mark.parametrize(
    "value",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_validate_history_rejects_non_finite_values(
    value,
):
    model = DummyRegimeVARModel()

    history = np.array(
        [[value]],
        dtype=np.float64,
    )

    with pytest.raises(ValueError):
        model._validate_history(history)


@pytest.mark.parametrize(
    "regime",
    [
        0,
        1,
        2,
        np.int64(1),
    ],
)
def test_validate_regime_index_accepts_valid_values(
    regime,
):
    model = DummyRegimeVARModel(
        RegimeVARConfig(n_regimes=3)
    )

    result = model._validate_regime_index(regime)

    assert result == int(regime)


@pytest.mark.parametrize(
    "regime",
    [
        -1,
        3,
        100,
    ],
)
def test_validate_regime_index_rejects_out_of_range_values(
    regime,
):
    model = DummyRegimeVARModel(
        RegimeVARConfig(n_regimes=3)
    )

    with pytest.raises(ValueError):
        model._validate_regime_index(regime)


@pytest.mark.parametrize(
    "regime",
    [
        1.0,
        "1",
        None,
        True,
    ],
)
def test_validate_regime_index_rejects_invalid_types(
    regime,
):
    model = DummyRegimeVARModel()

    with pytest.raises(TypeError):
        model._validate_regime_index(regime)


def test_predict_requires_fitted_model():
    model = DummyRegimeVARModel()

    history = np.array([[1.0]])

    with pytest.raises(RuntimeError):
        model.predict(
            history=history,
            regime=0,
        )


def test_predict_returns_last_history_observation():
    model = DummyRegimeVARModel(
        RegimeVARConfig(
            lag_order=2,
            n_features=2,
        )
    )

    model.fit(
        data=np.zeros((3, 2)),
        regimes=np.array([0, 1, 2]),
    )

    history = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    prediction = model.predict(
        history=history,
        regime=1,
    )

    assert prediction.shape == (2,)
    assert np.allclose(
        prediction,
        [3.0, 4.0],
    )


def test_metadata():
    model = DummyRegimeVARModel()

    metadata = model.get_metadata()

    assert metadata["model_name"] == (
        "regime_switching_var"
    )
    assert metadata["is_fitted"] is False