import numpy as np
import pytest

from src.models.regime_var.config import RegimeVARConfig
from src.models.regime_var.model import RegimeSwitchingVAR


def make_training_data():
    data = np.array(
        [
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
            [6.0],
            [7.0],
            [8.0],
        ],
        dtype=np.float64,
    )

    regimes = np.array(
        [0, 0, 0, 1, 1, 1, 2, 2],
        dtype=np.int64,
    )

    return data, regimes


def test_default_creation():
    model = RegimeSwitchingVAR()

    assert model.is_fitted is False
    assert isinstance(model.config, RegimeVARConfig)


def test_custom_creation():
    config = RegimeVARConfig(
        n_regimes=4,
        lag_order=2,
        n_features=3,
        ridge_alpha=0.1,
    )

    model = RegimeSwitchingVAR(config=config)

    assert model.config is config
    assert model.is_fitted is False


def test_fit_returns_self():
    data, regimes = make_training_data()

    model = RegimeSwitchingVAR(
        RegimeVARConfig(
            n_regimes=3,
            lag_order=1,
            n_features=1,
        )
    )

    result = model.fit(data, regimes)

    assert result is model
    assert model.is_fitted is True


def test_fitted_regimes():
    data, regimes = make_training_data()

    model = RegimeSwitchingVAR(
        RegimeVARConfig(
            n_regimes=3,
            lag_order=1,
            n_features=1,
        )
    )

    model.fit(data, regimes)

    assert model.get_metadata()["fitted_regimes"] == [
        0,
        1,
        2,
    ]


def test_sample_counts():
    data, regimes = make_training_data()

    model = RegimeSwitchingVAR(
        RegimeVARConfig(
            n_regimes=3,
            lag_order=1,
            n_features=1,
        )
    )

    model.fit(data, regimes)

    counts = model.get_metadata()["sample_counts"]

    assert counts == {
        0: 2,
        1: 3,
        2: 2,
    }


def test_predict_before_fit_rejected():
    model = RegimeSwitchingVAR()

    with pytest.raises(RuntimeError):
        model.predict(
            history=np.array([[1.0]]),
            regime=0,
        )


@pytest.mark.parametrize(
    "regime",
    [
        0,
        1,
        2,
    ],
)
def test_predict_shape(regime):
    data, regimes = make_training_data()

    model = RegimeSwitchingVAR(
        RegimeVARConfig(
            n_regimes=3,
            lag_order=1,
            n_features=1,
        )
    )

    model.fit(data, regimes)

    prediction = model.predict(
        history=np.array([[2.0]]),
        regime=regime,
    )

    assert prediction.shape == (1,)


def test_predict_values_for_linear_data():
    data = np.arange(
        1,
        11,
        dtype=np.float64,
    ).reshape(-1, 1)

    regimes = np.zeros(
        10,
        dtype=np.int64,
    )

    model = RegimeSwitchingVAR(
        RegimeVARConfig(
            n_regimes=1,
            lag_order=1,
            n_features=1,
        )
    )

    model.fit(data, regimes)

    prediction = model.predict(
        history=np.array([[10.0]]),
        regime=0,
    )

    assert np.allclose(
        prediction,
        [11.0],
        atol=1e-8,
    )


def test_get_regime_parameters():
    data, regimes = make_training_data()

    model = RegimeSwitchingVAR(
        RegimeVARConfig(
            n_regimes=3,
            lag_order=1,
            n_features=1,
        )
    )

    model.fit(data, regimes)

    parameters = model.get_regime_parameters(1)

    assert parameters["regime"] == 1
    assert parameters["coefficients"].shape == (1, 1)
    assert parameters["intercept"].shape == (1,)
    assert parameters["sample_count"] == 3


def test_get_parameters_before_fit_rejected():
    model = RegimeSwitchingVAR()

    with pytest.raises(RuntimeError):
        model.get_regime_parameters(0)


def test_unfitted_regime_rejected():
    data = np.array(
        [[1.0], [2.0], [3.0]],
    )

    regimes = np.array(
        [0, 0, 0],
        dtype=np.int64,
    )

    model = RegimeSwitchingVAR(
        RegimeVARConfig(
            n_regimes=2,
            lag_order=1,
            n_features=1,
        )
    )

    model.fit(data, regimes)

    with pytest.raises(ValueError):
        model.predict(
            history=np.array([[2.0]]),
            regime=1,
        )


def test_metadata_before_fit():
    model = RegimeSwitchingVAR()

    metadata = model.get_metadata()

    assert metadata["is_fitted"] is False
    assert metadata["fitted_regimes"] == []
    assert metadata["sample_counts"] == {}


def test_metadata_after_fit():
    data, regimes = make_training_data()

    model = RegimeSwitchingVAR(
        RegimeVARConfig(
            n_regimes=3,
            lag_order=1,
            n_features=1,
        )
    )

    model.fit(data, regimes)

    metadata = model.get_metadata()

    assert metadata["is_fitted"] is True
    assert metadata["n_regimes"] == 3
    assert metadata["lag_order"] == 1
    assert metadata["n_features"] == 1


def test_fit_rejects_missing_regimes():
    model = RegimeSwitchingVAR()

    data = np.array(
        [[1.0], [2.0], [3.0]],
    )

    with pytest.raises(ValueError):
        model.fit(
            data=data,
            regimes=None,
        )


def test_fit_rejects_wrong_regime_length():
    model = RegimeSwitchingVAR()

    data = np.array(
        [[1.0], [2.0], [3.0]],
    )

    with pytest.raises(ValueError):
        model.fit(
            data=data,
            regimes=np.array([0, 1]),
        )


def test_fit_rejects_wrong_feature_count():
    model = RegimeSwitchingVAR(
        RegimeVARConfig(
            n_features=2,
        )
    )

    data = np.array(
        [[1.0], [2.0], [3.0]],
    )

    regimes = np.array([0, 0, 0])

    with pytest.raises(ValueError):
        model.fit(data, regimes)


def test_predict_rejects_invalid_history_shape():
    data, regimes = make_training_data()

    model = RegimeSwitchingVAR(
        RegimeVARConfig(
            n_regimes=3,
            lag_order=2,
            n_features=1,
        )
    )

    model.fit(data, regimes)

    with pytest.raises(ValueError):
        model.predict(
            history=np.array([[1.0]]),
            regime=0,
        )


def test_predict_rejects_non_array_history():
    data, regimes = make_training_data()

    model = RegimeSwitchingVAR(
        RegimeVARConfig(
            n_regimes=3,
            lag_order=1,
            n_features=1,
        )
    )

    model.fit(data, regimes)

    with pytest.raises(TypeError):
        model.predict(
            history=[[1.0]],
            regime=0,
        )


def test_predict_rejects_invalid_regime():
    data, regimes = make_training_data()

    model = RegimeSwitchingVAR(
        RegimeVARConfig(
            n_regimes=3,
            lag_order=1,
            n_features=1,
        )
    )

    model.fit(data, regimes)

    with pytest.raises(ValueError):
        model.predict(
            history=np.array([[1.0]]),
            regime=3,
        )


def test_predict_rejects_invalid_regime_type():
    data, regimes = make_training_data()

    model = RegimeSwitchingVAR(
        RegimeVARConfig(
            n_regimes=3,
            lag_order=1,
            n_features=1,
        )
    )

    model.fit(data, regimes)

    with pytest.raises(TypeError):
        model.predict(
            history=np.array([[1.0]]),
            regime="1",
        )


def test_ridge_regularization():
    data = np.arange(
        1,
        8,
        dtype=np.float64,
    ).reshape(-1, 1)

    regimes = np.zeros(
        7,
        dtype=np.int64,
    )

    model = RegimeSwitchingVAR(
        RegimeVARConfig(
            n_regimes=1,
            lag_order=1,
            n_features=1,
            ridge_alpha=1.0,
        )
    )

    model.fit(data, regimes)

    prediction = model.predict(
        history=np.array([[7.0]]),
        regime=0,
    )

    assert prediction.shape == (1,)
    assert np.isfinite(prediction).all()


def test_no_intercept():
    data = np.arange(
        1,
        8,
        dtype=np.float64,
    ).reshape(-1, 1)

    regimes = np.zeros(
        7,
        dtype=np.int64,
    )

    model = RegimeSwitchingVAR(
        RegimeVARConfig(
            n_regimes=1,
            lag_order=1,
            n_features=1,
            include_intercept=False,
        )
    )

    model.fit(data, regimes)

    parameters = model.get_regime_parameters(0)

    assert np.allclose(
        parameters["intercept"],
        [0.0],
    )