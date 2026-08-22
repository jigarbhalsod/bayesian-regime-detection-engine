import pytest

from src.regime.config import RegimeConfig


def test_default_config_values():
    config = RegimeConfig()

    assert config.n_regimes == 3
    assert config.feature_columns == ()
    assert config.lookback_window is None
    assert config.min_samples == 30
    assert config.random_state == 42
    assert config.model_parameters == {}


def test_custom_config_values():
    config = RegimeConfig(
        n_regimes=4,
        feature_columns=("return", "volatility"),
        lookback_window=60,
        min_samples=100,
        random_state=123,
        model_parameters={"n_iter": 200},
    )

    assert config.n_regimes == 4
    assert config.feature_columns == ("return", "volatility")
    assert config.lookback_window == 60
    assert config.min_samples == 100
    assert config.random_state == 123
    assert config.model_parameters == {"n_iter": 200}


def test_config_is_immutable():
    config = RegimeConfig()

    with pytest.raises(Exception):
        config.n_regimes = 5


def test_default_feature_columns_are_empty():
    first = RegimeConfig()
    second = RegimeConfig()

    assert first.feature_columns == ()
    assert second.feature_columns == ()


def test_default_model_parameters_are_independent():
    first = RegimeConfig()
    second = RegimeConfig()

    assert first.model_parameters == {}
    assert second.model_parameters == {}

    assert first.model_parameters is not second.model_parameters