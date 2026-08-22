import pytest

from src.models.regime_var.config import RegimeVARConfig


def test_default_creation():
    config = RegimeVARConfig()

    assert config.model_name == "regime_switching_var"
    assert config.n_regimes == 3
    assert config.lag_order == 1
    assert config.n_features == 1
    assert config.include_intercept is True
    assert config.ridge_alpha == 0.0


def test_custom_creation():
    config = RegimeVARConfig(
        model_name="custom_regime_var",
        n_regimes=4,
        lag_order=2,
        n_features=3,
        include_intercept=False,
        ridge_alpha=0.25,
    )

    assert config.model_name == "custom_regime_var"
    assert config.n_regimes == 4
    assert config.lag_order == 2
    assert config.n_features == 3
    assert config.include_intercept is False
    assert config.ridge_alpha == 0.25


@pytest.mark.parametrize(
    "value",
    [
        "",
        "   ",
    ],
)
def test_empty_model_name_rejected(value):
    with pytest.raises(ValueError):
        RegimeVARConfig(model_name=value)


@pytest.mark.parametrize(
    "value",
    [
        123,
        None,
        True,
    ],
)
def test_invalid_model_name_type_rejected(value):
    with pytest.raises(TypeError):
        RegimeVARConfig(model_name=value)


@pytest.mark.parametrize(
    "field_name",
    [
        "n_regimes",
        "lag_order",
        "n_features",
    ],
)
@pytest.mark.parametrize(
    "value",
    [
        0,
        -1,
        -10,
    ],
)
def test_non_positive_integer_values_rejected(
    field_name,
    value,
):
    with pytest.raises(ValueError):
        RegimeVARConfig(
            **{
                field_name: value,
            }
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "n_regimes",
        "lag_order",
        "n_features",
    ],
)
@pytest.mark.parametrize(
    "value",
    [
        1.0,
        "1",
        None,
        True,
    ],
)
def test_invalid_integer_types_rejected(
    field_name,
    value,
):
    with pytest.raises(TypeError):
        RegimeVARConfig(
            **{
                field_name: value,
            }
        )


@pytest.mark.parametrize(
    "value",
    [
        0,
        1,
        -1,
        "true",
        None,
    ],
)
def test_invalid_include_intercept_rejected(value):
    with pytest.raises(TypeError):
        RegimeVARConfig(
            include_intercept=value
        )


@pytest.mark.parametrize(
    "value",
    [
        -0.1,
        -1.0,
        -100,
    ],
)
def test_negative_ridge_alpha_rejected(value):
    with pytest.raises(ValueError):
        RegimeVARConfig(
            ridge_alpha=value
        )


@pytest.mark.parametrize(
    "value",
    [
        "0.1",
        None,
        True,
    ],
)
def test_invalid_ridge_alpha_type_rejected(value):
    with pytest.raises(TypeError):
        RegimeVARConfig(
            ridge_alpha=value
        )


@pytest.mark.parametrize(
    "value",
    [
        0,
        0.0,
        0.5,
        1,
        10.0,
    ],
)
def test_valid_ridge_alpha_accepted(value):
    config = RegimeVARConfig(
        ridge_alpha=value
    )

    assert config.ridge_alpha == float(value)


def test_model_name_is_stripped():
    config = RegimeVARConfig(
        model_name="  regime_var_test  "
    )

    assert config.model_name == "regime_var_test"