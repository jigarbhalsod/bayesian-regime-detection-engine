import pytest

from src.models.foundation.config import (
    FoundationModelConfig,
)


def test_default_config():
    config = FoundationModelConfig()

    assert config.model_name == "foundation_model"
    assert config.context_length == 60
    assert config.forecast_horizon == 5
    assert config.n_features == 1
    assert config.output_dim == 1


def test_custom_config():
    config = FoundationModelConfig(
        model_name="Test_Model",
        context_length=30,
        forecast_horizon=10,
        n_features=4,
        output_dim=3,
    )

    assert config.model_name == "test_model"
    assert config.context_length == 30
    assert config.forecast_horizon == 10
    assert config.n_features == 4
    assert config.output_dim == 3


@pytest.mark.parametrize(
    "model_name",
    [
        "",
        "   ",
    ],
)
def test_empty_model_name_rejected(model_name):
    with pytest.raises(ValueError):
        FoundationModelConfig(
            model_name=model_name
        )


@pytest.mark.parametrize(
    "model_name",
    [
        None,
        123,
        True,
    ],
)
def test_invalid_model_name_type_rejected(model_name):
    with pytest.raises(TypeError):
        FoundationModelConfig(
            model_name=model_name
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "context_length",
        "forecast_horizon",
        "n_features",
        "output_dim",
    ],
)
@pytest.mark.parametrize(
    "value",
    [
        0,
        -1,
    ],
)
def test_invalid_positive_integer_values_rejected(
    field_name,
    value,
):
    kwargs = {
        field_name: value,
    }

    with pytest.raises(ValueError):
        FoundationModelConfig(**kwargs)


@pytest.mark.parametrize(
    "field_name",
    [
        "context_length",
        "forecast_horizon",
        "n_features",
        "output_dim",
    ],
)
@pytest.mark.parametrize(
    "value",
    [
        1.5,
        "10",
        None,
        True,
    ],
)
def test_invalid_positive_integer_types_rejected(
    field_name,
    value,
):
    kwargs = {
        field_name: value,
    }

    with pytest.raises(TypeError):
        FoundationModelConfig(**kwargs)