import pytest
import torch
from torch import nn

from src.models.foundation.adapter import (
    FoundationModelAdapter,
)
from src.models.foundation.config import (
    FoundationModelConfig,
)


@pytest.fixture
def config():
    return FoundationModelConfig(
        model_name="test_adapter",
        context_length=4,
        forecast_horizon=2,
        n_features=3,
        output_dim=1,
    )


@pytest.fixture
def model(config):
    torch.manual_seed(42)

    return FoundationModelAdapter(
        config=config,
        hidden_dim=8,
    )


@pytest.fixture
def x():
    torch.manual_seed(123)

    return torch.randn(10, 4, 3)


@pytest.fixture
def y():
    torch.manual_seed(456)

    return torch.randn(10, 2, 1)


def test_default_creation():
    model = FoundationModelAdapter()

    assert isinstance(
        model.config,
        FoundationModelConfig,
    )

    assert model.hidden_dim == 64
    assert model.is_fitted is False


def test_custom_creation(model, config):
    assert model.config is config
    assert model.hidden_dim == 8
    assert isinstance(model.network, nn.Sequential)


def test_network_output_shape(model, x):
    output = model._forward(x)

    assert output.shape == (10, 2, 1)


@pytest.mark.parametrize(
    "hidden_dim",
    [
        0,
        -1,
    ],
)
def test_invalid_hidden_dim_values_rejected(
    config,
    hidden_dim,
):
    with pytest.raises(ValueError):
        FoundationModelAdapter(
            config=config,
            hidden_dim=hidden_dim,
        )


@pytest.mark.parametrize(
    "hidden_dim",
    [
        1.5,
        "8",
        None,
        True,
    ],
)
def test_invalid_hidden_dim_types_rejected(
    config,
    hidden_dim,
):
    with pytest.raises(TypeError):
        FoundationModelAdapter(
            config=config,
            hidden_dim=hidden_dim,
        )


def test_predict_before_fit_rejected(model, x):
    with pytest.raises(RuntimeError):
        model.predict(x)


def test_fit_returns_self(model, x, y):
    result = model.fit(
        x,
        y,
        epochs=2,
    )

    assert result is model
    assert model.is_fitted is True


def test_fit_updates_network_parameters(model, x, y):
    before = [
        parameter.detach().clone()
        for parameter in model.network.parameters()
    ]

    model.fit(
        x,
        y,
        epochs=2,
    )

    after = list(model.network.parameters())

    assert any(
        not torch.allclose(
            previous,
            current,
        )
        for previous, current in zip(
            before,
            after,
        )
    )


def test_predict_shape_after_fit(model, x, y):
    model.fit(
        x,
        y,
        epochs=2,
    )

    predictions = model.predict(x)

    assert predictions.shape == (10, 2, 1)
    assert predictions.dtype == torch.float32


def test_predict_is_deterministic_after_fit(
    model,
    x,
    y,
):
    model.fit(
        x,
        y,
        epochs=2,
    )

    first = model.predict(x)
    second = model.predict(x)

    assert torch.allclose(first, second)


@pytest.mark.parametrize(
    "epochs",
    [
        0,
        -1,
    ],
)
def test_invalid_epochs_values_rejected(
    model,
    x,
    y,
    epochs,
):
    with pytest.raises(ValueError):
        model.fit(
            x,
            y,
            epochs=epochs,
        )


@pytest.mark.parametrize(
    "epochs",
    [
        1.5,
        "2",
        None,
        True,
    ],
)
def test_invalid_epochs_types_rejected(
    model,
    x,
    y,
    epochs,
):
    with pytest.raises(TypeError):
        model.fit(
            x,
            y,
            epochs=epochs,
        )


@pytest.mark.parametrize(
    "learning_rate",
    [
        0,
        -0.01,
    ],
)
def test_invalid_learning_rate_values_rejected(
    model,
    x,
    y,
    learning_rate,
):
    with pytest.raises(ValueError):
        model.fit(
            x,
            y,
            learning_rate=learning_rate,
        )


@pytest.mark.parametrize(
    "learning_rate",
    [
        "0.01",
        None,
        True,
    ],
)
def test_invalid_learning_rate_types_rejected(
    model,
    x,
    y,
    learning_rate,
):
    with pytest.raises(TypeError):
        model.fit(
            x,
            y,
            learning_rate=learning_rate,
        )


def test_fit_rejects_invalid_feature_shape(model, y):
    invalid_x = torch.randn(10, 3)

    with pytest.raises(ValueError):
        model.fit(
            invalid_x,
            y,
        )


def test_fit_rejects_invalid_target_shape(model, x):
    invalid_y = torch.randn(10, 2)

    with pytest.raises(ValueError):
        model.fit(
            x,
            invalid_y,
        )


def test_predict_rejects_invalid_feature_count(
    model,
    y,
):
    model.fit(
        torch.randn(10, 4, 3),
        y,
        epochs=1,
    )

    invalid_x = torch.randn(10, 4, 2)

    with pytest.raises(ValueError):
        model.predict(invalid_x)


def test_metadata_before_fit(model):
    metadata = model.get_metadata()

    assert metadata["model_name"] == "test_adapter"
    assert metadata["context_length"] == 4
    assert metadata["forecast_horizon"] == 2
    assert metadata["n_features"] == 3
    assert metadata["output_dim"] == 1
    assert metadata["is_fitted"] is False
    assert metadata["hidden_dim"] == 8
    assert (
        metadata["adapter_type"]
        == "FoundationModelAdapter"
    )


def test_metadata_after_fit(model, x, y):
    model.fit(
        x,
        y,
        epochs=1,
    )

    metadata = model.get_metadata()

    assert metadata["is_fitted"] is True