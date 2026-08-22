import pytest
import torch
from torch import Tensor

from src.models.foundation.base import (
    BaseFoundationModel,
)
from src.models.foundation.config import (
    FoundationModelConfig,
)


class DummyFoundationModel(BaseFoundationModel):
    """Minimal concrete implementation for testing."""

    def fit(
        self,
        x: Tensor,
        y: Tensor,
        **kwargs,
    ):
        x = self._validate_sequence_input(x)
        self._validate_targets(
            y,
            batch_size=x.shape[0],
        )

        self._is_fitted = True

        return self

    def predict(
        self,
        x: Tensor,
        **kwargs,
    ) -> Tensor:
        x = self._validate_sequence_input(x)

        return torch.zeros(
            (
                x.shape[0],
                self.config.forecast_horizon,
                self.config.output_dim,
            ),
            dtype=torch.float32,
        )


@pytest.fixture
def config():
    return FoundationModelConfig(
        model_name="dummy",
        context_length=4,
        forecast_horizon=2,
        n_features=3,
        output_dim=1,
    )


@pytest.fixture
def model(config):
    return DummyFoundationModel(config)


@pytest.fixture
def x():
    return torch.randn(5, 4, 3)


@pytest.fixture
def y():
    return torch.randn(5, 2, 1)


def test_default_base_config():
    model = DummyFoundationModel()

    assert isinstance(
        model.config,
        FoundationModelConfig,
    )
    assert model.is_fitted is False


def test_custom_config(model, config):
    assert model.config is config
    assert model.is_fitted is False


def test_invalid_config_rejected():
    with pytest.raises(TypeError):
        DummyFoundationModel(config={})


def test_is_fitted_after_fit(model, x, y):
    result = model.fit(x, y)

    assert result is model
    assert model.is_fitted is True


def test_predict_shape(model, x):
    predictions = model.predict(x)

    assert predictions.shape == (5, 2, 1)
    assert predictions.dtype == torch.float32


def test_predict_proba_not_implemented(model, x):
    with pytest.raises(NotImplementedError):
        model.predict_proba(x)


def test_metadata_before_fit(model):
    metadata = model.get_metadata()

    assert metadata == {
        "model_name": "dummy",
        "context_length": 4,
        "forecast_horizon": 2,
        "n_features": 3,
        "output_dim": 1,
        "is_fitted": False,
    }


def test_metadata_after_fit(model, x, y):
    model.fit(x, y)

    metadata = model.get_metadata()

    assert metadata["is_fitted"] is True


def test_non_tensor_sequence_rejected(model):
    with pytest.raises(TypeError):
        model.predict(
            [[[1, 2, 3]]]
        )


@pytest.mark.parametrize(
    "shape",
    [
        (5, 3),
        (5, 4, 3, 1),
    ],
)
def test_invalid_sequence_dimensions_rejected(
    model,
    shape,
):
    x = torch.randn(*shape)

    with pytest.raises(ValueError):
        model.predict(x)


def test_empty_sequence_batch_rejected(model):
    x = torch.empty(0, 4, 3)

    with pytest.raises(ValueError):
        model.predict(x)


def test_wrong_context_length_rejected(model):
    x = torch.randn(5, 5, 3)

    with pytest.raises(ValueError):
        model.predict(x)


def test_wrong_feature_count_rejected(model):
    x = torch.randn(5, 4, 2)

    with pytest.raises(ValueError):
        model.predict(x)


def test_integer_sequence_converted_to_float(model):
    x = torch.ones(
        5,
        4,
        3,
        dtype=torch.int64,
    )

    predictions = model.predict(x)

    assert predictions.dtype == torch.float32


def test_non_tensor_targets_rejected(model, x):
    with pytest.raises(TypeError):
        model.fit(
            x,
            [[[1]]],
        )


@pytest.mark.parametrize(
    "shape",
    [
        (5, 2),
        (5, 2, 1, 1),
    ],
)
def test_invalid_target_dimensions_rejected(
    model,
    x,
    shape,
):
    y = torch.randn(*shape)

    with pytest.raises(ValueError):
        model.fit(x, y)


def test_target_batch_size_mismatch_rejected(
    model,
    x,
):
    y = torch.randn(3, 2, 1)

    with pytest.raises(ValueError):
        model.fit(x, y)


def test_wrong_forecast_horizon_rejected(
    model,
    x,
):
    y = torch.randn(5, 3, 1)

    with pytest.raises(ValueError):
        model.fit(x, y)


def test_wrong_output_dim_rejected(
    model,
    x,
):
    y = torch.randn(5, 2, 2)

    with pytest.raises(ValueError):
        model.fit(x, y)


def test_integer_targets_converted_to_float(
    model,
    x,
):
    y = torch.ones(
        5,
        2,
        1,
        dtype=torch.int64,
    )

    result = model.fit(x, y)

    assert result is model
    assert model.is_fitted is True