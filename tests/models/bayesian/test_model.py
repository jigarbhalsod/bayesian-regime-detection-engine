import pytest
import torch

from src.models.bayesian.config import BayesianModelConfig
from src.models.bayesian.model import (
    BayesianPrediction,
    BayesianRegimeModel,
)


@pytest.fixture
def config():
    return BayesianModelConfig(
        n_regimes=3,
        hidden_dims=(16, 8),
        dropout_rate=0.3,
        mc_samples=5,
    )


@pytest.fixture
def model(config):
    torch.manual_seed(42)

    return BayesianRegimeModel(
        input_dim=4,
        config=config,
    )


@pytest.fixture
def sample_features():
    torch.manual_seed(123)

    return torch.randn(12, 4)


@pytest.fixture
def sample_targets():
    return torch.tensor(
        [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
        dtype=torch.long,
    )


def test_default_model_creation():
    model = BayesianRegimeModel(input_dim=4)

    assert model.input_dim == 4
    assert isinstance(model.config, BayesianModelConfig)
    assert model.network.input_dim == 4


def test_custom_model_creation(model, config):
    assert model.input_dim == 4
    assert model.config is config
    assert model.network.config is config
    assert model.inference.network is model.network


@pytest.mark.parametrize("input_dim", [0, -1])
def test_invalid_input_dim_value_rejected(input_dim):
    with pytest.raises(ValueError):
        BayesianRegimeModel(input_dim=input_dim)


@pytest.mark.parametrize(
    "input_dim",
    [4.0, "4", None, True],
)
def test_invalid_input_dim_type_rejected(input_dim):
    with pytest.raises(TypeError):
        BayesianRegimeModel(input_dim=input_dim)


def test_invalid_config_type_rejected():
    with pytest.raises(TypeError):
        BayesianRegimeModel(
            input_dim=4,
            config={},
        )


def test_predict_shape_and_range(model, sample_features):
    predictions = model.predict(sample_features)

    assert predictions.shape == (12,)
    assert predictions.dtype == torch.int64
    assert torch.all(predictions >= 0)
    assert torch.all(predictions < 3)


def test_predict_proba_shape_and_normalization(
    model,
    sample_features,
):
    probabilities = model.predict_proba(sample_features)

    assert probabilities.shape == (12, 3)

    assert torch.allclose(
        probabilities.sum(dim=1),
        torch.ones(12),
        atol=1e-6,
    )


def test_predict_with_uncertainty_returns_prediction(
    model,
    sample_features,
):
    result = model.predict_with_uncertainty(
        sample_features
    )

    assert isinstance(result, BayesianPrediction)

    assert result.predictions.shape == (12,)
    assert result.probabilities.shape == (12, 3)
    assert result.confidence.shape == (12,)
    assert (
        result.uncertainty.predictive_entropy.shape
        == (12,)
    )


def test_prediction_matches_probability_argmax(
    model,
    sample_features,
):
    torch.manual_seed(99)

    result = model.predict_with_uncertainty(
        sample_features
    )

    expected = torch.argmax(
        result.probabilities,
        dim=1,
    )

    assert torch.equal(
        result.predictions,
        expected,
    )


def test_confidence_matches_probability_maximum(
    model,
    sample_features,
):
    torch.manual_seed(99)

    result = model.predict_with_uncertainty(
        sample_features
    )

    expected = torch.max(
        result.probabilities,
        dim=1,
    ).values

    assert torch.allclose(
        result.confidence,
        expected,
        atol=1e-6,
    )


def test_fit_returns_self(
    model,
    sample_features,
    sample_targets,
):
    result = model.fit(
        sample_features,
        sample_targets,
        epochs=2,
    )

    assert result is model


def test_fit_updates_network_parameters(
    model,
    sample_features,
    sample_targets,
):
    before = [
        parameter.detach().clone()
        for parameter in model.network.parameters()
    ]

    model.fit(
        sample_features,
        sample_targets,
        epochs=2,
    )

    after = list(
        model.network.parameters()
    )

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


@pytest.mark.parametrize(
    "epochs",
    [0, -1],
)
def test_invalid_epochs_value_rejected(
    model,
    sample_features,
    sample_targets,
    epochs,
):
    with pytest.raises(ValueError):
        model.fit(
            sample_features,
            sample_targets,
            epochs=epochs,
        )


@pytest.mark.parametrize(
    "epochs",
    [1.5, "10", None, True],
)
def test_invalid_epochs_type_rejected(
    model,
    sample_features,
    sample_targets,
    epochs,
):
    with pytest.raises(TypeError):
        model.fit(
            sample_features,
            sample_targets,
            epochs=epochs,
        )


@pytest.mark.parametrize(
    "learning_rate",
    [0, -0.01],
)
def test_invalid_learning_rate_value_rejected(
    model,
    sample_features,
    sample_targets,
    learning_rate,
):
    with pytest.raises(ValueError):
        model.fit(
            sample_features,
            sample_targets,
            learning_rate=learning_rate,
        )


@pytest.mark.parametrize(
    "learning_rate",
    ["0.01", None, True],
)
def test_invalid_learning_rate_type_rejected(
    model,
    sample_features,
    sample_targets,
    learning_rate,
):
    with pytest.raises(TypeError):
        model.fit(
            sample_features,
            sample_targets,
            learning_rate=learning_rate,
        )


def test_fit_rejects_non_tensor_features(
    model,
    sample_targets,
):
    with pytest.raises(TypeError):
        model.fit(
            [[1, 2, 3, 4]],
            sample_targets,
        )


def test_fit_rejects_non_tensor_targets(
    model,
    sample_features,
):
    with pytest.raises(TypeError):
        model.fit(
            sample_features,
            [0] * 12,
        )


@pytest.mark.parametrize(
    "shape",
    [(4,), (2, 3, 4)],
)
def test_feature_dimensions_rejected(
    model,
    shape,
):
    x = torch.randn(*shape)

    with pytest.raises(ValueError):
        model.predict(x)


def test_wrong_feature_count_rejected(model):
    x = torch.randn(5, 3)

    with pytest.raises(ValueError):
        model.predict(x)


def test_empty_feature_batch_rejected(model):
    x = torch.empty(0, 4)

    with pytest.raises(ValueError):
        model.predict(x)


def test_target_dimensions_rejected(
    model,
    sample_features,
):
    y = torch.ones(12, 1, dtype=torch.long)

    with pytest.raises(ValueError):
        model.fit(
            sample_features,
            y,
        )


def test_target_batch_size_mismatch_rejected(
    model,
    sample_features,
):
    y = torch.tensor(
        [0, 1, 2],
        dtype=torch.long,
    )

    with pytest.raises(ValueError):
        model.fit(
            sample_features,
            y,
        )


def test_floating_targets_rejected(
    model,
    sample_features,
):
    y = torch.zeros(12, dtype=torch.float32)

    with pytest.raises(TypeError):
        model.fit(
            sample_features,
            y,
        )


@pytest.mark.parametrize(
    "targets",
    [
        torch.tensor(
            [0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 0, 1]
        ),
        torch.tensor(
            [-1, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1]
        ),
    ],
)
def test_invalid_target_range_rejected(
    model,
    sample_features,
    targets,
):
    with pytest.raises(ValueError):
        model.fit(
            sample_features,
            targets,
        )