import pytest
import torch
from torch import nn

from src.models.bayesian.config import BayesianModelConfig
from src.models.bayesian.network import BayesianRegimeNetwork


@pytest.fixture
def config():
    return BayesianModelConfig(
        n_regimes=3,
        hidden_dims=(16, 8),
        dropout_rate=0.2,
    )


@pytest.fixture
def network(config):
    torch.manual_seed(42)

    return BayesianRegimeNetwork(
        input_dim=5,
        config=config,
    )


@pytest.fixture
def sample_input():
    torch.manual_seed(42)

    return torch.randn(
        10,
        5,
    )


def test_default_config_creation():
    network = BayesianRegimeNetwork(
        input_dim=4,
    )

    assert network.input_dim == 4
    assert network.n_regimes == 3
    assert isinstance(
        network.config,
        BayesianModelConfig,
    )


def test_custom_network_creation(
    network,
    config,
):
    assert network.input_dim == 5
    assert network.n_regimes == 3
    assert network.config is config


@pytest.mark.parametrize(
    "input_dim",
    [
        0,
        -1,
    ],
)
def test_invalid_input_dim_value_rejected(
    input_dim,
):
    with pytest.raises(ValueError):
        BayesianRegimeNetwork(
            input_dim=input_dim,
        )


@pytest.mark.parametrize(
    "input_dim",
    [
        5.0,
        "5",
        None,
        True,
    ],
)
def test_invalid_input_dim_type_rejected(
    input_dim,
):
    with pytest.raises(TypeError):
        BayesianRegimeNetwork(
            input_dim=input_dim,
        )


def test_invalid_config_type_rejected():
    with pytest.raises(TypeError):
        BayesianRegimeNetwork(
            input_dim=5,
            config={},
        )


def test_hidden_layer_structure(
    network,
):
    layers = list(network.hidden_layers)

    assert len(layers) == 6

    assert isinstance(
        layers[0],
        nn.Linear,
    )
    assert isinstance(
        layers[1],
        nn.ReLU,
    )
    assert isinstance(
        layers[2],
        nn.Dropout,
    )

    assert layers[0].in_features == 5
    assert layers[0].out_features == 16

    assert layers[3].in_features == 16
    assert layers[3].out_features == 8

    assert isinstance(
        network.output_layer,
        nn.Linear,
    )
    assert network.output_layer.in_features == 8
    assert network.output_layer.out_features == 3


def test_dropout_rates(
    network,
):
    dropout_layers = [
        layer
        for layer in network.hidden_layers
        if isinstance(layer, nn.Dropout)
    ]

    assert len(dropout_layers) == 2

    for layer in dropout_layers:
        assert layer.p == pytest.approx(0.2)


def test_forward_output_shape(
    network,
    sample_input,
):
    output = network(sample_input)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (10, 3)


def test_forward_output_is_logits(
    network,
    sample_input,
):
    output = network(sample_input)

    probabilities = torch.softmax(
        output,
        dim=1,
    )

    assert torch.allclose(
        probabilities.sum(dim=1),
        torch.ones(10),
        atol=1e-6,
    )


def test_forward_converts_input_to_float(
    network,
):
    x = torch.ones(
        4,
        5,
        dtype=torch.int64,
    )

    output = network(x)

    assert output.dtype == torch.float32


def test_forward_rejects_non_tensor(
    network,
):
    with pytest.raises(TypeError):
        network(
            [
                [1, 2, 3, 4, 5],
            ]
        )


@pytest.mark.parametrize(
    "shape",
    [
        (5,),
        (2, 3, 5),
    ],
)
def test_forward_rejects_invalid_dimensions(
    network,
    shape,
):
    x = torch.randn(*shape)

    with pytest.raises(ValueError):
        network(x)


@pytest.mark.parametrize(
    "feature_count",
    [
        4,
        6,
    ],
)
def test_forward_rejects_wrong_feature_count(
    network,
    feature_count,
):
    x = torch.randn(
        5,
        feature_count,
    )

    with pytest.raises(ValueError):
        network(x)


def test_predict_proba_shape_and_normalization(
    network,
    sample_input,
):
    probabilities = network.predict_proba(
        sample_input
    )

    assert probabilities.shape == (10, 3)

    assert torch.all(
        probabilities >= 0
    )

    assert torch.all(
        probabilities <= 1
    )

    assert torch.allclose(
        probabilities.sum(dim=1),
        torch.ones(10),
        atol=1e-6,
    )


def test_predict_proba_restores_training_mode(
    network,
    sample_input,
):
    network.train()

    network.predict_proba(sample_input)

    assert network.training is True

    network.eval()

    network.predict_proba(sample_input)

    assert network.training is False


def test_predict_shape_and_range(
    network,
    sample_input,
):
    predictions = network.predict(
        sample_input
    )

    assert predictions.shape == (10,)
    assert predictions.dtype == torch.int64

    assert torch.all(predictions >= 0)
    assert torch.all(predictions < 3)


def test_predict_matches_probability_argmax(
    network,
    sample_input,
):
    network.eval()

    probabilities = network.predict_proba(
        sample_input
    )

    predictions = network.predict(
        sample_input
    )

    expected = torch.argmax(
        probabilities,
        dim=1,
    )

    assert torch.equal(
        predictions,
        expected,
    )


def test_predict_proba_is_deterministic_in_eval_mode(
    network,
    sample_input,
):
    network.eval()

    probabilities_one = network.predict_proba(
        sample_input
    )

    probabilities_two = network.predict_proba(
        sample_input
    )

    assert torch.allclose(
        probabilities_one,
        probabilities_two,
    )


def test_dropout_creates_stochastic_outputs_in_training_mode(
    network,
    sample_input,
):
    network.train()

    output_one = network(sample_input)
    output_two = network(sample_input)

    assert not torch.allclose(
        output_one,
        output_two,
    )


def test_network_parameters_are_trainable(
    network,
):
    parameters = list(network.parameters())

    assert len(parameters) > 0

    assert all(
        parameter.requires_grad
        for parameter in parameters
    )


def test_output_layer_uses_last_hidden_dimension():
    config = BayesianModelConfig(
        n_regimes=4,
        hidden_dims=(32, 16, 8),
    )

    network = BayesianRegimeNetwork(
        input_dim=10,
        config=config,
    )

    assert (
        network.output_layer.in_features == 8
    )
    assert (
        network.output_layer.out_features == 4
    )