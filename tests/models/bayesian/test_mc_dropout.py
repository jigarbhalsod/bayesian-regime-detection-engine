import pytest
import torch
from torch import nn

from src.models.bayesian.config import BayesianModelConfig
from src.models.bayesian.mc_dropout import MCDropoutInference
from src.models.bayesian.network import BayesianRegimeNetwork


@pytest.fixture
def config():
    return BayesianModelConfig(
        n_regimes=3,
        hidden_dims=(16, 8),
        dropout_rate=0.5,
        mc_samples=10,
    )


@pytest.fixture
def network(config):
    torch.manual_seed(42)

    return BayesianRegimeNetwork(
        input_dim=5,
        config=config,
    )


@pytest.fixture
def inference(network):
    return MCDropoutInference(network)


@pytest.fixture
def sample_input():
    torch.manual_seed(123)

    return torch.randn(6, 5)


def test_creation_uses_config_mc_samples(
    inference,
):
    assert inference.mc_samples == 10


def test_custom_mc_samples(
    network,
):
    inference = MCDropoutInference(
        network,
        mc_samples=25,
    )

    assert inference.mc_samples == 25


def test_invalid_network_rejected():
    with pytest.raises(TypeError):
        MCDropoutInference(
            network={},
        )


@pytest.mark.parametrize(
    "mc_samples",
    [
        0,
        -1,
    ],
)
def test_invalid_mc_samples_value_rejected(
    network,
    mc_samples,
):
    with pytest.raises(ValueError):
        MCDropoutInference(
            network,
            mc_samples=mc_samples,
        )


@pytest.mark.parametrize(
    "mc_samples",
    [
        1.5,
        "10",
        True,
    ],
)
def test_invalid_mc_samples_type_rejected(
    network,
    mc_samples,
):
    with pytest.raises(TypeError):
        MCDropoutInference(
            network,
            mc_samples=mc_samples,
        )


def test_sample_probabilities_shape(
    inference,
    sample_input,
):
    samples = inference.sample_probabilities(
        sample_input
    )

    assert samples.shape == (10, 6, 3)


def test_sample_probabilities_are_normalized(
    inference,
    sample_input,
):
    samples = inference.sample_probabilities(
        sample_input
    )

    row_sums = samples.sum(dim=2)

    assert torch.allclose(
        row_sums,
        torch.ones_like(row_sums),
        atol=1e-6,
    )


def test_sample_probabilities_are_in_probability_range(
    inference,
    sample_input,
):
    samples = inference.sample_probabilities(
        sample_input
    )

    assert torch.all(samples >= 0)
    assert torch.all(samples <= 1)


def test_mc_dropout_produces_stochastic_samples(
    inference,
    sample_input,
):
    samples = inference.sample_probabilities(
        sample_input
    )

    assert not torch.allclose(
        samples[0],
        samples[1],
    )


def test_network_training_mode_restored_when_training(
    inference,
    network,
    sample_input,
):
    network.train()

    inference.sample_probabilities(sample_input)

    assert network.training is True


def test_network_training_mode_restored_when_eval(
    inference,
    network,
    sample_input,
):
    network.eval()

    inference.sample_probabilities(sample_input)

    assert network.training is False


def test_dropout_states_restored(
    inference,
    network,
    sample_input,
):
    network.eval()

    dropout_layers = [
        module
        for module in network.modules()
        if isinstance(module, nn.Dropout)
    ]

    assert all(
        layer.training is False
        for layer in dropout_layers
    )

    inference.sample_probabilities(sample_input)

    assert all(
        layer.training is False
        for layer in dropout_layers
    )


def test_mean_probabilities_shape_and_normalization(
    inference,
    sample_input,
):
    probabilities = inference.mean_probabilities(
        sample_input
    )

    assert probabilities.shape == (6, 3)

    assert torch.allclose(
        probabilities.sum(dim=1),
        torch.ones(6),
        atol=1e-6,
    )


def test_predict_shape_and_range(
    inference,
    sample_input,
):
    predictions = inference.predict(
        sample_input
    )

    assert predictions.shape == (6,)
    assert predictions.dtype == torch.int64
    assert torch.all(predictions >= 0)
    assert torch.all(predictions < 3)


def test_confidence_shape_and_range(
    inference,
    sample_input,
):
    confidence = inference.confidence(
        sample_input
    )

    assert confidence.shape == (6,)
    assert torch.all(confidence >= 0)
    assert torch.all(confidence <= 1)


def test_probability_std_shape_and_non_negative(
    inference,
    sample_input,
):
    std = inference.probability_std(
        sample_input
    )

    assert std.shape == (6, 3)
    assert torch.all(std >= 0)


def test_input_validation_rejects_non_tensor(
    inference,
):
    with pytest.raises(TypeError):
        inference.sample_probabilities(
            [[1, 2, 3, 4, 5]]
        )


@pytest.mark.parametrize(
    "shape",
    [
        (5,),
        (2, 3, 5),
    ],
)
def test_input_validation_rejects_invalid_dimensions(
    inference,
    shape,
):
    x = torch.randn(*shape)

    with pytest.raises(ValueError):
        inference.sample_probabilities(x)


@pytest.mark.parametrize(
    "feature_count",
    [
        4,
        6,
    ],
)
def test_input_validation_rejects_wrong_feature_count(
    inference,
    feature_count,
):
    x = torch.randn(
        6,
        feature_count,
    )

    with pytest.raises(ValueError):
        inference.sample_probabilities(x)


def test_integer_input_is_converted_to_float(
    inference,
):
    x = torch.ones(
        4,
        5,
        dtype=torch.int64,
    )

    samples = inference.sample_probabilities(x)

    assert samples.dtype == torch.float32


def test_single_mc_sample_supported(
    network,
    sample_input,
):
    inference = MCDropoutInference(
        network,
        mc_samples=1,
    )

    samples = inference.sample_probabilities(
        sample_input
    )

    assert samples.shape == (1, 6, 3)

def test_none_mc_samples_uses_config_default(
    network,
):
    inference = MCDropoutInference(
        network,
        mc_samples=None,
    )

    assert (
        inference.mc_samples
        == network.config.mc_samples
    )