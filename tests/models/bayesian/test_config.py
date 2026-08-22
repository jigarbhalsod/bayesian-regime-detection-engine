import pytest

from src.models.bayesian.config import BayesianModelConfig


def test_default_config():
    config = BayesianModelConfig()

    assert config.model_name == "bayesian_model"
    assert config.n_features == 1
    assert config.n_outputs == 1
    assert config.prior_mean == 0.0
    assert config.prior_std == 1.0
    assert config.observation_noise == 1.0

    assert config.n_regimes == 3
    assert config.hidden_dims == (32, 16)
    assert config.dropout_rate == 0.2
    assert config.mc_samples == 20

    assert config.random_seed is None


def test_custom_config():
    config = BayesianModelConfig(
        model_name="bayesian_regime_model",
        n_features=5,
        n_outputs=2,
        prior_mean=1.0,
        prior_std=2.0,
        observation_noise=0.5,
        n_regimes=4,
        hidden_dims=(64, 32),
        dropout_rate=0.3,
        mc_samples=50,
        random_seed=42,
    )

    assert config.model_name == "bayesian_regime_model"
    assert config.n_features == 5
    assert config.n_outputs == 2
    assert config.prior_mean == 1.0
    assert config.prior_std == 2.0
    assert config.observation_noise == 0.5

    assert config.n_regimes == 4
    assert config.hidden_dims == (64, 32)
    assert config.dropout_rate == 0.3
    assert config.mc_samples == 50

    assert config.random_seed == 42


@pytest.mark.parametrize(
    ("model_name", "exception"),
    [
        (123, TypeError),
        ("", ValueError),
        ("   ", ValueError),
    ],
)
def test_invalid_model_name(model_name, exception):
    with pytest.raises(exception):
        BayesianModelConfig(model_name=model_name)


@pytest.mark.parametrize(
    "n_features",
    [
        0,
        -1,
        1.5,
        True,
        "3",
    ],
)
def test_invalid_n_features(n_features):
    with pytest.raises((TypeError, ValueError)):
        BayesianModelConfig(n_features=n_features)


@pytest.mark.parametrize(
    "n_outputs",
    [
        0,
        -1,
        1.5,
        True,
        "2",
    ],
)
def test_invalid_n_outputs(n_outputs):
    with pytest.raises((TypeError, ValueError)):
        BayesianModelConfig(n_outputs=n_outputs)


@pytest.mark.parametrize(
    "prior_mean",
    [
        "invalid",
        True,
    ],
)
def test_invalid_prior_mean(prior_mean):
    with pytest.raises(TypeError):
        BayesianModelConfig(prior_mean=prior_mean)


@pytest.mark.parametrize(
    "prior_std",
    [
        0,
        -1,
        "invalid",
        True,
    ],
)
def test_invalid_prior_std(prior_std):
    with pytest.raises((TypeError, ValueError)):
        BayesianModelConfig(prior_std=prior_std)


@pytest.mark.parametrize(
    "observation_noise",
    [
        0,
        -1,
        "invalid",
        True,
    ],
)
def test_invalid_observation_noise(observation_noise):
    with pytest.raises((TypeError, ValueError)):
        BayesianModelConfig(
            observation_noise=observation_noise,
        )


@pytest.mark.parametrize(
    "n_regimes",
    [
        0,
        -1,
        1.5,
        True,
        "3",
    ],
)
def test_invalid_n_regimes(n_regimes):
    with pytest.raises((TypeError, ValueError)):
        BayesianModelConfig(n_regimes=n_regimes)


@pytest.mark.parametrize(
    "hidden_dims",
    [
        (),
        [],
        (32, 0),
        (32, -1),
        (32, 1.5),
        (32, True),
    ],
)
def test_invalid_hidden_dims(hidden_dims):
    with pytest.raises((TypeError, ValueError)):
        BayesianModelConfig(hidden_dims=hidden_dims)


@pytest.mark.parametrize(
    "dropout_rate",
    [
        -0.1,
        1.0,
        1.5,
        "invalid",
        True,
    ],
)
def test_invalid_dropout_rate(dropout_rate):
    with pytest.raises((TypeError, ValueError)):
        BayesianModelConfig(
            dropout_rate=dropout_rate,
        )


@pytest.mark.parametrize(
    "mc_samples",
    [
        0,
        -1,
        1.5,
        True,
        "20",
    ],
)
def test_invalid_mc_samples(mc_samples):
    with pytest.raises((TypeError, ValueError)):
        BayesianModelConfig(mc_samples=mc_samples)


@pytest.mark.parametrize(
    "random_seed",
    [
        1.5,
        True,
        "42",
    ],
)
def test_invalid_random_seed(random_seed):
    with pytest.raises(TypeError):
        BayesianModelConfig(random_seed=random_seed)


def test_to_dict():
    config = BayesianModelConfig(
        model_name="test_model",
        n_features=3,
        n_outputs=2,
        prior_mean=0.5,
        prior_std=1.5,
        observation_noise=0.2,
        n_regimes=4,
        hidden_dims=(16, 8),
        dropout_rate=0.3,
        mc_samples=10,
        random_seed=7,
    )

    result = config.to_dict()

    assert result == {
        "model_name": "test_model",
        "n_features": 3,
        "n_outputs": 2,
        "prior_mean": 0.5,
        "prior_std": 1.5,
        "observation_noise": 0.2,
        "n_regimes": 4,
        "hidden_dims": (16, 8),
        "dropout_rate": 0.3,
        "mc_samples": 10,
        "random_seed": 7,
    }


def test_config_is_frozen():
    config = BayesianModelConfig()

    with pytest.raises(
        (AttributeError, TypeError),
    ):
        config.n_features = 10