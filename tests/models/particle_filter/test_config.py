import pytest

from src.models.particle_filter.config import (
    ParticleFilterConfig,
)


def test_default_config():
    config = ParticleFilterConfig()

    assert config.model_name == "particle_filter"
    assert config.n_particles == 100
    assert config.state_dim == 1
    assert config.observation_dim == 1
    assert config.process_noise == 1.0
    assert config.observation_noise == 1.0
    assert config.resampling_threshold == 0.5
    assert config.random_seed is None


def test_custom_config():
    config = ParticleFilterConfig(
        model_name="custom_particle_filter",
        n_particles=500,
        state_dim=3,
        observation_dim=2,
        process_noise=0.25,
        observation_noise=0.5,
        resampling_threshold=0.75,
        random_seed=42,
    )

    assert config.model_name == "custom_particle_filter"
    assert config.n_particles == 500
    assert config.state_dim == 3
    assert config.observation_dim == 2
    assert config.process_noise == 0.25
    assert config.observation_noise == 0.5
    assert config.resampling_threshold == 0.75
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
        ParticleFilterConfig(
            model_name=model_name
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "n_particles",
        "state_dim",
        "observation_dim",
    ],
)
@pytest.mark.parametrize(
    "value",
    [
        0,
        -1,
    ],
)
def test_invalid_positive_integer_values(
    field_name,
    value,
):
    with pytest.raises(ValueError):
        ParticleFilterConfig(
            **{field_name: value}
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "n_particles",
        "state_dim",
        "observation_dim",
    ],
)
@pytest.mark.parametrize(
    "value",
    [
        1.5,
        "10",
        True,
    ],
)
def test_invalid_positive_integer_types(
    field_name,
    value,
):
    with pytest.raises(TypeError):
        ParticleFilterConfig(
            **{field_name: value}
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "process_noise",
        "observation_noise",
    ],
)
@pytest.mark.parametrize(
    "value",
    [
        0,
        -1,
    ],
)
def test_invalid_noise_values(
    field_name,
    value,
):
    with pytest.raises(ValueError):
        ParticleFilterConfig(
            **{field_name: value}
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "process_noise",
        "observation_noise",
    ],
)
@pytest.mark.parametrize(
    "value",
    [
        "invalid",
        True,
    ],
)
def test_invalid_noise_types(
    field_name,
    value,
):
    with pytest.raises(TypeError):
        ParticleFilterConfig(
            **{field_name: value}
        )


@pytest.mark.parametrize(
    "value",
    [
        0,
        -0.1,
        1.1,
        2,
    ],
)
def test_invalid_resampling_threshold_values(value):
    with pytest.raises(ValueError):
        ParticleFilterConfig(
            resampling_threshold=value
        )


@pytest.mark.parametrize(
    "value",
    [
        "0.5",
        None,
        True,
    ],
)
def test_invalid_resampling_threshold_types(value):
    with pytest.raises(TypeError):
        ParticleFilterConfig(
            resampling_threshold=value
        )


@pytest.mark.parametrize(
    "random_seed",
    [
        1.5,
        "42",
        True,
    ],
)
def test_invalid_random_seed(random_seed):
    with pytest.raises(TypeError):
        ParticleFilterConfig(
            random_seed=random_seed
        )


def test_to_dict():
    config = ParticleFilterConfig(
        n_particles=250,
        state_dim=2,
        observation_dim=3,
        process_noise=0.2,
        observation_noise=0.4,
        resampling_threshold=0.8,
        random_seed=7,
    )

    assert config.to_dict() == {
        "model_name": "particle_filter",
        "n_particles": 250,
        "state_dim": 2,
        "observation_dim": 3,
        "process_noise": 0.2,
        "observation_noise": 0.4,
        "resampling_threshold": 0.8,
        "random_seed": 7,
    }


def test_config_is_frozen():
    config = ParticleFilterConfig()

    with pytest.raises(
        (AttributeError, TypeError),
    ):
        config.n_particles = 500