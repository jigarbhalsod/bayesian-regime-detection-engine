import numpy as np
import pytest

from src.models.particle_filter.config import (
    ParticleFilterConfig,
)
from src.models.particle_filter.state import (
    ParticleState,
)


@pytest.fixture
def config():
    return ParticleFilterConfig(
        n_particles=4,
        state_dim=2,
        observation_dim=1,
    )


@pytest.fixture
def particles():
    return np.array(
        [
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
        ]
    )


def test_state_creation(config):
    state = ParticleState(config)

    assert state.config == config
    assert state.particles is None
    assert state.weights is None
    assert state.is_initialized is False


def test_invalid_config_rejected():
    with pytest.raises(TypeError):
        ParticleState("invalid")


def test_n_particles_property(config):
    state = ParticleState(config)

    assert state.n_particles == 4


def test_state_dim_property(config):
    state = ParticleState(config)

    assert state.state_dim == 2


def test_initialize_with_uniform_weights(
    config,
    particles,
):
    state = ParticleState(config)

    result = state.initialize(particles)

    assert result is state
    assert state.is_initialized is True
    assert np.allclose(
        state.weights,
        np.ones(4) / 4,
    )


def test_initialize_with_custom_weights(
    config,
    particles,
):
    state = ParticleState(config)

    state.initialize(
        particles,
        np.array([1.0, 2.0, 3.0, 4.0]),
    )

    assert np.isclose(state.weights.sum(), 1.0)
    assert np.allclose(
        state.weights,
        np.array([0.1, 0.2, 0.3, 0.4]),
    )


def test_initialize_rejects_invalid_particles(
    config,
):
    state = ParticleState(config)

    with pytest.raises(TypeError):
        state.initialize("invalid")


def test_initialize_rejects_invalid_weights(
    config,
    particles,
):
    state = ParticleState(config)

    with pytest.raises(TypeError):
        state.initialize(
            particles,
            [1.0, 2.0, 3.0, 4.0],
        )


@pytest.mark.parametrize(
    "value",
    [
        np.array([1.0, 2.0]),
        np.array([[[1.0]]]),
    ],
)
def test_validate_particles_rejects_wrong_dimensions(
    config,
    value,
):
    state = ParticleState(config)

    with pytest.raises(ValueError):
        state._validate_particles(value)


def test_validate_particles_rejects_wrong_shape(
    config,
):
    state = ParticleState(config)

    with pytest.raises(ValueError):
        state._validate_particles(
            np.ones((3, 2))
        )


def test_validate_particles_rejects_non_numeric(
    config,
):
    state = ParticleState(config)

    with pytest.raises(TypeError):
        state._validate_particles(
            np.array(
                [
                    ["a", "b"],
                    ["c", "d"],
                    ["e", "f"],
                    ["g", "h"],
                ],
                dtype=object,
            )
        )


@pytest.mark.parametrize(
    "value",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_validate_particles_rejects_non_finite(
    config,
    value,
):
    state = ParticleState(config)

    particles = np.ones((4, 2))
    particles[0, 0] = value

    with pytest.raises(ValueError):
        state._validate_particles(particles)


def test_validate_particles_returns_copy(
    config,
    particles,
):
    state = ParticleState(config)

    result = state._validate_particles(particles)
    result[0, 0] = 999.0

    assert particles[0, 0] == 1.0


def test_validate_weights_normalizes(config):
    state = ParticleState(config)

    result = state._validate_weights(
        np.array([1.0, 2.0, 3.0, 4.0])
    )

    assert np.isclose(result.sum(), 1.0)


def test_validate_weights_rejects_non_array(config):
    state = ParticleState(config)

    with pytest.raises(TypeError):
        state._validate_weights(
            [1.0, 2.0, 3.0, 4.0]
        )


def test_validate_weights_rejects_wrong_dimensions(
    config,
):
    state = ParticleState(config)

    with pytest.raises(ValueError):
        state._validate_weights(
            np.ones((4, 1))
        )


def test_validate_weights_rejects_wrong_length(config):
    state = ParticleState(config)

    with pytest.raises(ValueError):
        state._validate_weights(
            np.ones(3)
        )


def test_validate_weights_rejects_negative(
    config,
):
    state = ParticleState(config)

    with pytest.raises(ValueError):
        state._validate_weights(
            np.array([1.0, -1.0, 1.0, 1.0])
        )


def test_validate_weights_rejects_zero_sum(
    config,
):
    state = ParticleState(config)

    with pytest.raises(ValueError):
        state._validate_weights(
            np.zeros(4)
        )


@pytest.mark.parametrize(
    "value",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_validate_weights_rejects_non_finite(
    config,
    value,
):
    state = ParticleState(config)

    weights = np.ones(4)
    weights[0] = value

    with pytest.raises(ValueError):
        state._validate_weights(weights)


def test_set_particles_returns_self(
    config,
    particles,
):
    state = ParticleState(config)

    result = state.set_particles(particles)

    assert result is state
    assert np.allclose(
        state.particles,
        particles,
    )


def test_set_weights_returns_self(config):
    state = ParticleState(config)

    result = state.set_weights(
        np.array([1.0, 2.0, 3.0, 4.0])
    )

    assert result is state
    assert np.isclose(
        state.weights.sum(),
        1.0,
    )


def test_weighted_mean(config, particles):
    state = ParticleState(config)

    state.initialize(
        particles,
        np.array([1.0, 1.0, 1.0, 1.0]),
    )

    result = state.weighted_mean()

    assert np.allclose(
        result,
        np.array([2.5, 3.5]),
    )


def test_weighted_mean_before_initialization_rejected(
    config,
):
    state = ParticleState(config)

    with pytest.raises(RuntimeError):
        state.weighted_mean()


def test_weighted_covariance(config, particles):
    state = ParticleState(config)

    state.initialize(
        particles,
        np.ones(4),
    )

    result = state.weighted_covariance()

    expected = np.array(
        [
            [1.25, 1.25],
            [1.25, 1.25],
        ]
    )

    assert np.allclose(result, expected)


def test_weighted_covariance_before_initialization_rejected(
    config,
):
    state = ParticleState(config)

    with pytest.raises(RuntimeError):
        state.weighted_covariance()


def test_effective_sample_size_uniform(
    config,
    particles,
):
    state = ParticleState(config)

    state.initialize(particles)

    assert np.isclose(
        state.effective_sample_size(),
        4.0,
    )


def test_effective_sample_size_concentrated(
    config,
    particles,
):
    state = ParticleState(config)

    state.initialize(
        particles,
        np.array([1.0, 0.0, 0.0, 0.0]),
    )

    assert np.isclose(
        state.effective_sample_size(),
        1.0,
    )


def test_effective_sample_size_before_initialization_rejected(
    config,
):
    state = ParticleState(config)

    with pytest.raises(RuntimeError):
        state.effective_sample_size()


def test_reset(config, particles):
    state = ParticleState(config)

    state.initialize(particles)
    result = state.reset()

    assert result is state
    assert state.particles is None
    assert state.weights is None
    assert state.is_initialized is False


def test_get_particles_returns_copy(
    config,
    particles,
):
    state = ParticleState(config)

    state.initialize(particles)

    result = state.get_particles()
    result[0, 0] = 999.0

    assert state.particles[0, 0] == 1.0


def test_get_weights_returns_copy(
    config,
    particles,
):
    state = ParticleState(config)

    state.initialize(particles)

    result = state.get_weights()
    result[0] = 999.0

    assert state.weights[0] != 999.0


def test_get_particles_before_initialization_rejected(
    config,
):
    state = ParticleState(config)

    with pytest.raises(RuntimeError):
        state.get_particles()


def test_get_weights_before_initialization_rejected(
    config,
):
    state = ParticleState(config)

    with pytest.raises(RuntimeError):
        state.get_weights()


def test_constructor_with_particles_and_weights(
    config,
    particles,
):
    state = ParticleState(
        config=config,
        particles=particles,
        weights=np.ones(4),
    )

    assert state.is_initialized is True
    assert np.isclose(
        state.weights.sum(),
        1.0,
    )


def test_constructor_rejects_mismatched_entries(
    config,
    particles,
):
    with pytest.raises(ValueError):
        ParticleState(
            config=config,
            particles=particles,
            weights=np.ones(3),
        )