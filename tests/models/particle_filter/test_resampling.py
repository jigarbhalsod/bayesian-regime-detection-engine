import numpy as np
import pytest

from src.models.particle_filter.config import (
    ParticleFilterConfig,
)
from src.models.particle_filter.resampling import (
    ParticleResampler,
)


@pytest.fixture
def config():
    return ParticleFilterConfig(
        n_particles=100,
        state_dim=2,
        observation_dim=1,
        resample_threshold=0.5,
        random_seed=42,
    )


@pytest.fixture
def resampler(config):
    return ParticleResampler(config)


@pytest.fixture
def uniform_weights():
    return np.ones(100, dtype=float) / 100


@pytest.fixture
def concentrated_weights():
    weights = np.zeros(100, dtype=float)
    weights[0] = 1.0
    return weights


def test_resampler_creation(config):
    resampler = ParticleResampler(config)

    assert resampler.config == config
    assert resampler.method == "systematic"


def test_resampler_custom_method(config):
    resampler = ParticleResampler(
        config,
        method="multinomial",
    )

    assert resampler.method == "multinomial"


def test_invalid_config_rejected():
    with pytest.raises(TypeError):
        ParticleResampler("invalid")


@pytest.mark.parametrize(
    "method",
    [
        "invalid",
        "",
        123,
    ],
)
def test_invalid_method_rejected(config, method):
    with pytest.raises(ValueError):
        ParticleResampler(config, method=method)


def test_n_particles_property(resampler):
    assert resampler.n_particles == 100


def test_validate_weights_normalizes(resampler):
    result = resampler.validate_weights(
        np.arange(1, 101, dtype=float)
    )

    assert np.isclose(result.sum(), 1.0)


def test_validate_weights_returns_copy(
    resampler,
):
    weights = np.ones(100)

    result = resampler.validate_weights(weights)
    result[0] = 999.0

    assert weights[0] == 1.0


def test_validate_weights_rejects_non_array(
    resampler,
):
    with pytest.raises(TypeError):
        resampler.validate_weights(
            [1.0] * 100
        )


@pytest.mark.parametrize(
    "weights",
    [
        np.ones((100, 1)),
        np.ones((1, 100)),
        np.ones((10, 10)),
    ],
)
def test_validate_weights_rejects_wrong_dimensions(
    resampler,
    weights,
):
    with pytest.raises(ValueError):
        resampler.validate_weights(weights)


@pytest.mark.parametrize(
    "length",
    [
        99,
        101,
    ],
)
def test_validate_weights_rejects_wrong_length(
    resampler,
    length,
):
    with pytest.raises(ValueError):
        resampler.validate_weights(
            np.ones(length)
        )


def test_validate_weights_rejects_non_numeric(
    resampler,
):
    with pytest.raises(TypeError):
        resampler.validate_weights(
            np.array(
                ["invalid"] * 100,
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
def test_validate_weights_rejects_non_finite(
    resampler,
    value,
):
    weights = np.ones(100)
    weights[0] = value

    with pytest.raises(ValueError):
        resampler.validate_weights(weights)


def test_validate_weights_rejects_negative(
    resampler,
):
    weights = np.ones(100)
    weights[0] = -1.0

    with pytest.raises(ValueError):
        resampler.validate_weights(weights)


def test_validate_weights_rejects_zero_sum(
    resampler,
):
    with pytest.raises(ValueError):
        resampler.validate_weights(
            np.zeros(100)
        )


def test_effective_sample_size_uniform(
    resampler,
    uniform_weights,
):
    result = resampler.effective_sample_size(
        uniform_weights
    )

    assert np.isclose(result, 100.0)


def test_effective_sample_size_concentrated(
    resampler,
    concentrated_weights,
):
    result = resampler.effective_sample_size(
        concentrated_weights
    )

    assert np.isclose(result, 1.0)


def test_should_not_resample_uniform(
    resampler,
    uniform_weights,
):
    assert resampler.should_resample(
        uniform_weights
    ) is False


def test_should_resample_concentrated(
    resampler,
    concentrated_weights,
):
    assert resampler.should_resample(
        concentrated_weights
    ) is True


def test_should_resample_custom_threshold(
    resampler,
    uniform_weights,
):
    assert resampler.should_resample(
        uniform_weights,
        threshold=1.0,
    ) is True


@pytest.mark.parametrize(
    "threshold",
    [
        0,
        -1,
    ],
)
def test_should_resample_invalid_threshold_value(
    resampler,
    uniform_weights,
    threshold,
):
    with pytest.raises(ValueError):
        resampler.should_resample(
            uniform_weights,
            threshold=threshold,
        )


@pytest.mark.parametrize(
    "threshold",
    [
        "invalid",
        True,
        None,
    ],
)
def test_should_resample_invalid_threshold_type(
    resampler,
    uniform_weights,
    threshold,
):
    if threshold is None:
        return

    with pytest.raises(TypeError):
        resampler.should_resample(
            uniform_weights,
            threshold=threshold,
        )


@pytest.mark.parametrize(
    "threshold",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_should_resample_rejects_non_finite_threshold(
    resampler,
    uniform_weights,
    threshold,
):
    with pytest.raises(ValueError):
        resampler.should_resample(
            uniform_weights,
            threshold=threshold,
        )


@pytest.mark.parametrize(
    "method",
    [
        "multinomial",
        "systematic",
        "stratified",
    ],
)
def test_resample_returns_valid_indices(
    resampler,
    uniform_weights,
    method,
):
    indices = resampler.resample(
        uniform_weights,
        method=method,
    )

    assert indices.shape == (100,)
    assert np.issubdtype(
        indices.dtype,
        np.integer,
    )
    assert np.all(indices >= 0)
    assert np.all(indices < 100)


def test_resample_uses_default_method(
    resampler,
    uniform_weights,
):
    indices = resampler.resample(uniform_weights)

    assert indices.shape == (100,)


def test_resample_rejects_invalid_method(
    resampler,
    uniform_weights,
):
    with pytest.raises(ValueError):
        resampler.resample(
            uniform_weights,
            method="invalid",
        )


def test_multinomial_returns_valid_indices(
    resampler,
    uniform_weights,
):
    indices = resampler.multinomial(
        uniform_weights
    )

    assert indices.shape == (100,)
    assert np.all(indices >= 0)
    assert np.all(indices < 100)


def test_systematic_returns_valid_indices(
    resampler,
    uniform_weights,
):
    indices = resampler.systematic(
        uniform_weights
    )

    assert indices.shape == (100,)
    assert np.all(indices >= 0)
    assert np.all(indices < 100)


def test_stratified_returns_valid_indices(
    resampler,
    uniform_weights,
):
    indices = resampler.stratified(
        uniform_weights
    )

    assert indices.shape == (100,)
    assert np.all(indices >= 0)
    assert np.all(indices < 100)


def test_multinomial_concentrated_weights(
    resampler,
    concentrated_weights,
):
    indices = resampler.multinomial(
        concentrated_weights
    )

    assert np.all(indices == 0)


def test_systematic_concentrated_weights(
    resampler,
    concentrated_weights,
):
    indices = resampler.systematic(
        concentrated_weights
    )

    assert np.all(indices == 0)


def test_stratified_concentrated_weights(
    resampler,
    concentrated_weights,
):
    indices = resampler.stratified(
        concentrated_weights
    )

    assert np.all(indices == 0)


def test_reproducible_resampling(config):
    first = ParticleResampler(config)
    second = ParticleResampler(config)

    weights = np.arange(
        1,
        101,
        dtype=float,
    )

    result_one = first.resample(weights)
    result_two = second.resample(weights)

    assert np.array_equal(
        result_one,
        result_two,
    )