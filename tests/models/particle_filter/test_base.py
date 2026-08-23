import numpy as np
import pytest

from src.models.particle_filter.base import (
    BaseParticleFilter,
    ParticleFilterResult,
)
from src.models.particle_filter.config import (
    ParticleFilterConfig,
)


class DummyParticleFilter(BaseParticleFilter):
    def fit(self, observations):
        observations = self._validate_observations(
            observations
        )
        self._n_samples_seen = observations.shape[0]
        self._is_fitted = True
        return self

    def predict(self, observations):
        observations = self._validate_observations(
            observations
        )

        if not self.is_fitted:
            raise RuntimeError(
                "model must be fitted before prediction."
            )

        state_mean = np.zeros(
            self.config.state_dim,
            dtype=np.float64,
        )

        state_covariance = np.eye(
            self.config.state_dim,
            dtype=np.float64,
        )

        return ParticleFilterResult(
            state_mean=state_mean,
            state_covariance=state_covariance,
            effective_sample_size=float(
                self.config.n_particles
            ),
        )


def test_result_creation():
    result = ParticleFilterResult(
        state_mean=np.array([1.0]),
        state_covariance=np.array([[0.5]]),
        effective_sample_size=80.0,
    )

    assert result.effective_sample_size == 80.0
    assert result.metadata == {}


def test_result_with_metadata():
    result = ParticleFilterResult(
        state_mean=np.array([1.0]),
        state_covariance=np.array([[0.5]]),
        effective_sample_size=50.0,
        metadata={"step": 10},
    )

    assert result.metadata == {"step": 10}


def test_default_config_creation():
    model = DummyParticleFilter()

    assert isinstance(
        model.config,
        ParticleFilterConfig,
    )
    assert model.is_fitted is False


def test_custom_config_creation():
    config = ParticleFilterConfig(
        n_particles=50,
        state_dim=2,
        observation_dim=3,
    )

    model = DummyParticleFilter(config)

    assert model.config == config
    assert model.is_fitted is False


def test_invalid_config_rejected():
    with pytest.raises(TypeError):
        DummyParticleFilter(config="invalid")


def test_fit_returns_self():
    model = DummyParticleFilter()

    observations = np.array(
        [[1.0], [2.0], [3.0]]
    )

    result = model.fit(observations)

    assert result is model


def test_fit_marks_model_as_fitted():
    model = DummyParticleFilter()

    model.fit(
        np.array([[1.0], [2.0]])
    )

    assert model.is_fitted is True


def test_predict_before_fit_rejected():
    model = DummyParticleFilter()

    with pytest.raises(RuntimeError):
        model.predict(np.array([[1.0]]))


def test_predict_after_fit():
    model = DummyParticleFilter()

    observations = np.array(
        [[1.0], [2.0]]
    )

    model.fit(observations)
    result = model.predict(observations)

    assert isinstance(result, ParticleFilterResult)


def test_validate_observations_returns_float64():
    model = DummyParticleFilter()

    result = model._validate_observations(
        np.array([[1], [2]], dtype=np.int32)
    )

    assert result.dtype == np.float64


@pytest.mark.parametrize(
    "observations",
    [
        [[1.0]],
        "invalid",
        123,
    ],
)
def test_validate_observations_rejects_non_array(
    observations,
):
    model = DummyParticleFilter()

    with pytest.raises(TypeError):
        model._validate_observations(observations)


@pytest.mark.parametrize(
    "observations",
    [
        np.array([1.0, 2.0]),
        np.array([[[1.0]]]),
    ],
)
def test_validate_observations_rejects_wrong_dimensions(
    observations,
):
    model = DummyParticleFilter()

    with pytest.raises(ValueError):
        model._validate_observations(observations)


def test_validate_observations_rejects_empty():
    model = DummyParticleFilter()

    with pytest.raises(ValueError):
        model._validate_observations(
            np.empty((0, 1))
        )


def test_validate_observations_rejects_wrong_dimension():
    model = DummyParticleFilter()

    with pytest.raises(ValueError):
        model._validate_observations(
            np.ones((2, 2))
        )


def test_validate_observations_rejects_non_numeric():
    model = DummyParticleFilter()

    with pytest.raises(TypeError):
        model._validate_observations(
            np.array(
                [["invalid"]],
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
def test_validate_observations_rejects_non_finite(value):
    model = DummyParticleFilter()

    with pytest.raises(ValueError):
        model._validate_observations(
            np.array([[value]])
        )


def test_validate_states_returns_float64():
    model = DummyParticleFilter()

    result = model._validate_states(
        np.array([[1], [2]], dtype=np.int32)
    )

    assert result.dtype == np.float64


def test_validate_states_rejects_wrong_dimension():
    model = DummyParticleFilter()

    with pytest.raises(ValueError):
        model._validate_states(
            np.ones((2, 2))
        )


def test_validate_states_rejects_empty():
    model = DummyParticleFilter()

    with pytest.raises(ValueError):
        model._validate_states(
            np.empty((0, 1))
        )


def test_validate_states_rejects_non_numeric():
    model = DummyParticleFilter()

    with pytest.raises(TypeError):
        model._validate_states(
            np.array(
                [["invalid"]],
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
def test_validate_states_rejects_non_finite(value):
    model = DummyParticleFilter()

    with pytest.raises(ValueError):
        model._validate_states(
            np.array([[value]])
        )


def test_validate_weights_normalizes():
    model = DummyParticleFilter()

    result = model._validate_weights(
        np.array([1.0] * 100)
    )

    assert np.isclose(result.sum(), 1.0)
    assert np.allclose(
        result,
        np.ones(100) / 100,
    )


def test_validate_weights_rejects_wrong_length():
    model = DummyParticleFilter()

    with pytest.raises(ValueError):
        model._validate_weights(
            np.ones(99)
        )


def test_validate_weights_rejects_negative():
    model = DummyParticleFilter()

    weights = np.ones(100)
    weights[0] = -1.0

    with pytest.raises(ValueError):
        model._validate_weights(weights)


def test_validate_weights_rejects_zero_sum():
    model = DummyParticleFilter()

    with pytest.raises(ValueError):
        model._validate_weights(
            np.zeros(100)
        )


def test_validate_weights_rejects_non_array():
    model = DummyParticleFilter()

    with pytest.raises(TypeError):
        model._validate_weights(
            [1.0] * 100
        )


def test_metadata_before_fit():
    model = DummyParticleFilter()

    metadata = model.get_metadata()

    assert metadata["is_fitted"] is False
    assert metadata["n_samples_seen"] == 0


def test_metadata_after_fit():
    model = DummyParticleFilter()

    model.fit(
        np.array(
            [[1.0], [2.0], [3.0]]
        )
    )

    metadata = model.get_metadata()

    assert metadata["is_fitted"] is True
    assert metadata["n_samples_seen"] == 3