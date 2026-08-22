from __future__ import annotations

import numpy as np
import pytest

from src.models.bayesian.base import BayesianModelResult
from src.models.bayesian.config import BayesianModelConfig
from src.models.bayesian.linear import BayesianLinearRegression

def test_fit_returns_self():
    model = BayesianLinearRegression()

    data = np.array(
        [
            [1.0],
            [2.0],
            [3.0],
        ],
        dtype=np.float64,
    )

    targets = np.array(
        [
            [2.0],
            [4.0],
            [6.0],
        ],
        dtype=np.float64,
    )

    result = model.fit(
        data=data,
        targets=targets,
    )

    assert result is model


def test_fit_marks_model_as_fitted():
    model = BayesianLinearRegression()

    data = np.array(
        [
            [1.0],
            [2.0],
            [3.0],
        ],
        dtype=np.float64,
    )

    targets = np.array(
        [
            [2.0],
            [4.0],
            [6.0],
        ],
        dtype=np.float64,
    )

    model.fit(
        data=data,
        targets=targets,
    )

    assert model.is_fitted is True


def test_fit_creates_posterior_mean():
    model = BayesianLinearRegression()

    data = np.array(
        [
            [1.0],
            [2.0],
            [3.0],
        ],
        dtype=np.float64,
    )

    targets = np.array(
        [
            [2.0],
            [4.0],
            [6.0],
        ],
        dtype=np.float64,
    )

    model.fit(
        data=data,
        targets=targets,
    )

    assert model._posterior_mean is not None
    assert model._posterior_mean.shape == (1, 1)


def test_fit_creates_posterior_covariance():
    model = BayesianLinearRegression()

    data = np.array(
        [
            [1.0],
            [2.0],
            [3.0],
        ],
        dtype=np.float64,
    )

    targets = np.array(
        [
            [2.0],
            [4.0],
            [6.0],
        ],
        dtype=np.float64,
    )

    model.fit(
        data=data,
        targets=targets,
    )

    assert model._posterior_covariance is not None
    assert model._posterior_covariance.shape == (1, 1)


def test_fit_multi_feature_multi_output():
    config = BayesianModelConfig(
        n_features=2,
        n_outputs=2,
    )

    model = BayesianLinearRegression(
        config=config,
    )

    data = np.array(
        [
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
        ],
        dtype=np.float64,
    )

    targets = np.array(
        [
            [1.0, 2.0],
            [2.0, 4.0],
            [3.0, 6.0],
        ],
        dtype=np.float64,
    )

    model.fit(
        data=data,
        targets=targets,
    )

    assert model._posterior_mean is not None
    assert model._posterior_mean.shape == (2, 2)

    assert model._posterior_covariance is not None
    assert model._posterior_covariance.shape == (2, 2)


def test_fit_rejects_invalid_features():
    model = BayesianLinearRegression()

    with pytest.raises(TypeError):
        model.fit(
            data="invalid",
            targets=np.array(
                [[1.0]],
                dtype=np.float64,
            ),
        )


def test_fit_rejects_invalid_targets():
    model = BayesianLinearRegression()

    with pytest.raises(TypeError):
        model.fit(
            data=np.array(
                [[1.0]],
                dtype=np.float64,
            ),
            targets="invalid",
        )

def test_posterior_mean_learns_linear_relationship():
    config = BayesianModelConfig(
        n_features=1,
        n_outputs=1,
        prior_mean=0.0,
        prior_std=10.0,
        observation_noise=0.01,
    )

    model = BayesianLinearRegression(
        config=config,
    )

    data = np.array(
        [
            [1.0],
            [2.0],
            [3.0],
            [4.0],
        ],
        dtype=np.float64,
    )

    targets = np.array(
        [
            [2.0],
            [4.0],
            [6.0],
            [8.0],
        ],
        dtype=np.float64,
    )

    model.fit(
        data=data,
        targets=targets,
    )

    assert model._posterior_mean is not None
    assert model._posterior_mean.shape == (1, 1)

    learned_weight = model._posterior_mean[0, 0]

    assert learned_weight == pytest.approx(
        2.0,
        abs=0.01,
    )

def test_predict_before_fit_rejected():
    model = BayesianLinearRegression()

    data = np.array(
        [
            [1.0],
        ],
        dtype=np.float64,
    )

    with pytest.raises(RuntimeError):
        model.predict(data)


def test_predict_returns_bayesian_result():
    model = BayesianLinearRegression()

    data = np.array(
        [
            [1.0],
            [2.0],
            [3.0],
        ],
        dtype=np.float64,
    )

    targets = np.array(
        [
            [2.0],
            [4.0],
            [6.0],
        ],
        dtype=np.float64,
    )

    model.fit(data, targets)

    result = model.predict(
        np.array(
            [
                [4.0],
            ],
            dtype=np.float64,
        )
    )

    assert isinstance(
        result,
        BayesianModelResult,
    )


def test_predict_result_shapes():
    model = BayesianLinearRegression()

    data = np.array(
        [
            [1.0],
            [2.0],
            [3.0],
        ],
        dtype=np.float64,
    )

    targets = np.array(
        [
            [2.0],
            [4.0],
            [6.0],
        ],
        dtype=np.float64,
    )

    model.fit(data, targets)

    result = model.predict(
        np.array(
            [
                [4.0],
                [5.0],
            ],
            dtype=np.float64,
        )
    )

    assert result.mean.shape == (2, 1)
    assert result.variance.shape == (2, 1)


def test_predict_learns_expected_mean():
    config = BayesianModelConfig(
        n_features=1,
        n_outputs=1,
        prior_mean=0.0,
        prior_std=10.0,
        observation_noise=0.01,
    )

    model = BayesianLinearRegression(
        config=config,
    )

    data = np.array(
        [
            [1.0],
            [2.0],
            [3.0],
            [4.0],
        ],
        dtype=np.float64,
    )

    targets = np.array(
        [
            [2.0],
            [4.0],
            [6.0],
            [8.0],
        ],
        dtype=np.float64,
    )

    model.fit(data, targets)

    result = model.predict(
        np.array(
            [
                [5.0],
            ],
            dtype=np.float64,
        )
    )

    assert result.mean[0, 0] == pytest.approx(
        10.0,
        abs=0.05,
    )


def test_predict_variance_is_positive():
    model = BayesianLinearRegression()

    data = np.array(
        [
            [1.0],
            [2.0],
            [3.0],
        ],
        dtype=np.float64,
    )

    targets = np.array(
        [
            [2.0],
            [4.0],
            [6.0],
        ],
        dtype=np.float64,
    )

    model.fit(data, targets)

    result = model.predict(
        np.array(
            [
                [4.0],
                [5.0],
            ],
            dtype=np.float64,
        )
    )

    assert np.all(result.variance > 0.0)


def test_predict_multi_output_shapes():
    config = BayesianModelConfig(
        n_features=2,
        n_outputs=2,
    )

    model = BayesianLinearRegression(
        config=config,
    )

    data = np.array(
        [
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
        ],
        dtype=np.float64,
    )

    targets = np.array(
        [
            [2.0, 3.0],
            [4.0, 6.0],
            [6.0, 9.0],
        ],
        dtype=np.float64,
    )

    model.fit(data, targets)

    result = model.predict(
        np.array(
            [
                [4.0, 5.0],
                [5.0, 6.0],
            ],
            dtype=np.float64,
        )
    )

    assert result.mean.shape == (2, 2)
    assert result.variance.shape == (2, 2)


def test_predict_rejects_invalid_features():
    model = BayesianLinearRegression()

    model.fit(
        np.array(
            [[1.0]],
            dtype=np.float64,
        ),
        np.array(
            [[2.0]],
            dtype=np.float64,
        ),
    )

    with pytest.raises(TypeError):
        model.predict("invalid")

def test_prediction_is_close_to_expected_linear_value():
    config = BayesianModelConfig(
        n_features=1,
        n_outputs=1,
        prior_mean=0.0,
        prior_std=10.0,
        observation_noise=0.01,
    )

    model = BayesianLinearRegression(config=config)

    data = np.array(
        [
            [1.0],
            [2.0],
            [3.0],
            [4.0],
        ],
        dtype=np.float64,
    )

    targets = np.array(
        [
            [3.0],
            [6.0],
            [9.0],
            [12.0],
        ],
        dtype=np.float64,
    )

    model.fit(data, targets)

    result = model.predict(
        np.array(
            [[5.0]],
            dtype=np.float64,
        )
    )

    assert result.mean[0, 0] == pytest.approx(
        15.0,
        abs=0.05,
    )


def test_prediction_variance_increases_farther_from_training_data():
    config = BayesianModelConfig(
        n_features=1,
        n_outputs=1,
        prior_std=1.0,
        observation_noise=0.1,
    )

    model = BayesianLinearRegression(config=config)

    data = np.array(
        [
            [1.0],
            [2.0],
            [3.0],
        ],
        dtype=np.float64,
    )

    targets = np.array(
        [
            [2.0],
            [4.0],
            [6.0],
        ],
        dtype=np.float64,
    )

    model.fit(data, targets)

    near_result = model.predict(
        np.array(
            [[2.0]],
            dtype=np.float64,
        )
    )

    far_result = model.predict(
        np.array(
            [[10.0]],
            dtype=np.float64,
        )
    )

    assert (
        far_result.variance[0, 0]
        > near_result.variance[0, 0]
    )


def test_higher_observation_noise_increases_uncertainty():
    low_noise_model = BayesianLinearRegression(
        config=BayesianModelConfig(
            observation_noise=0.1,
        )
    )

    high_noise_model = BayesianLinearRegression(
        config=BayesianModelConfig(
            observation_noise=2.0,
        )
    )

    data = np.array(
        [
            [1.0],
            [2.0],
            [3.0],
            [4.0],
        ],
        dtype=np.float64,
    )

    targets = np.array(
        [
            [2.0],
            [4.0],
            [6.0],
            [8.0],
        ],
        dtype=np.float64,
    )

    low_noise_model.fit(data, targets)
    high_noise_model.fit(data, targets)

    prediction_data = np.array(
        [[5.0]],
        dtype=np.float64,
    )

    low_noise_result = low_noise_model.predict(
        prediction_data
    )

    high_noise_result = high_noise_model.predict(
        prediction_data
    )

    assert (
        high_noise_result.variance[0, 0]
        > low_noise_result.variance[0, 0]
    )


def test_multi_output_predictions_are_finite():
    config = BayesianModelConfig(
        n_features=2,
        n_outputs=2,
        observation_noise=0.5,
    )

    model = BayesianLinearRegression(config=config)

    data = np.array(
        [
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
        ],
        dtype=np.float64,
    )

    targets = np.array(
        [
            [2.0, 1.0],
            [4.0, 2.0],
            [6.0, 3.0],
            [8.0, 4.0],
        ],
        dtype=np.float64,
    )

    model.fit(data, targets)

    result = model.predict(
        np.array(
            [
                [5.0, 6.0],
                [6.0, 7.0],
            ],
            dtype=np.float64,
        )
    )

    assert np.all(np.isfinite(result.mean))
    assert np.all(np.isfinite(result.variance))
    assert np.all(result.variance > 0.0)