import numpy as np
import pytest

from src.models.bayesian.config import BayesianModelConfig
from src.models.bayesian.linear import BayesianLinearRegression


def _create_fitted_model() -> BayesianLinearRegression:
    config = BayesianModelConfig(
        model_name="test_bayesian_model",
        n_features=2,
        n_outputs=1,
        prior_mean=0.0,
        prior_std=1.0,
        observation_noise=1.0,
    )

    model = BayesianLinearRegression(
        config=config,
    )

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
            [3.0],
            [5.0],
            [7.0],
            [9.0],
        ],
        dtype=np.float64,
    )

    model.fit(
        data=data,
        targets=targets,
    )

    return model


# ============================================================
# F4.1 — Posterior parameter access
# ============================================================

def test_get_posterior_parameters_before_fit_rejected():
    model = BayesianLinearRegression()

    with pytest.raises(RuntimeError):
        model.get_posterior_parameters()


def test_get_posterior_parameters_returns_expected_keys():
    model = _create_fitted_model()

    parameters = model.get_posterior_parameters()

    assert set(parameters.keys()) == {
        "mean",
        "covariance",
    }


def test_posterior_parameter_shapes():
    model = _create_fitted_model()

    parameters = model.get_posterior_parameters()

    assert parameters["mean"].shape == (2, 1)
    assert parameters["covariance"].shape == (2, 2)


def test_posterior_parameters_are_copies():
    model = _create_fitted_model()

    parameters = model.get_posterior_parameters()

    original_mean = model._posterior_mean.copy()
    original_covariance = (
        model._posterior_covariance.copy()
    )

    parameters["mean"][0, 0] = 999.0
    parameters["covariance"][0, 0] = 999.0

    assert np.array_equal(
        model._posterior_mean,
        original_mean,
    )

    assert np.array_equal(
        model._posterior_covariance,
        original_covariance,
    )


def test_posterior_parameters_are_finite():
    model = _create_fitted_model()

    parameters = model.get_posterior_parameters()

    assert np.all(np.isfinite(parameters["mean"]))
    assert np.all(
        np.isfinite(parameters["covariance"])
    )


# ============================================================
# F4.2 — Metadata and model summary
# ============================================================

def test_metadata_before_fit():
    model = BayesianLinearRegression()

    metadata = model.get_metadata()

    assert metadata["is_fitted"] is False
    assert metadata["model_class"] == (
        "BayesianLinearRegression"
    )
    assert metadata["n_features"] == 1
    assert metadata["n_outputs"] == 1
    assert metadata["has_posterior"] is False


def test_metadata_after_fit():
    model = _create_fitted_model()

    metadata = model.get_metadata()

    assert metadata["is_fitted"] is True
    assert metadata["model_class"] == (
        "BayesianLinearRegression"
    )
    assert metadata["n_features"] == 2
    assert metadata["n_outputs"] == 1
    assert metadata["has_posterior"] is True


def test_metadata_contains_configuration_values():
    model = _create_fitted_model()

    metadata = model.get_metadata()

    assert metadata["prior_mean"] == 0.0
    assert metadata["prior_std"] == 1.0
    assert metadata["observation_noise"] == 1.0


def test_model_summary_before_fit_rejected():
    model = BayesianLinearRegression()

    with pytest.raises(RuntimeError):
        model.get_model_summary()


def test_model_summary_after_fit():
    model = _create_fitted_model()

    summary = model.get_model_summary()

    assert "metadata" in summary
    assert "posterior_mean_shape" in summary
    assert "posterior_covariance_shape" in summary
    assert "posterior_mean_finite" in summary
    assert "posterior_covariance_finite" in summary


def test_model_summary_parameter_shapes():
    model = _create_fitted_model()

    summary = model.get_model_summary()

    assert summary["posterior_mean_shape"] == (2, 1)
    assert summary["posterior_covariance_shape"] == (2, 2)


def test_model_summary_parameters_are_finite():
    model = _create_fitted_model()

    summary = model.get_model_summary()

    assert summary["posterior_mean_finite"] is True
    assert summary["posterior_covariance_finite"] is True