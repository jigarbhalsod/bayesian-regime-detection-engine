import numpy as np
import pytest

from src.models.bayesian.config import BayesianModelConfig
from src.models.bayesian.service import BayesianModelService


def _create_service() -> BayesianModelService:
    config = BayesianModelConfig(
        model_name="service_test_model",
        n_features=2,
        n_outputs=1,
        prior_mean=0.0,
        prior_std=1.0,
        observation_noise=1.0,
    )

    return BayesianModelService(config=config)


def _training_data() -> tuple[np.ndarray, np.ndarray]:
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

    return data, targets


def test_service_creation():
    service = BayesianModelService()

    assert service.model is not None
    assert service.is_fitted is False


def test_service_custom_config():
    service = _create_service()

    assert service.model.config.model_name == "service_test_model"
    assert service.model.config.n_features == 2
    assert service.model.config.n_outputs == 1


def test_service_fit_returns_self():
    service = _create_service()
    data, targets = _training_data()

    result = service.fit(
        data=data,
        targets=targets,
    )

    assert result is service


def test_service_is_fitted_after_fit():
    service = _create_service()
    data, targets = _training_data()

    service.fit(
        data=data,
        targets=targets,
    )

    assert service.is_fitted is True


def test_service_predict_before_fit_rejected():
    service = _create_service()

    data = np.array(
        [[1.0, 2.0]],
        dtype=np.float64,
    )

    with pytest.raises(RuntimeError):
        service.predict(data=data)


def test_service_predict_after_fit():
    service = _create_service()
    data, targets = _training_data()

    service.fit(
        data=data,
        targets=targets,
    )

    predictions = service.predict(
        data=np.array(
            [
                [5.0, 6.0],
                [6.0, 7.0],
            ],
            dtype=np.float64,
        )
    )

    assert predictions.mean.shape == (2, 1)
    assert predictions.variance.shape == (2, 1)

    assert np.all(np.isfinite(predictions.mean))
    assert np.all(np.isfinite(predictions.variance))


def test_service_metadata():
    service = _create_service()

    metadata = service.get_metadata()

    assert metadata["model_class"] == "BayesianLinearRegression"
    assert metadata["is_fitted"] is False
    assert metadata["n_features"] == 2
    assert metadata["n_outputs"] == 1


def test_service_summary_before_fit_rejected():
    service = _create_service()

    with pytest.raises(RuntimeError):
        service.get_model_summary()


def test_service_summary_after_fit():
    service = _create_service()
    data, targets = _training_data()

    service.fit(
        data=data,
        targets=targets,
    )

    summary = service.get_model_summary()

    assert "metadata" in summary
    assert summary["metadata"]["is_fitted"] is True
    assert summary["posterior_mean_shape"] == (2, 1)
    assert summary["posterior_covariance_shape"] == (2, 2)


def test_service_posterior_parameters_before_fit_rejected():
    service = _create_service()

    with pytest.raises(RuntimeError):
        service.get_posterior_parameters()


def test_service_posterior_parameters_after_fit():
    service = _create_service()
    data, targets = _training_data()

    service.fit(
        data=data,
        targets=targets,
    )

    parameters = service.get_posterior_parameters()

    assert set(parameters.keys()) == {
        "mean",
        "covariance",
    }

    assert parameters["mean"].shape == (2, 1)
    assert parameters["covariance"].shape == (2, 2)


def test_service_end_to_end_workflow():
    service = _create_service()
    data, targets = _training_data()

    assert service.is_fitted is False

    service.fit(
        data=data,
        targets=targets,
    )

    assert service.is_fitted is True

    prediction = service.predict(
        data=np.array(
            [[5.0, 6.0]],
            dtype=np.float64,
        )
    )

    metadata = service.get_metadata()
    summary = service.get_model_summary()
    parameters = service.get_posterior_parameters()

    assert prediction.mean.shape == (1, 1)
    assert prediction.variance.shape == (1, 1)

    assert metadata["is_fitted"] is True
    assert summary["metadata"]["is_fitted"] is True

    assert parameters["mean"].shape == (2, 1)
    assert parameters["covariance"].shape == (2, 2)