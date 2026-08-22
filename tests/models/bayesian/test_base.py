from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from src.models.bayesian.base import (
    BaseBayesianModel,
    BayesianModelResult,
)
from src.models.bayesian.config import BayesianModelConfig


class DummyBayesianModel(BaseBayesianModel):
    """Concrete implementation used for testing."""

    def fit(
        self,
        data: Any,
        targets: Any,
        **kwargs: Any,
    ) -> "DummyBayesianModel":
        features = self._validate_features(data)
        self._validate_targets(
            targets,
            n_samples=features.shape[0],
        )
        self._is_fitted = True
        return self

    def predict(
        self,
        data: Any,
        **kwargs: Any,
    ) -> BayesianModelResult:
        self._validate_fitted()
        features = self._validate_features(data)

        mean = np.zeros(
            (
                features.shape[0],
                self.config.n_outputs,
            ),
            dtype=np.float64,
        )

        variance = np.ones(
            (
                features.shape[0],
                self.config.n_outputs,
            ),
            dtype=np.float64,
        )

        return BayesianModelResult(
            mean=mean,
            variance=variance,
        )


def make_config() -> BayesianModelConfig:
    return BayesianModelConfig(
        model_name="dummy_bayesian",
        n_features=3,
        n_outputs=2,
    )


def make_model() -> DummyBayesianModel:
    return DummyBayesianModel(
        config=make_config(),
    )


def make_features(
    n_samples: int = 5,
) -> np.ndarray:
    return np.arange(
        n_samples * 3,
        dtype=np.float64,
    ).reshape(
        n_samples,
        3,
    )


def make_targets(
    n_samples: int = 5,
) -> np.ndarray:
    return np.ones(
        (
            n_samples,
            2,
        ),
        dtype=np.float64,
    )


def test_result_creation():
    result = BayesianModelResult(
        mean=np.array(
            [[1.0, 2.0]],
            dtype=np.float64,
        ),
        variance=np.array(
            [[0.1, 0.2]],
            dtype=np.float64,
        ),
    )

    assert result.mean.shape == (1, 2)
    assert result.variance.shape == (1, 2)
    assert result.metadata is None


def test_result_with_metadata():
    result = BayesianModelResult(
        mean=np.zeros(
            (2, 1),
            dtype=np.float64,
        ),
        variance=np.ones(
            (2, 1),
            dtype=np.float64,
        ),
        metadata={
            "source": "test",
        },
    )

    assert result.metadata == {
        "source": "test",
    }


def test_default_config_creation():
    model = DummyBayesianModel()

    assert isinstance(
        model.config,
        BayesianModelConfig,
    )
    assert model.is_fitted is False


def test_custom_config_creation():
    config = make_config()
    model = DummyBayesianModel(config=config)

    assert model.config is config
    assert model.config.n_features == 3
    assert model.config.n_outputs == 2


def test_predict_before_fit_rejected():
    model = make_model()

    with pytest.raises(RuntimeError):
        model.predict(make_features())


def test_fit_returns_self():
    model = make_model()

    result = model.fit(
        make_features(),
        make_targets(),
    )

    assert result is model
    assert model.is_fitted is True


def test_predict_after_fit():
    model = make_model()

    model.fit(
        make_features(),
        make_targets(),
    )

    result = model.predict(
        make_features(3),
    )

    assert isinstance(
        result,
        BayesianModelResult,
    )
    assert result.mean.shape == (3, 2)
    assert result.variance.shape == (3, 2)


def test_validate_features_returns_float64():
    model = make_model()

    data = np.ones(
        (4, 3),
        dtype=np.int64,
    )

    result = model._validate_features(data)

    assert result.dtype == np.float64
    assert result.shape == (4, 3)


@pytest.mark.parametrize(
    "data",
    [
        [[1.0, 2.0, 3.0]],
        "invalid",
        123,
    ],
)
def test_validate_features_rejects_non_array(
    data: Any,
):
    model = make_model()

    with pytest.raises(TypeError):
        model._validate_features(data)


@pytest.mark.parametrize(
    "data",
    [
        np.array(
            [1.0, 2.0, 3.0],
            dtype=np.float64,
        ),
        np.ones(
            (2, 2, 3),
            dtype=np.float64,
        ),
    ],
)
def test_validate_features_rejects_wrong_dimensions(
    data: np.ndarray,
):
    model = make_model()

    with pytest.raises(ValueError):
        model._validate_features(data)


def test_validate_features_rejects_empty():
    model = make_model()

    data = np.empty(
        (0, 3),
        dtype=np.float64,
    )

    with pytest.raises(ValueError):
        model._validate_features(data)


def test_validate_features_rejects_wrong_count():
    model = make_model()

    data = np.ones(
        (3, 2),
        dtype=np.float64,
    )

    with pytest.raises(ValueError):
        model._validate_features(data)


def test_validate_features_rejects_non_numeric():
    model = make_model()

    data = np.array(
        [
            ["a", "b", "c"],
        ],
        dtype=str,
    )

    with pytest.raises(TypeError):
        model._validate_features(data)


@pytest.mark.parametrize(
    "value",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_validate_features_rejects_non_finite(
    value: float,
):
    model = make_model()

    data = make_features()
    data[0, 0] = value

    with pytest.raises(ValueError):
        model._validate_features(data)


def test_validate_targets_accepts_1d():
    model = DummyBayesianModel(
        config=BayesianModelConfig(
            n_features=3,
            n_outputs=1,
        )
    )

    targets = np.array(
        [1, 2, 3],
        dtype=np.int64,
    )

    result = model._validate_targets(
        targets,
        n_samples=3,
    )

    assert result.shape == (3, 1)
    assert result.dtype == np.float64


def test_validate_targets_returns_float64():
    model = make_model()

    targets = np.ones(
        (4, 2),
        dtype=np.int64,
    )

    result = model._validate_targets(
        targets,
        n_samples=4,
    )

    assert result.dtype == np.float64
    assert result.shape == (4, 2)


@pytest.mark.parametrize(
    "targets",
    [
        [1, 2, 3],
        "invalid",
        123,
    ],
)
def test_validate_targets_rejects_non_array(
    targets: Any,
):
    model = make_model()

    with pytest.raises(TypeError):
        model._validate_targets(
            targets,
            n_samples=3,
        )


def test_validate_targets_rejects_wrong_dimensions():
    model = make_model()

    targets = np.ones(
        (2, 2, 2),
        dtype=np.float64,
    )

    with pytest.raises(ValueError):
        model._validate_targets(
            targets,
            n_samples=2,
        )


def test_validate_targets_rejects_wrong_sample_count():
    model = make_model()

    targets = make_targets(4)

    with pytest.raises(ValueError):
        model._validate_targets(
            targets,
            n_samples=5,
        )


def test_validate_targets_rejects_wrong_output_count():
    model = make_model()

    targets = np.ones(
        (5, 1),
        dtype=np.float64,
    )

    with pytest.raises(ValueError):
        model._validate_targets(
            targets,
            n_samples=5,
        )


def test_validate_targets_rejects_non_numeric():
    model = make_model()

    targets = np.array(
        [
            ["a", "b"],
        ],
        dtype=str,
    )

    with pytest.raises(TypeError):
        model._validate_targets(
            targets,
            n_samples=1,
        )


@pytest.mark.parametrize(
    "value",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_validate_targets_rejects_non_finite(
    value: float,
):
    model = make_model()

    targets = make_targets()
    targets[0, 0] = value

    with pytest.raises(ValueError):
        model._validate_targets(
            targets,
            n_samples=5,
        )


def test_metadata_before_fit():
    model = make_model()

    metadata = model.get_metadata()

    assert metadata == {
        "model_name": "dummy_bayesian",
        "n_features": 3,
        "n_outputs": 2,
        "is_fitted": False,
    }


def test_metadata_after_fit():
    model = make_model()

    model.fit(
        make_features(),
        make_targets(),
    )

    metadata = model.get_metadata()

    assert metadata["is_fitted"] is True
    assert metadata["model_name"] == "dummy_bayesian"
