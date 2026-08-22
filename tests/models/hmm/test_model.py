import numpy as np
import pytest

from src.models.hmm.config import HMMConfig
from src.models.hmm.model import GaussianHMMRegimeModel
from src.models.result import AdvancedModelResult


@pytest.fixture
def sample_data():
    rng = np.random.RandomState(42)

    regime_one = rng.normal(
        loc=[-1.0, 1.0],
        scale=[0.2, 0.2],
        size=(30, 2),
    )

    regime_two = rng.normal(
        loc=[1.0, -1.0],
        scale=[0.2, 0.2],
        size=(30, 2),
    )

    return np.vstack([
        regime_one,
        regime_two,
    ])


@pytest.fixture
def config():
    return HMMConfig(
        n_regimes=2,
        n_iter=50,
        random_state=42,
    )


@pytest.fixture
def model(config):
    return GaussianHMMRegimeModel(config)


def test_default_model_creation():
    model = GaussianHMMRegimeModel()

    assert isinstance(model.hmm_config, HMMConfig)
    assert model.model is None
    assert model.n_features is None
    assert model.is_fitted is False


def test_custom_configuration(config):
    model = GaussianHMMRegimeModel(config)

    assert model.model_name == "gaussian_hmm"
    assert model.n_regimes == 2
    assert model.is_fitted is False


def test_invalid_configuration_rejected():
    with pytest.raises(TypeError):
        GaussianHMMRegimeModel(config={})


def test_validate_input_returns_float_array(model):
    result = model.validate_input([
        [1, 2],
        [3, 4],
    ])

    assert isinstance(result, np.ndarray)
    assert result.dtype == float
    assert result.shape == (2, 2)


@pytest.mark.parametrize(
    "data",
    [
        None,
        [],
    ],
)
def test_empty_or_none_input_rejected(model, data):
    with pytest.raises(ValueError):
        model.validate_input(data)


@pytest.mark.parametrize(
    "data",
    [
        "invalid",
        [["a", "b"]],
    ],
)
def test_non_numeric_input_rejected(model, data):
    with pytest.raises(TypeError):
        model.validate_input(data)


@pytest.mark.parametrize(
    "data",
    [
        [1, 2, 3],
        np.array([1, 2, 3]),
    ],
)
def test_non_2d_input_rejected(model, data):
    with pytest.raises(ValueError):
        model.validate_input(data)


def test_non_finite_input_rejected(model):
    with pytest.raises(ValueError):
        model.validate_input([
            [1.0, np.nan],
            [2.0, 3.0],
        ])


def test_feature_column_count_validation():
    config = HMMConfig(
        n_regimes=2,
        feature_columns=[
            "returns",
            "volatility",
            "volume",
        ],
    )

    model = GaussianHMMRegimeModel(config)

    with pytest.raises(ValueError):
        model.fit(np.ones((10, 2)))


def test_insufficient_samples_rejected(model):
    with pytest.raises(ValueError):
        model.fit([[1.0, 2.0]])


def test_fit_marks_model_as_fitted(model, sample_data):
    returned = model.fit(sample_data)

    assert returned is model
    assert model.is_fitted is True
    assert model.model is not None
    assert model.n_features == 2


def test_predict_before_fit_rejected(model, sample_data):
    with pytest.raises(RuntimeError):
        model.predict(sample_data)


def test_predict_returns_advanced_model_result(
    model,
    sample_data,
):
    model.fit(sample_data)

    result = model.predict(sample_data)

    assert isinstance(result, AdvancedModelResult)
    assert result.n_predictions == len(sample_data)
    assert result.has_probabilities is True
    assert result.has_confidence is True
    assert result.model_name == "gaussian_hmm"

    assert len(result.probabilities) == len(sample_data)
    assert len(result.confidence) == len(sample_data)


def test_regimes_are_valid_state_indices(
    model,
    sample_data,
):
    model.fit(sample_data)

    result = model.predict(sample_data)

    assert all(
        0 <= regime < model.n_regimes
        for regime in result.regimes
    )


def test_probability_rows_sum_to_one(
    model,
    sample_data,
):
    model.fit(sample_data)

    probabilities = model.predict_proba(sample_data)

    assert len(probabilities) == len(sample_data)

    for row in probabilities:
        assert sum(row) == pytest.approx(1.0)


def test_confidence_matches_max_probability(
    model,
    sample_data,
):
    model.fit(sample_data)

    result = model.predict(sample_data)

    for row, confidence in zip(
        result.probabilities,
        result.confidence,
    ):
        assert confidence == pytest.approx(max(row))


def test_prediction_feature_count_must_match_training(
    model,
    sample_data,
):
    model.fit(sample_data)

    wrong_data = np.ones(
        (10, 3),
        dtype=float,
    )

    with pytest.raises(ValueError):
        model.predict(wrong_data)


def test_predict_proba_before_fit_rejected(
    model,
    sample_data,
):
    with pytest.raises(RuntimeError):
        model.predict_proba(sample_data)


def test_score_returns_float(model, sample_data):
    model.fit(sample_data)

    score = model.score(sample_data)

    assert isinstance(score, float)


def test_score_before_fit_rejected(model, sample_data):
    with pytest.raises(RuntimeError):
        model.score(sample_data)


def test_transition_matrix_shape(
    model,
    sample_data,
):
    model.fit(sample_data)

    matrix = model.get_transition_matrix()

    assert len(matrix) == model.n_regimes

    for row in matrix:
        assert len(row) == model.n_regimes


def test_transition_matrix_rows_sum_to_one(
    model,
    sample_data,
):
    model.fit(sample_data)

    matrix = model.get_transition_matrix()

    for row in matrix:
        assert sum(row) == pytest.approx(1.0)


def test_state_counts(model, sample_data):
    model.fit(sample_data)

    counts = model.get_state_counts(sample_data)

    assert len(counts) == model.n_regimes
    assert sum(counts) == len(sample_data)
    assert all(isinstance(count, int) for count in counts)


def test_diagnostics(model, sample_data):
    model.fit(sample_data)

    diagnostics = model.get_diagnostics()

    assert "converged" in diagnostics
    assert "iterations" in diagnostics
    assert "log_likelihood_history" in diagnostics

    assert isinstance(
        diagnostics["converged"],
        bool,
    )
    assert diagnostics["iterations"] >= 1
    assert isinstance(
        diagnostics["log_likelihood_history"],
        list,
    )


def test_predict_metadata(model, sample_data):
    model.fit(sample_data)

    result = model.predict(sample_data)

    assert result.metadata["n_regimes"] == 2
    assert result.metadata["n_features"] == 2
    assert result.metadata["covariance_type"] == "diag"


def test_reproducible_predictions(sample_data):
    config = HMMConfig(
        n_regimes=2,
        n_iter=50,
        random_state=123,
    )

    model_one = GaussianHMMRegimeModel(config)
    model_two = GaussianHMMRegimeModel(config)

    model_one.fit(sample_data)
    model_two.fit(sample_data)

    prediction_one = model_one.predict(sample_data)
    prediction_two = model_two.predict(sample_data)

    assert prediction_one.regimes == prediction_two.regimes