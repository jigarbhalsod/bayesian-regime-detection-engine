import numpy as np
import pytest

from hmmlearn.hmm import GMMHMM

from src.models.hmm.config import HMMConfig
from src.models.hmm.gmm import GMMHMMRegimeModel
from src.models.result import AdvancedModelResult


@pytest.fixture
def sample_data():
    rng = np.random.RandomState(42)

    regime_one = rng.normal(
        loc=[-1.0, 1.0],
        scale=[0.25, 0.25],
        size=(40, 2),
    )

    regime_two = rng.normal(
        loc=[1.0, -1.0],
        scale=[0.25, 0.25],
        size=(40, 2),
    )

    return np.vstack(
        [
            regime_one,
            regime_two,
        ]
    )


@pytest.fixture
def config():
    return HMMConfig(
        model_name="gmm_hmm",
        n_regimes=2,
        n_mix=2,
        n_iter=50,
        random_state=42,
    )


@pytest.fixture
def model(config):
    return GMMHMMRegimeModel(config)


def test_default_model_creation():
    model = GMMHMMRegimeModel()

    assert isinstance(model.gmm_config, HMMConfig)
    assert model.model_name == "gmm_hmm"
    assert model.model is None
    assert model.n_features is None
    assert model.is_fitted is False


def test_custom_model_creation(model):
    assert model.n_regimes == 2
    assert model.gmm_config.n_mix == 2
    assert model.is_fitted is False


def test_invalid_config_rejected():
    with pytest.raises(TypeError):
        GMMHMMRegimeModel(config={})


def test_create_model(model):
    underlying_model = model._create_model()

    assert isinstance(underlying_model, GMMHMM)
    assert underlying_model.n_components == 2
    assert underlying_model.n_mix == 2


def test_validate_input_returns_float_array(model):
    result = model.validate_input(
        [
            [1, 2],
            [3, 4],
        ]
    )

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
def test_empty_input_rejected(model, data):
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


def test_non_2d_input_rejected(model):
    with pytest.raises(ValueError):
        model.validate_input([1, 2, 3])


def test_non_finite_input_rejected(model):
    with pytest.raises(ValueError):
        model.validate_input(
            [
                [1.0, np.nan],
                [2.0, 3.0],
            ]
        )


def test_insufficient_samples_rejected(model):
    with pytest.raises(ValueError):
        model.fit([[1.0, 2.0]])


def test_fit(model, sample_data):
    returned = model.fit(sample_data)

    assert returned is model
    assert model.is_fitted is True
    assert model.model is not None
    assert model.n_features == 2


def test_predict_before_fit_rejected(
    model,
    sample_data,
):
    with pytest.raises(RuntimeError):
        model.predict(sample_data)


def test_predict_returns_standard_result(
    model,
    sample_data,
):
    model.fit(sample_data)

    result = model.predict(sample_data)

    assert isinstance(result, AdvancedModelResult)
    assert result.n_predictions == len(sample_data)
    assert result.has_probabilities is True
    assert result.has_confidence is True
    assert result.model_name == "gmm_hmm"


def test_regime_indices_are_valid(
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

    for row in probabilities:
        assert sum(row) == pytest.approx(1.0)


def test_confidence_matches_max_probability(
    model,
    sample_data,
):
    model.fit(sample_data)

    result = model.predict(sample_data)

    for probabilities, confidence in zip(
        result.probabilities,
        result.confidence,
    ):
        assert confidence == pytest.approx(
            max(probabilities)
        )


def test_prediction_feature_mismatch_rejected(
    model,
    sample_data,
):
    model.fit(sample_data)

    with pytest.raises(ValueError):
        model.predict(np.ones((10, 3)))


def test_score_returns_float(model, sample_data):
    model.fit(sample_data)

    assert isinstance(
        model.score(sample_data),
        float,
    )


def test_transition_matrix(model, sample_data):
    model.fit(sample_data)

    matrix = model.get_transition_matrix()

    assert len(matrix) == 2

    for row in matrix:
        assert len(row) == 2
        assert sum(row) == pytest.approx(1.0)


def test_state_counts(model, sample_data):
    model.fit(sample_data)

    counts = model.get_state_counts(sample_data)

    assert len(counts) == 2
    assert sum(counts) == len(sample_data)


def test_diagnostics(model, sample_data):
    model.fit(sample_data)

    diagnostics = model.get_diagnostics()

    assert "converged" in diagnostics
    assert "iterations" in diagnostics
    assert "log_likelihood_history" in diagnostics
    assert diagnostics["n_mix"] == 2


def test_predict_metadata(model, sample_data):
    model.fit(sample_data)

    result = model.predict(sample_data)

    assert result.metadata["n_regimes"] == 2
    assert result.metadata["n_mix"] == 2
    assert result.metadata["n_features"] == 2
    assert result.metadata["algorithm"] == "viterbi"


@pytest.mark.parametrize(
    "n_mix",
    [
        0,
        -1,
    ],
)
def test_invalid_n_mix_value_rejected(n_mix):
    with pytest.raises(ValueError):
        HMMConfig(n_mix=n_mix)


@pytest.mark.parametrize(
    "n_mix",
    [
        1.5,
        "2",
        True,
        None,
    ],
)
def test_invalid_n_mix_type_rejected(n_mix):
    with pytest.raises(TypeError):
        HMMConfig(n_mix=n_mix)


@pytest.mark.parametrize(
    "algorithm",
    [
        "viterbi",
        "map",
        " VITERBI ",
    ],
)
def test_valid_algorithm(algorithm):
    config = HMMConfig(
        algorithm=algorithm
    )

    assert config.algorithm in {
        "viterbi",
        "map",
    }


@pytest.mark.parametrize(
    "algorithm",
    [
        "",
        "invalid",
        123,
        None,
    ],
)
def test_invalid_algorithm_rejected(algorithm):
    expected_error = (
        TypeError
        if not isinstance(algorithm, str)
        else ValueError
    )

    with pytest.raises(expected_error):
        HMMConfig(
            algorithm=algorithm
        )


def test_reproducible_predictions(sample_data):
    config_one = HMMConfig(
        model_name="gmm_hmm",
        n_regimes=2,
        n_mix=2,
        n_iter=50,
        random_state=123,
    )

    config_two = HMMConfig(
        model_name="gmm_hmm",
        n_regimes=2,
        n_mix=2,
        n_iter=50,
        random_state=123,
    )

    model_one = GMMHMMRegimeModel(config_one)
    model_two = GMMHMMRegimeModel(config_two)

    model_one.fit(sample_data)
    model_two.fit(sample_data)

    result_one = model_one.predict(sample_data)
    result_two = model_two.predict(sample_data)

    assert result_one.regimes == result_two.regimes