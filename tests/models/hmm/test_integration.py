import numpy as np
import pytest

from src.models.hmm.config import HMMConfig
from src.models.hmm.gmm import GMMHMMRegimeModel
from src.models.hmm.integration import HMMRegimeIntegration
from src.models.hmm.model import GaussianHMMRegimeModel
from src.models.result import AdvancedModelResult


@pytest.fixture
def sample_data():
    rng = np.random.RandomState(42)

    regime_one = rng.normal(
        loc=[-1.0, 1.0],
        scale=[0.25, 0.25],
        size=(50, 2),
    )

    regime_two = rng.normal(
        loc=[1.0, -1.0],
        scale=[0.25, 0.25],
        size=(50, 2),
    )

    return np.vstack(
        [
            regime_one,
            regime_two,
        ]
    )


@pytest.fixture
def gaussian_config():
    return HMMConfig(
        model_name="gaussian_hmm",
        n_regimes=2,
        n_iter=50,
        random_state=42,
    )


@pytest.fixture
def gmm_config():
    return HMMConfig(
        model_name="gmm_hmm",
        n_regimes=2,
        n_mix=2,
        n_iter=50,
        random_state=42,
    )


def test_default_integration_uses_gaussian_hmm():
    integration = HMMRegimeIntegration()

    assert integration.model_type == "gaussian_hmm"
    assert isinstance(
        integration.model,
        GaussianHMMRegimeModel,
    )
    assert integration.is_fitted is False


def test_gaussian_model_creation(gaussian_config):
    integration = HMMRegimeIntegration(
        gaussian_config
    )

    assert isinstance(
        integration.model,
        GaussianHMMRegimeModel,
    )
    assert integration.model_type == "gaussian_hmm"


def test_gmm_model_creation(gmm_config):
    integration = HMMRegimeIntegration(
        gmm_config
    )

    assert isinstance(
        integration.model,
        GMMHMMRegimeModel,
    )
    assert integration.model_type == "gmm_hmm"


def test_invalid_config_type_rejected():
    with pytest.raises(TypeError):
        HMMRegimeIntegration(config={})


@pytest.mark.parametrize(
    "model_name",
    [
        "invalid",
        "hmm",
        "lstm",
    ],
)
def test_invalid_model_name_rejected(model_name):
    config = HMMConfig(
        model_name=model_name
    )

    with pytest.raises(ValueError):
        HMMRegimeIntegration(config)


def test_empty_model_name_rejected():
    with pytest.raises(ValueError):
        HMMConfig(
            model_name=""
        )


@pytest.mark.parametrize(
    "model_name",
    [
        " GAUSSIAN_HMM ",
        "gMm_HmM",
    ],
)
def test_model_name_normalization(model_name):
    config = HMMConfig(
        model_name=model_name
    )

    integration = HMMRegimeIntegration(config)

    assert integration.model_type in {
        "gaussian_hmm",
        "gmm_hmm",
    }


def test_supported_model_types():
    assert HMMRegimeIntegration.supported_model_types() == [
        "gaussian_hmm",
        "gmm_hmm",
    ]


def test_fit_returns_self(
    gaussian_config,
    sample_data,
):
    integration = HMMRegimeIntegration(
        gaussian_config
    )

    returned = integration.fit(sample_data)

    assert returned is integration
    assert integration.is_fitted is True


@pytest.mark.parametrize(
    "config_name",
    [
        "gaussian",
        "gmm",
    ],
)
def test_predict_before_fit_rejected(
    config_name,
    gaussian_config,
    gmm_config,
    sample_data,
):
    config = (
        gaussian_config
        if config_name == "gaussian"
        else gmm_config
    )

    integration = HMMRegimeIntegration(config)

    with pytest.raises(RuntimeError):
        integration.predict(sample_data)


@pytest.mark.parametrize(
    "config_name",
    [
        "gaussian",
        "gmm",
    ],
)
def test_fit_and_predict(
    config_name,
    gaussian_config,
    gmm_config,
    sample_data,
):
    config = (
        gaussian_config
        if config_name == "gaussian"
        else gmm_config
    )

    integration = HMMRegimeIntegration(config)

    integration.fit(sample_data)
    result = integration.predict(sample_data)

    assert isinstance(result, AdvancedModelResult)
    assert result.n_predictions == len(sample_data)
    assert result.has_probabilities is True
    assert result.has_confidence is True


@pytest.mark.parametrize(
    "config_name",
    [
        "gaussian",
        "gmm",
    ],
)
def test_predict_probability_rows_sum_to_one(
    config_name,
    gaussian_config,
    gmm_config,
    sample_data,
):
    config = (
        gaussian_config
        if config_name == "gaussian"
        else gmm_config
    )

    integration = HMMRegimeIntegration(config)

    integration.fit(sample_data)

    probabilities = integration.predict_proba(
        sample_data
    )

    for row in probabilities:
        assert sum(row) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "config_name",
    [
        "gaussian",
        "gmm",
    ],
)
def test_score_returns_float(
    config_name,
    gaussian_config,
    gmm_config,
    sample_data,
):
    config = (
        gaussian_config
        if config_name == "gaussian"
        else gmm_config
    )

    integration = HMMRegimeIntegration(config)

    integration.fit(sample_data)

    assert isinstance(
        integration.score(sample_data),
        float,
    )


@pytest.mark.parametrize(
    "config_name",
    [
        "gaussian",
        "gmm",
    ],
)
def test_transition_matrix(
    config_name,
    gaussian_config,
    gmm_config,
    sample_data,
):
    config = (
        gaussian_config
        if config_name == "gaussian"
        else gmm_config
    )

    integration = HMMRegimeIntegration(config)

    integration.fit(sample_data)

    matrix = integration.get_transition_matrix()

    assert len(matrix) == 2

    for row in matrix:
        assert len(row) == 2
        assert sum(row) == pytest.approx(1.0)


def test_gaussian_metadata_before_fit(
    gaussian_config,
):
    integration = HMMRegimeIntegration(
        gaussian_config
    )

    metadata = integration.get_model_metadata()

    assert metadata["model_type"] == "gaussian_hmm"
    assert metadata["model_name"] == "gaussian_hmm"
    assert metadata["is_fitted"] is False
    assert metadata["n_regimes"] == 2
    assert metadata["n_mix"] == 2
    assert metadata["n_features"] is None


def test_gmm_metadata_after_fit(
    gmm_config,
    sample_data,
):
    integration = HMMRegimeIntegration(
        gmm_config
    )

    integration.fit(sample_data)

    metadata = integration.get_model_metadata()

    assert metadata["model_type"] == "gmm_hmm"
    assert metadata["model_name"] == "gmm_hmm"
    assert metadata["is_fitted"] is True
    assert metadata["n_regimes"] == 2
    assert metadata["n_mix"] == 2
    assert metadata["n_features"] == 2


@pytest.mark.parametrize(
    "config_name",
    [
        "gaussian",
        "gmm",
    ],
)
def test_diagnostics_available_after_fit(
    config_name,
    gaussian_config,
    gmm_config,
    sample_data,
):
    config = (
        gaussian_config
        if config_name == "gaussian"
        else gmm_config
    )

    integration = HMMRegimeIntegration(config)

    integration.fit(sample_data)

    diagnostics = integration.get_diagnostics()

    assert isinstance(diagnostics, dict)
    assert "converged" in diagnostics
    assert "iterations" in diagnostics


def test_get_diagnostics_before_fit_rejected(
    gaussian_config,
):
    integration = HMMRegimeIntegration(
        gaussian_config
    )

    with pytest.raises(RuntimeError):
        integration.get_diagnostics()