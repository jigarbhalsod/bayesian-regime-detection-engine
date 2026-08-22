import pytest

from src.models.config import AdvancedModelConfig
from src.models.hmm.config import HMMConfig


def test_default_configuration():
    config = HMMConfig()

    assert isinstance(config, AdvancedModelConfig)

    assert config.model_name == "gaussian_hmm"
    assert config.feature_columns is None
    assert config.n_regimes == 3
    assert config.random_state == 42

    assert config.covariance_type == "diag"
    assert config.n_iter == 100
    assert config.tol == 1e-3
    assert config.min_covar == 1e-3
    assert config.init_params == "stmc"
    assert config.params == "stmc"
    assert config.verbose is False


def test_custom_configuration():
    config = HMMConfig(
        model_name="custom_hmm",
        feature_columns=[
            "returns",
            "volatility",
        ],
        n_regimes=4,
        random_state=123,
        model_params={
            "custom_parameter": "value",
        },
        covariance_type="FULL",
        n_iter=250,
        tol=0.01,
        min_covar=0.0,
        init_params="stm",
        params="mc",
        verbose=True,
    )

    assert config.model_name == "custom_hmm"
    assert config.feature_columns == [
        "returns",
        "volatility",
    ]
    assert config.n_regimes == 4
    assert config.random_state == 123

    assert config.covariance_type == "full"
    assert config.n_iter == 250
    assert config.tol == 0.01
    assert config.min_covar == 0.0
    assert config.init_params == "stm"
    assert config.params == "mc"
    assert config.verbose is True

    assert (
        config.get_param("custom_parameter")
        == "value"
    )


@pytest.mark.parametrize(
    "value",
    [
        "spherical",
        "diag",
        "full",
        "tied",
        " FULL ",
    ],
)
def test_valid_covariance_types(value):
    config = HMMConfig(covariance_type=value)

    assert config.covariance_type in {
        "spherical",
        "diag",
        "full",
        "tied",
    }


@pytest.mark.parametrize(
    "value",
    [
        "",
        "   ",
        "invalid",
        "diagonal",
        123,
        None,
    ],
)
def test_invalid_covariance_type_rejected(value):
    expected_error = (
        TypeError
        if not isinstance(value, str)
        else ValueError
    )

    with pytest.raises(expected_error):
        HMMConfig(covariance_type=value)


@pytest.mark.parametrize(
    "value",
    [
        1,
        10,
        100,
        500,
    ],
)
def test_valid_n_iter(value):
    config = HMMConfig(n_iter=value)

    assert config.n_iter == value


@pytest.mark.parametrize(
    "value",
    [
        0,
        -1,
        -100,
    ],
)
def test_invalid_n_iter_value_rejected(value):
    with pytest.raises(ValueError):
        HMMConfig(n_iter=value)


@pytest.mark.parametrize(
    "value",
    [
        1.5,
        "100",
        True,
        None,
    ],
)
def test_invalid_n_iter_type_rejected(value):
    with pytest.raises(TypeError):
        HMMConfig(n_iter=value)


@pytest.mark.parametrize(
    "field_name",
    [
        "tol",
        "min_covar",
    ],
)
def test_numeric_parameters_are_converted_to_float(field_name):
    config = HMMConfig(**{field_name: 1})

    assert getattr(config, field_name) == 1.0


def test_valid_tol():
    config = HMMConfig(tol=0.0001)

    assert config.tol == 0.0001


@pytest.mark.parametrize(
    "value",
    [
        0,
        -0.1,
        -1,
    ],
)
def test_invalid_tol_value_rejected(value):
    with pytest.raises(ValueError):
        HMMConfig(tol=value)


@pytest.mark.parametrize(
    "value",
    [
        "0.01",
        True,
        None,
    ],
)
def test_invalid_tol_type_rejected(value):
    with pytest.raises(TypeError):
        HMMConfig(tol=value)


@pytest.mark.parametrize(
    "value",
    [
        0,
        0.001,
        1,
    ],
)
def test_valid_min_covar(value):
    config = HMMConfig(min_covar=value)

    assert config.min_covar == float(value)


@pytest.mark.parametrize(
    "value",
    [
        -0.001,
        -1,
    ],
)
def test_invalid_min_covar_value_rejected(value):
    with pytest.raises(ValueError):
        HMMConfig(min_covar=value)


@pytest.mark.parametrize(
    "value",
    [
        "0.001",
        True,
        None,
    ],
)
def test_invalid_min_covar_type_rejected(value):
    with pytest.raises(TypeError):
        HMMConfig(min_covar=value)


@pytest.mark.parametrize(
    "field_name",
    [
        "init_params",
        "params",
    ],
)
@pytest.mark.parametrize(
    "value",
    [
        "s",
        "t",
        "m",
        "c",
        "st",
        "stm",
        "stmc",
        "mc",
        " STMC ",
    ],
)
def test_valid_parameter_codes(field_name, value):
    config = HMMConfig(**{field_name: value})

    normalized = value.strip().lower()

    assert getattr(config, field_name) == normalized


@pytest.mark.parametrize(
    "field_name",
    [
        "init_params",
        "params",
    ],
)
@pytest.mark.parametrize(
    "value",
    [
        "",
        "   ",
        "x",
        "stx",
        "ss",
        "stmm",
        123,
        None,
    ],
)
def test_invalid_parameter_codes_rejected(field_name, value):
    expected_error = (
        TypeError
        if not isinstance(value, str)
        else ValueError
    )

    with pytest.raises(expected_error):
        HMMConfig(**{field_name: value})


def test_verbose_true_allowed():
    config = HMMConfig(verbose=True)

    assert config.verbose is True


def test_verbose_false_allowed():
    config = HMMConfig(verbose=False)

    assert config.verbose is False


@pytest.mark.parametrize(
    "value",
    [
        1,
        0,
        "true",
        None,
    ],
)
def test_invalid_verbose_rejected(value):
    with pytest.raises(TypeError):
        HMMConfig(verbose=value)


def test_inherited_configuration_validation():
    with pytest.raises(ValueError):
        HMMConfig(n_regimes=0)


def test_model_specific_params_are_supported():
    config = HMMConfig(
        model_params={
            "implementation": "future_backend",
        }
    )

    assert (
        config.get_param("implementation")
        == "future_backend"
    )


def test_to_dict_contains_all_common_and_hmm_fields():
    config = HMMConfig(
        feature_columns=["returns"],
        model_params={"alpha": 1},
        covariance_type="full",
        n_iter=200,
        tol=0.01,
        min_covar=0.1,
        init_params="stm",
        params="mc",
        verbose=True,
    )

    data = config.to_dict()

    assert data["model_name"] == "gaussian_hmm"
    assert data["feature_columns"] == ["returns"]
    assert data["n_regimes"] == 3
    assert data["random_state"] == 42
    assert data["model_params"] == {"alpha": 1}

    assert data["covariance_type"] == "full"
    assert data["n_iter"] == 200
    assert data["tol"] == 0.01
    assert data["min_covar"] == 0.1
    assert data["init_params"] == "stm"
    assert data["params"] == "mc"
    assert data["verbose"] is True


def test_to_dict_returns_independent_collections():
    config = HMMConfig(
        feature_columns=["returns"],
        model_params={"alpha": 1},
    )

    data = config.to_dict()

    data["feature_columns"].append("volatility")
    data["model_params"]["alpha"] = 99

    assert config.feature_columns == ["returns"]
    assert config.model_params == {"alpha": 1}