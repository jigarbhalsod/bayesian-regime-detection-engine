import pytest

from src.models.config import AdvancedModelConfig


def test_default_configuration():
    config = AdvancedModelConfig()

    assert config.model_name == "advanced_model"
    assert config.feature_columns is None
    assert config.n_regimes == 3
    assert config.random_state == 42
    assert config.model_params == {}


def test_custom_configuration():
    config = AdvancedModelConfig(
        model_name="hmm",
        feature_columns=["returns", "volatility"],
        n_regimes=4,
        random_state=123,
        model_params={"covariance_type": "full"},
    )

    assert config.model_name == "hmm"
    assert config.feature_columns == [
        "returns",
        "volatility",
    ]
    assert config.n_regimes == 4
    assert config.random_state == 123
    assert config.get_param("covariance_type") == "full"


@pytest.mark.parametrize(
    "value",
    [
        "",
        "   ",
    ],
)
def test_empty_model_name_rejected(value):
    with pytest.raises(ValueError):
        AdvancedModelConfig(model_name=value)


def test_non_string_model_name_rejected():
    with pytest.raises(TypeError):
        AdvancedModelConfig(model_name=123)


def test_feature_columns_none_allowed():
    config = AdvancedModelConfig(feature_columns=None)

    assert config.feature_columns is None


def test_feature_columns_are_normalized():
    config = AdvancedModelConfig(
        feature_columns=[
            " returns ",
            "volatility",
        ]
    )

    assert config.feature_columns == [
        "returns",
        "volatility",
    ]


def test_invalid_feature_columns_container_rejected():
    with pytest.raises(TypeError):
        AdvancedModelConfig(feature_columns="returns")


def test_invalid_feature_column_type_rejected():
    with pytest.raises(TypeError):
        AdvancedModelConfig(
            feature_columns=["returns", 123]
        )


def test_empty_feature_column_rejected():
    with pytest.raises(ValueError):
        AdvancedModelConfig(
            feature_columns=["returns", " "]
        )


def test_duplicate_feature_columns_rejected():
    with pytest.raises(ValueError):
        AdvancedModelConfig(
            feature_columns=[
                "returns",
                "returns",
            ]
        )


@pytest.mark.parametrize(
    "value",
    [
        0,
        -1,
    ],
)
def test_invalid_n_regimes_rejected(value):
    with pytest.raises(ValueError):
        AdvancedModelConfig(n_regimes=value)


@pytest.mark.parametrize(
    "value",
    [
        1.5,
        "3",
        True,
    ],
)
def test_non_integer_n_regimes_rejected(value):
    with pytest.raises(TypeError):
        AdvancedModelConfig(n_regimes=value)


def test_random_state_none_allowed():
    config = AdvancedModelConfig(random_state=None)

    assert config.random_state is None


@pytest.mark.parametrize(
    "value",
    [
        "42",
        1.5,
        True,
    ],
)
def test_invalid_random_state_rejected(value):
    with pytest.raises(TypeError):
        AdvancedModelConfig(random_state=value)


def test_model_params_none_becomes_empty_dict():
    config = AdvancedModelConfig(model_params=None)

    assert config.model_params == {}


def test_invalid_model_params_rejected():
    with pytest.raises(TypeError):
        AdvancedModelConfig(model_params=[])


def test_get_param_with_default():
    config = AdvancedModelConfig()

    assert config.get_param(
        "missing",
        "default_value",
    ) == "default_value"


def test_set_param():
    config = AdvancedModelConfig()

    config.set_param("learning_rate", 0.01)

    assert config.get_param("learning_rate") == 0.01


@pytest.mark.parametrize(
    "name",
    [
        "",
        "   ",
    ],
)
def test_empty_param_name_rejected(name):
    config = AdvancedModelConfig()

    with pytest.raises(ValueError):
        config.set_param(name, 1)


def test_non_string_param_name_rejected():
    config = AdvancedModelConfig()

    with pytest.raises(TypeError):
        config.set_param(123, 1)


def test_to_dict_returns_independent_collections():
    config = AdvancedModelConfig(
        feature_columns=["returns"],
        model_params={"alpha": 1},
    )

    result = config.to_dict()

    result["feature_columns"].append("volatility")
    result["model_params"]["alpha"] = 99

    assert config.feature_columns == ["returns"]
    assert config.model_params == {"alpha": 1}