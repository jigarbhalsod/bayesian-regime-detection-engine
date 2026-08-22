import pytest

from src.models.bayesian.config import BayesianModelConfig


def test_default_configuration():
    config = BayesianModelConfig()

    assert config.model_name == "bayesian_neural"
    assert config.hidden_dims == (64, 32)
    assert config.dropout_rate == 0.2
    assert config.mc_samples == 30
    assert config.learning_rate == pytest.approx(1e-3)
    assert config.batch_size == 32
    assert config.n_epochs == 50
    assert config.weight_decay == 0.0


def test_custom_configuration():
    config = BayesianModelConfig(
        hidden_dims=[128, 64, 32],
        dropout_rate=0.3,
        mc_samples=50,
        learning_rate=0.005,
        batch_size=64,
        n_epochs=100,
        weight_decay=0.01,
    )

    assert config.hidden_dims == (128, 64, 32)
    assert config.dropout_rate == 0.3
    assert config.mc_samples == 50
    assert config.learning_rate == pytest.approx(0.005)
    assert config.batch_size == 64
    assert config.n_epochs == 100
    assert config.weight_decay == pytest.approx(0.01)


@pytest.mark.parametrize(
    "hidden_dims",
    [
        (),
        [],
    ],
)
def test_empty_hidden_dims_rejected(hidden_dims):
    with pytest.raises(ValueError):
        BayesianModelConfig(
            hidden_dims=hidden_dims
        )


@pytest.mark.parametrize(
    "hidden_dims",
    [
        "64,32",
        64,
        None,
    ],
)
def test_invalid_hidden_dims_container_rejected(hidden_dims):
    with pytest.raises(TypeError):
        BayesianModelConfig(
            hidden_dims=hidden_dims
        )


@pytest.mark.parametrize(
    "hidden_dims",
    [
        (64, 0),
        (-1, 32),
        (64, -5),
    ],
)
def test_non_positive_hidden_dimension_rejected(hidden_dims):
    with pytest.raises(ValueError):
        BayesianModelConfig(
            hidden_dims=hidden_dims
        )


@pytest.mark.parametrize(
    "hidden_dims",
    [
        (64, 32.0),
        (64, "32"),
        (True, 32),
        (None, 32),
    ],
)
def test_invalid_hidden_dimension_type_rejected(hidden_dims):
    with pytest.raises(TypeError):
        BayesianModelConfig(
            hidden_dims=hidden_dims
        )


@pytest.mark.parametrize(
    "dropout_rate",
    [
        0,
        0.2,
        0.999,
    ],
)
def test_valid_dropout_rate(dropout_rate):
    config = BayesianModelConfig(
        dropout_rate=dropout_rate
    )

    assert config.dropout_rate == pytest.approx(
        float(dropout_rate)
    )


@pytest.mark.parametrize(
    "dropout_rate",
    [
        -0.1,
        1,
        1.5,
    ],
)
def test_invalid_dropout_rate_value_rejected(dropout_rate):
    with pytest.raises(ValueError):
        BayesianModelConfig(
            dropout_rate=dropout_rate
        )


@pytest.mark.parametrize(
    "dropout_rate",
    [
        "0.2",
        None,
        True,
    ],
)
def test_invalid_dropout_rate_type_rejected(dropout_rate):
    with pytest.raises(TypeError):
        BayesianModelConfig(
            dropout_rate=dropout_rate
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "mc_samples",
        "batch_size",
        "n_epochs",
    ],
)
@pytest.mark.parametrize(
    "value",
    [
        0,
        -1,
    ],
)
def test_invalid_positive_integer_values(
    field_name,
    value,
):
    with pytest.raises(ValueError):
        BayesianModelConfig(
            **{field_name: value}
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "mc_samples",
        "batch_size",
        "n_epochs",
    ],
)
@pytest.mark.parametrize(
    "value",
    [
        1.5,
        "10",
        None,
        True,
    ],
)
def test_invalid_positive_integer_types(
    field_name,
    value,
):
    with pytest.raises(TypeError):
        BayesianModelConfig(
            **{field_name: value}
        )


@pytest.mark.parametrize(
    "learning_rate",
    [
        0,
        -0.001,
    ],
)
def test_invalid_learning_rate_value_rejected(
    learning_rate,
):
    with pytest.raises(ValueError):
        BayesianModelConfig(
            learning_rate=learning_rate
        )


@pytest.mark.parametrize(
    "learning_rate",
    [
        "0.001",
        None,
        True,
    ],
)
def test_invalid_learning_rate_type_rejected(
    learning_rate,
):
    with pytest.raises(TypeError):
        BayesianModelConfig(
            learning_rate=learning_rate
        )


@pytest.mark.parametrize(
    "weight_decay",
    [
        0,
        0.01,
        1,
    ],
)
def test_valid_weight_decay(weight_decay):
    config = BayesianModelConfig(
        weight_decay=weight_decay
    )

    assert config.weight_decay == pytest.approx(
        float(weight_decay)
    )


@pytest.mark.parametrize(
    "weight_decay",
    [
        -0.001,
        -1,
    ],
)
def test_negative_weight_decay_rejected(weight_decay):
    with pytest.raises(ValueError):
        BayesianModelConfig(
            weight_decay=weight_decay
        )


@pytest.mark.parametrize(
    "weight_decay",
    [
        "0.01",
        None,
        True,
    ],
)
def test_invalid_weight_decay_type_rejected(
    weight_decay,
):
    with pytest.raises(TypeError):
        BayesianModelConfig(
            weight_decay=weight_decay
        )


def test_hidden_dims_normalized_to_tuple():
    config = BayesianModelConfig(
        hidden_dims=[128, 64]
    )

    assert isinstance(config.hidden_dims, tuple)
    assert config.hidden_dims == (128, 64)


def test_to_dict_contains_bayesian_fields():
    config = BayesianModelConfig()

    data = config.to_dict()

    expected_fields = {
        "hidden_dims",
        "dropout_rate",
        "mc_samples",
        "learning_rate",
        "batch_size",
        "n_epochs",
        "weight_decay",
    }

    assert expected_fields.issubset(data.keys())

    assert data["hidden_dims"] == (64, 32)
    assert data["dropout_rate"] == 0.2
    assert data["mc_samples"] == 30