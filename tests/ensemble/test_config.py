import math

import pytest

from src.ensemble.config import EnsembleConfig


def test_default_config():
    config = EnsembleConfig()

    assert config.method == "weighted_average"
    assert config.normalize_weights is True
    assert config.min_confidence == 0.0
    assert config.max_confidence == 1.0
    assert config.uncertainty_floor == 0.0
    assert config.uncertainty_ceiling == 1.0


def test_custom_config():
    config = EnsembleConfig(
        method="MEAN",
        normalize_weights=False,
        min_confidence=0.2,
        max_confidence=0.9,
        uncertainty_floor=0.1,
        uncertainty_ceiling=0.8,
    )

    assert config.method == "mean"
    assert config.normalize_weights is False
    assert config.min_confidence == 0.2
    assert config.max_confidence == 0.9
    assert config.uncertainty_floor == 0.1
    assert config.uncertainty_ceiling == 0.8


@pytest.mark.parametrize(
    "method",
    [
        "",
        "   ",
    ],
)
def test_empty_method_rejected(method):
    with pytest.raises(ValueError):
        EnsembleConfig(method=method)


@pytest.mark.parametrize(
    "method",
    [
        None,
        123,
        True,
    ],
)
def test_invalid_method_type_rejected(method):
    with pytest.raises(TypeError):
        EnsembleConfig(method=method)


def test_unknown_method_rejected():
    with pytest.raises(ValueError):
        EnsembleConfig(method="median")


@pytest.mark.parametrize(
    "value",
    [
        None,
        "weighted_average",
        1,
        0,
    ],
)
def test_normalize_weights_must_be_boolean(value):
    with pytest.raises(TypeError):
        EnsembleConfig(normalize_weights=value)


@pytest.mark.parametrize(
    "field_name",
    [
        "min_confidence",
        "max_confidence",
        "uncertainty_floor",
        "uncertainty_ceiling",
    ],
)
@pytest.mark.parametrize(
    "value",
    [
        -0.1,
        1.1,
        math.inf,
        -math.inf,
        math.nan,
    ],
)
def test_invalid_probability_values_rejected(
    field_name,
    value,
):
    with pytest.raises(ValueError):
        EnsembleConfig(**{field_name: value})


@pytest.mark.parametrize(
    "field_name",
    [
        "min_confidence",
        "max_confidence",
        "uncertainty_floor",
        "uncertainty_ceiling",
    ],
)
@pytest.mark.parametrize(
    "value",
    [
        None,
        "0.5",
        True,
    ],
)
def test_invalid_probability_types_rejected(
    field_name,
    value,
):
    with pytest.raises(TypeError):
        EnsembleConfig(**{field_name: value})


def test_invalid_confidence_range_rejected():
    with pytest.raises(ValueError):
        EnsembleConfig(
            min_confidence=0.8,
            max_confidence=0.2,
        )


def test_invalid_uncertainty_range_rejected():
    with pytest.raises(ValueError):
        EnsembleConfig(
            uncertainty_floor=0.9,
            uncertainty_ceiling=0.2,
        )


def test_config_is_frozen():
    config = EnsembleConfig()

    with pytest.raises(
        Exception,
    ):
        config.method = "mean"