import math

import pytest

from src.ensemble.weighting import (
    EnsembleWeighting,
)


def test_equal_weights_for_three_models():
    weights = EnsembleWeighting.equal_weights(3)

    assert len(weights) == 3
    assert weights == pytest.approx(
        (
            1 / 3,
            1 / 3,
            1 / 3,
        )
    )
    assert sum(weights) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "n_models",
    [
        1,
        2,
        5,
        10,
    ],
)
def test_equal_weights_sum_to_one(n_models):
    weights = EnsembleWeighting.equal_weights(
        n_models
    )

    assert len(weights) == n_models
    assert sum(weights) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "n_models",
    [
        0,
        -1,
    ],
)
def test_invalid_model_count_value_rejected(n_models):
    with pytest.raises(ValueError):
        EnsembleWeighting.equal_weights(
            n_models
        )


@pytest.mark.parametrize(
    "n_models",
    [
        1.5,
        "3",
        None,
        True,
    ],
)
def test_invalid_model_count_type_rejected(n_models):
    with pytest.raises(TypeError):
        EnsembleWeighting.equal_weights(
            n_models
        )


def test_valid_weights():
    weights = EnsembleWeighting.validate_weights(
        [
            0.2,
            0.3,
            0.5,
        ]
    )

    assert weights == (
        0.2,
        0.3,
        0.5,
    )


def test_integer_weights_are_converted_to_float():
    weights = EnsembleWeighting.validate_weights(
        [
            1,
            2,
            3,
        ]
    )

    assert weights == (
        1.0,
        2.0,
        3.0,
    )


def test_weights_match_model_count():
    weights = EnsembleWeighting.validate_weights(
        [
            1,
            2,
            3,
        ],
        n_models=3,
    )

    assert len(weights) == 3


def test_weights_model_count_mismatch_rejected():
    with pytest.raises(ValueError):
        EnsembleWeighting.validate_weights(
            [
                1,
                2,
            ],
            n_models=3,
        )


@pytest.mark.parametrize(
    "weights",
    [
        [],
    ],
)
def test_empty_weights_rejected(weights):
    with pytest.raises(ValueError):
        EnsembleWeighting.validate_weights(
            weights
        )


@pytest.mark.parametrize(
    "weights",
    [
        "0.5,0.5",
        b"0.5,0.5",
        0.5,
        None,
    ],
)
def test_invalid_weight_sequence_rejected(weights):
    with pytest.raises(TypeError):
        EnsembleWeighting.validate_weights(
            weights
        )


@pytest.mark.parametrize(
    "weights",
    [
        [-1.0, 1.0],
        [0.5, -0.5],
    ],
)
def test_negative_weights_rejected(weights):
    with pytest.raises(ValueError):
        EnsembleWeighting.validate_weights(
            weights
        )


def test_all_zero_weights_rejected():
    with pytest.raises(ValueError):
        EnsembleWeighting.validate_weights(
            [
                0.0,
                0.0,
                0.0,
            ]
        )


@pytest.mark.parametrize(
    "weights",
    [
        [math.nan, 1.0],
        [math.inf, 1.0],
        [-math.inf, 1.0],
    ],
)
def test_non_finite_weights_rejected(weights):
    with pytest.raises(ValueError):
        EnsembleWeighting.validate_weights(
            weights
        )


@pytest.mark.parametrize(
    "weights",
    [
        ["0.5", 0.5],
        [None, 0.5],
        [True, 0.5],
    ],
)
def test_invalid_weight_types_rejected(weights):
    with pytest.raises(TypeError):
        EnsembleWeighting.validate_weights(
            weights
        )


def test_normalize_weights():
    weights = EnsembleWeighting.normalize_weights(
        [
            2,
            3,
            5,
        ]
    )

    assert weights == pytest.approx(
        (
            0.2,
            0.3,
            0.5,
        )
    )
    assert sum(weights) == pytest.approx(1.0)


def test_normalize_weights_preserves_proportions():
    weights = EnsembleWeighting.normalize_weights(
        [
            1,
            1,
            2,
        ]
    )

    assert weights == pytest.approx(
        (
            0.25,
            0.25,
            0.5,
        )
    )


def test_normalize_weights_with_model_count():
    weights = EnsembleWeighting.normalize_weights(
        [
            1,
            2,
        ],
        n_models=2,
    )

    assert sum(weights) == pytest.approx(1.0)


def test_equal_weight_aggregation():
    result = EnsembleWeighting.aggregate(
        [
            1.0,
            2.0,
            3.0,
        ]
    )

    assert result == pytest.approx(2.0)


def test_weighted_aggregation():
    result = EnsembleWeighting.aggregate(
        values=[
            10.0,
            20.0,
            30.0,
        ],
        weights=[
            0.2,
            0.3,
            0.5,
        ],
    )

    assert result == pytest.approx(23.0)


def test_aggregation_normalizes_weights():
    result = EnsembleWeighting.aggregate(
        values=[
            10.0,
            20.0,
        ],
        weights=[
            1.0,
            3.0,
        ],
    )

    assert result == pytest.approx(17.5)


def test_single_value_aggregation():
    result = EnsembleWeighting.aggregate(
        [
            42.0,
        ]
    )

    assert result == pytest.approx(42.0)


def test_aggregation_weight_length_mismatch_rejected():
    with pytest.raises(ValueError):
        EnsembleWeighting.aggregate(
            values=[
                1.0,
                2.0,
            ],
            weights=[
                1.0,
            ],
        )


@pytest.mark.parametrize(
    "values",
    [
        [],
    ],
)
def test_empty_values_rejected(values):
    with pytest.raises(ValueError):
        EnsembleWeighting.aggregate(values)


@pytest.mark.parametrize(
    "values",
    [
        "1,2,3",
        b"1,2,3",
        1.0,
        None,
    ],
)
def test_invalid_values_sequence_rejected(values):
    with pytest.raises(TypeError):
        EnsembleWeighting.aggregate(values)


@pytest.mark.parametrize(
    "values",
    [
        [1.0, math.nan],
        [1.0, math.inf],
        [1.0, -math.inf],
    ],
)
def test_non_finite_values_rejected(values):
    with pytest.raises(ValueError):
        EnsembleWeighting.aggregate(values)


@pytest.mark.parametrize(
    "values",
    [
        [1.0, "2.0"],
        [1.0, None],
        [1.0, True],
    ],
)
def test_invalid_value_types_rejected(values):
    with pytest.raises(TypeError):
        EnsembleWeighting.aggregate(values)