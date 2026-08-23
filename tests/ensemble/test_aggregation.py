import math

import pytest

from src.ensemble.aggregation import (
    EnsembleAggregator,
)


def test_mean():
    result = EnsembleAggregator.mean(
        [
            1.0,
            2.0,
            3.0,
        ]
    )

    assert result == pytest.approx(2.0)


def test_mean_single_value():
    result = EnsembleAggregator.mean(
        [
            42.0,
        ]
    )

    assert result == pytest.approx(42.0)


def test_mean_integer_values():
    result = EnsembleAggregator.mean(
        [
            1,
            2,
            3,
            4,
        ]
    )

    assert result == pytest.approx(2.5)


@pytest.mark.parametrize(
    "values",
    [
        [],
    ],
)
def test_mean_empty_values_rejected(values):
    with pytest.raises(ValueError):
        EnsembleAggregator.mean(values)


@pytest.mark.parametrize(
    "values",
    [
        "1,2,3",
        b"1,2,3",
        1.0,
        None,
    ],
)
def test_mean_invalid_sequence_rejected(values):
    with pytest.raises(TypeError):
        EnsembleAggregator.mean(values)


@pytest.mark.parametrize(
    "values",
    [
        [1.0, math.nan],
        [1.0, math.inf],
        [1.0, -math.inf],
    ],
)
def test_mean_non_finite_values_rejected(values):
    with pytest.raises(ValueError):
        EnsembleAggregator.mean(values)


@pytest.mark.parametrize(
    "values",
    [
        [1.0, "2.0"],
        [1.0, None],
        [1.0, True],
    ],
)
def test_mean_invalid_value_types_rejected(values):
    with pytest.raises(TypeError):
        EnsembleAggregator.mean(values)


def test_weighted_mean_equal_weighting():
    result = EnsembleAggregator.weighted_mean(
        [
            10.0,
            20.0,
            30.0,
        ]
    )

    assert result == pytest.approx(20.0)


def test_weighted_mean_custom_weights():
    result = EnsembleAggregator.weighted_mean(
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


def test_weighted_mean_normalizes_weights():
    result = EnsembleAggregator.weighted_mean(
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


def test_weighted_mean_weight_length_mismatch_rejected():
    with pytest.raises(ValueError):
        EnsembleAggregator.weighted_mean(
            values=[
                1.0,
                2.0,
            ],
            weights=[
                1.0,
            ],
        )


def test_aggregate_vectors_equal_weights():
    result = EnsembleAggregator.aggregate_vectors(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    assert result == pytest.approx(
        (
            2.0,
            3.0,
        )
    )


def test_aggregate_vectors_custom_weights():
    result = EnsembleAggregator.aggregate_vectors(
        vectors=[
            [10.0, 20.0],
            [30.0, 40.0],
        ],
        weights=[
            0.25,
            0.75,
        ],
    )

    assert result == pytest.approx(
        (
            25.0,
            35.0,
        )
    )


def test_aggregate_vectors_normalizes_weights():
    result = EnsembleAggregator.aggregate_vectors(
        vectors=[
            [10.0, 20.0],
            [30.0, 40.0],
        ],
        weights=[
            1.0,
            3.0,
        ],
    )

    assert result == pytest.approx(
        (
            25.0,
            35.0,
        )
    )


def test_aggregate_three_vectors():
    result = EnsembleAggregator.aggregate_vectors(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
    )

    assert result == pytest.approx(
        (
            4.0,
            5.0,
            6.0,
        )
    )


def test_aggregate_single_vector():
    result = EnsembleAggregator.aggregate_vectors(
        [
            [1.0, 2.0, 3.0],
        ]
    )

    assert result == pytest.approx(
        (
            1.0,
            2.0,
            3.0,
        )
    )


def test_empty_vectors_rejected():
    with pytest.raises(ValueError):
        EnsembleAggregator.aggregate_vectors([])


@pytest.mark.parametrize(
    "vectors",
    [
        "invalid",
        b"invalid",
        123,
        None,
    ],
)
def test_invalid_vectors_container_rejected(vectors):
    with pytest.raises(TypeError):
        EnsembleAggregator.aggregate_vectors(vectors)


@pytest.mark.parametrize(
    "vectors",
    [
        [
            [1.0, 2.0],
            [3.0],
        ],
        [
            [1.0],
            [2.0, 3.0],
        ],
    ],
)
def test_mismatched_vector_lengths_rejected(vectors):
    with pytest.raises(ValueError):
        EnsembleAggregator.aggregate_vectors(vectors)


@pytest.mark.parametrize(
    "vectors",
    [
        [
            [],
        ],
        [
            [1.0, 2.0],
            [],
        ],
    ],
)
def test_empty_inner_vector_rejected(vectors):
    with pytest.raises(ValueError):
        EnsembleAggregator.aggregate_vectors(vectors)


@pytest.mark.parametrize(
    "vectors",
    [
        [
            "invalid",
        ],
        [
            123,
        ],
        [
            None,
        ],
    ],
)
def test_invalid_inner_vector_rejected(vectors):
    with pytest.raises(TypeError):
        EnsembleAggregator.aggregate_vectors(vectors)


@pytest.mark.parametrize(
    "vectors",
    [
        [
            [1.0, math.nan],
        ],
        [
            [math.inf, 2.0],
        ],
    ],
)
def test_non_finite_vector_values_rejected(vectors):
    with pytest.raises(ValueError):
        EnsembleAggregator.aggregate_vectors(vectors)


def test_vector_weight_length_mismatch_rejected():
    with pytest.raises(ValueError):
        EnsembleAggregator.aggregate_vectors(
            vectors=[
                [1.0, 2.0],
                [3.0, 4.0],
            ],
            weights=[
                1.0,
            ],
        )