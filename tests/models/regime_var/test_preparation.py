import numpy as np
import pytest

from src.models.regime_var.preparation import (
    RegimeVARPreparedData,
    RegimeVARPreparer,
)


def test_creation():
    preparer = RegimeVARPreparer(
        lag_order=2,
        n_features=3,
    )

    assert preparer.lag_order == 2
    assert preparer.n_features == 3


@pytest.mark.parametrize(
    "field_name",
    [
        "lag_order",
        "n_features",
    ],
)
@pytest.mark.parametrize(
    "value",
    [
        0,
        -1,
        -10,
    ],
)
def test_non_positive_creation_values_rejected(
    field_name,
    value,
):
    with pytest.raises(ValueError):
        RegimeVARPreparer(
            **{
                field_name: value,
                (
                    "n_features"
                    if field_name == "lag_order"
                    else "lag_order"
                ): 1,
            }
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "lag_order",
        "n_features",
    ],
)
@pytest.mark.parametrize(
    "value",
    [
        1.5,
        "1",
        None,
        True,
    ],
)
def test_invalid_creation_types_rejected(
    field_name,
    value,
):
    with pytest.raises(TypeError):
        RegimeVARPreparer(
            **{
                field_name: value,
                (
                    "n_features"
                    if field_name == "lag_order"
                    else "lag_order"
                ): 1,
            }
        )


def test_prepare_without_regimes():
    preparer = RegimeVARPreparer(
        lag_order=2,
        n_features=2,
    )

    data = np.array(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
        ]
    )

    result = preparer.prepare(data)

    assert isinstance(result, RegimeVARPreparedData)
    assert result.features.shape == (2, 4)
    assert result.targets.shape == (2, 2)
    assert result.regimes is None

    assert np.allclose(
        result.features[0],
        [1.0, 10.0, 2.0, 20.0],
    )
    assert np.allclose(
        result.targets[0],
        [3.0, 30.0],
    )


def test_prepare_with_regimes():
    preparer = RegimeVARPreparer(
        lag_order=2,
        n_features=1,
    )

    data = np.array(
        [
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
        ]
    )

    regimes = np.array(
        [0, 0, 1, 1, 2],
        dtype=np.int64,
    )

    result = preparer.prepare(
        data=data,
        regimes=regimes,
    )

    assert result.features.shape == (3, 2)
    assert result.targets.shape == (3, 1)
    assert np.array_equal(
        result.regimes,
        [1, 1, 2],
    )


def test_prepare_preserves_target_alignment():
    preparer = RegimeVARPreparer(
        lag_order=3,
        n_features=1,
    )

    data = np.arange(
        1,
        8,
        dtype=np.float64,
    ).reshape(-1, 1)

    regimes = np.arange(
        7,
        dtype=np.int64,
    )

    result = preparer.prepare(
        data=data,
        regimes=regimes,
    )

    assert np.array_equal(
        result.features,
        np.array(
            [
                [1.0, 2.0, 3.0],
                [2.0, 3.0, 4.0],
                [3.0, 4.0, 5.0],
                [4.0, 5.0, 6.0],
            ]
        ),
    )

    assert np.array_equal(
        result.targets.reshape(-1),
        [4.0, 5.0, 6.0, 7.0],
    )

    assert np.array_equal(
        result.regimes,
        [3, 4, 5, 6],
    )


def test_prepare_rejects_non_array_data():
    preparer = RegimeVARPreparer(
        lag_order=1,
        n_features=1,
    )

    with pytest.raises(TypeError):
        preparer.prepare([[1.0], [2.0]])


@pytest.mark.parametrize(
    "data",
    [
        np.array([1.0, 2.0]),
        np.array([[[1.0]]]),
    ],
)
def test_prepare_rejects_invalid_data_dimensions(data):
    preparer = RegimeVARPreparer(
        lag_order=1,
        n_features=1,
    )

    with pytest.raises(ValueError):
        preparer.prepare(data)


def test_prepare_rejects_wrong_feature_count():
    preparer = RegimeVARPreparer(
        lag_order=1,
        n_features=2,
    )

    data = np.array(
        [
            [1.0],
            [2.0],
        ]
    )

    with pytest.raises(ValueError):
        preparer.prepare(data)


def test_prepare_rejects_non_numeric_data():
    preparer = RegimeVARPreparer(
        lag_order=1,
        n_features=1,
    )

    data = np.array(
        [["a"], ["b"]],
        dtype=object,
    )

    with pytest.raises(TypeError):
        preparer.prepare(data)


@pytest.mark.parametrize(
    "value",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_prepare_rejects_non_finite_data(value):
    preparer = RegimeVARPreparer(
        lag_order=1,
        n_features=1,
    )

    data = np.array(
        [[1.0], [value]],
        dtype=np.float64,
    )

    with pytest.raises(ValueError):
        preparer.prepare(data)


def test_prepare_rejects_insufficient_observations():
    preparer = RegimeVARPreparer(
        lag_order=2,
        n_features=1,
    )

    data = np.array(
        [[1.0], [2.0]],
        dtype=np.float64,
    )

    with pytest.raises(ValueError):
        preparer.prepare(data)


def test_prepare_rejects_non_array_regimes():
    preparer = RegimeVARPreparer(
        lag_order=1,
        n_features=1,
    )

    data = np.array(
        [[1.0], [2.0]],
    )

    with pytest.raises(TypeError):
        preparer.prepare(
            data=data,
            regimes=[0, 1],
        )


def test_prepare_rejects_invalid_regime_dimensions():
    preparer = RegimeVARPreparer(
        lag_order=1,
        n_features=1,
    )

    data = np.array(
        [[1.0], [2.0]],
    )

    regimes = np.array([[0, 1]])

    with pytest.raises(ValueError):
        preparer.prepare(
            data=data,
            regimes=regimes,
        )


def test_prepare_rejects_wrong_regime_length():
    preparer = RegimeVARPreparer(
        lag_order=1,
        n_features=1,
    )

    data = np.array(
        [[1.0], [2.0], [3.0]],
    )

    regimes = np.array([0, 1])

    with pytest.raises(ValueError):
        preparer.prepare(
            data=data,
            regimes=regimes,
        )


def test_prepare_rejects_non_integer_regimes():
    preparer = RegimeVARPreparer(
        lag_order=1,
        n_features=1,
    )

    data = np.array(
        [[1.0], [2.0], [3.0]],
    )

    regimes = np.array(
        [0.0, 1.0, 2.0],
    )

    with pytest.raises(TypeError):
        preparer.prepare(
            data=data,
            regimes=regimes,
        )


def test_integer_data_is_converted_to_float64():
    preparer = RegimeVARPreparer(
        lag_order=1,
        n_features=1,
    )

    data = np.array(
        [[1], [2], [3]],
        dtype=np.int64,
    )

    result = preparer.prepare(data)

    assert result.features.dtype == np.float64
    assert result.targets.dtype == np.float64