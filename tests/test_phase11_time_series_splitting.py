import pytest

from src.validation.splitting import (
    TimeSeriesSplit,
    TimeSeriesSplitter,
)


# ============================================================
# Configuration Tests
# ============================================================


def test_default_ratios():
    splitter = TimeSeriesSplitter()

    assert splitter.train_ratio == 0.70
    assert splitter.validation_ratio == 0.15
    assert splitter.test_ratio == 0.15


def test_custom_ratios():
    splitter = TimeSeriesSplitter(
        train_ratio=0.60,
        validation_ratio=0.20,
        test_ratio=0.20,
    )

    assert splitter.train_ratio == 0.60
    assert splitter.validation_ratio == 0.20
    assert splitter.test_ratio == 0.20


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "train_ratio": 0.50,
            "validation_ratio": 0.20,
            "test_ratio": 0.20,
        },
        {
            "train_ratio": 0.80,
            "validation_ratio": 0.10,
            "test_ratio": 0.20,
        },
    ],
)
def test_ratios_must_sum_to_one(kwargs):
    with pytest.raises(ValueError):
        TimeSeriesSplitter(**kwargs)


@pytest.mark.parametrize(
    "parameter",
    [
        "train_ratio",
        "validation_ratio",
        "test_ratio",
    ],
)
def test_ratio_must_be_numeric(parameter):
    kwargs = {
        "train_ratio": 0.70,
        "validation_ratio": 0.15,
        "test_ratio": 0.15,
    }
    kwargs[parameter] = "invalid"

    with pytest.raises(TypeError):
        TimeSeriesSplitter(**kwargs)


@pytest.mark.parametrize(
    "parameter",
    [
        "train_ratio",
        "validation_ratio",
        "test_ratio",
    ],
)
@pytest.mark.parametrize(
    "invalid_value",
    [
        0,
        -0.1,
        1.0,
        1.1,
        True,
    ],
)
def test_ratio_must_be_between_zero_and_one(
    parameter,
    invalid_value,
):
    kwargs = {
        "train_ratio": 0.70,
        "validation_ratio": 0.15,
        "test_ratio": 0.15,
    }
    kwargs[parameter] = invalid_value

    with pytest.raises((TypeError, ValueError)):
        TimeSeriesSplitter(**kwargs)


# ============================================================
# Split Tests
# ============================================================


def test_split_returns_time_series_split():
    splitter = TimeSeriesSplitter()
    result = splitter.split(list(range(100)))

    assert isinstance(result, TimeSeriesSplit)


def test_default_split_sizes():
    splitter = TimeSeriesSplitter()
    result = splitter.split(list(range(100)))

    assert len(result.train) == 70
    assert len(result.validation) == 15
    assert len(result.test) == 15


def test_split_preserves_chronological_order():
    splitter = TimeSeriesSplitter()
    data = list(range(100))

    result = splitter.split(data)

    assert list(result.train) == list(range(70))
    assert list(result.validation) == list(range(70, 85))
    assert list(result.test) == list(range(85, 100))


def test_no_data_leakage_between_splits():
    splitter = TimeSeriesSplitter()
    result = splitter.split(list(range(100)))

    assert max(result.train) < min(result.validation)
    assert max(result.validation) < min(result.test)


def test_all_observations_are_preserved():
    splitter = TimeSeriesSplitter()
    data = list(range(101))

    result = splitter.split(data)

    combined = (
        list(result.train)
        + list(result.validation)
        + list(result.test)
    )

    assert combined == data
    assert len(combined) == len(data)


def test_custom_split_sizes():
    splitter = TimeSeriesSplitter(
        train_ratio=0.60,
        validation_ratio=0.20,
        test_ratio=0.20,
    )

    result = splitter.split(list(range(100)))

    assert len(result.train) == 60
    assert len(result.validation) == 20
    assert len(result.test) == 20


def test_split_with_tuple():
    splitter = TimeSeriesSplitter()
    data = tuple(range(100))

    result = splitter.split(data)

    assert result.train == tuple(range(70))
    assert result.validation == tuple(range(70, 85))
    assert result.test == tuple(range(85, 100))


# ============================================================
# Invalid Input Tests
# ============================================================


@pytest.mark.parametrize(
    "data",
    [
        [],
        [1],
        [1, 2],
    ],
)
def test_data_requires_at_least_three_observations(data):
    splitter = TimeSeriesSplitter()

    with pytest.raises(ValueError):
        splitter.split(data)


@pytest.mark.parametrize(
    "data",
    [
        "invalid",
        b"invalid",
        123,
        {"a": 1},
    ],
)
def test_invalid_data_type(data):
    splitter = TimeSeriesSplitter()

    with pytest.raises(TypeError):
        splitter.split(data)