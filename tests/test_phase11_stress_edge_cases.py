import math

import pytest

from src.validation.stress_testing import (
    StressTestResult,
    StressTester,
)


# ============================================================
# Configuration Tests
# ============================================================


def test_default_extreme_threshold():
    tester = StressTester()

    assert tester.extreme_threshold == 1_000_000.0


def test_custom_extreme_threshold():
    tester = StressTester(
        extreme_threshold=100.0
    )

    assert tester.extreme_threshold == 100.0


@pytest.mark.parametrize(
    "threshold",
    [
        0,
        -1,
        "invalid",
        None,
        True,
        float("nan"),
        float("inf"),
        -float("inf"),
    ],
)
def test_invalid_extreme_threshold(threshold):
    with pytest.raises((TypeError, ValueError)):
        StressTester(
            extreme_threshold=threshold
        )


# ============================================================
# Basic Analysis Tests
# ============================================================


def test_analyze_returns_result():
    result = StressTester().analyze(
        [1.0, 2.0, 3.0]
    )

    assert isinstance(result, StressTestResult)


def test_sample_count():
    result = StressTester().analyze(
        [1.0, 2.0, 3.0, 4.0]
    )

    assert result.sample_count == 4


def test_minimum_and_maximum():
    result = StressTester().analyze(
        [-5.0, 0.0, 10.0, 3.0]
    )

    assert result.minimum == -5.0
    assert result.maximum == 10.0


def test_mean():
    result = StressTester().analyze(
        [1.0, 2.0, 3.0]
    )

    assert result.mean == 2.0


def test_standard_deviation():
    result = StressTester().analyze(
        [1.0, 3.0, 5.0]
    )

    assert math.isclose(
        result.standard_deviation,
        2.0,
        abs_tol=1e-12,
    )


def test_value_range():
    result = StressTester().analyze(
        [-2.0, 4.0, 8.0]
    )

    assert result.value_range == 10.0


# ============================================================
# Edge Cases
# ============================================================


def test_single_value():
    result = StressTester().analyze([5.0])

    assert result.sample_count == 1
    assert result.minimum == 5.0
    assert result.maximum == 5.0
    assert result.mean == 5.0
    assert result.standard_deviation == 0.0
    assert result.value_range == 0.0
    assert result.has_constant_values is True


def test_constant_values():
    result = StressTester().analyze(
        [7.0, 7.0, 7.0, 7.0]
    )

    assert result.has_constant_values is True
    assert result.standard_deviation == 0.0
    assert result.value_range == 0.0


def test_non_constant_values():
    result = StressTester().analyze(
        [1.0, 2.0, 1.0]
    )

    assert result.has_constant_values is False


def test_zero_values_are_valid():
    result = StressTester().analyze(
        [0.0, 0.0, 0.0]
    )

    assert result.mean == 0.0
    assert result.has_constant_values is True


def test_negative_values_are_valid():
    result = StressTester().analyze(
        [-3.0, -2.0, -1.0]
    )

    assert result.minimum == -3.0
    assert result.maximum == -1.0
    assert result.mean == -2.0


# ============================================================
# Extreme Magnitude Tests
# ============================================================


def test_value_at_threshold_is_extreme():
    tester = StressTester(
        extreme_threshold=100.0
    )

    result = tester.analyze(
        [1.0, 100.0]
    )

    assert result.has_extreme_magnitude is True


def test_value_above_threshold_is_extreme():
    tester = StressTester(
        extreme_threshold=100.0
    )

    result = tester.analyze(
        [1.0, 101.0]
    )

    assert result.has_extreme_magnitude is True


def test_negative_value_at_threshold_is_extreme():
    tester = StressTester(
        extreme_threshold=100.0
    )

    result = tester.analyze(
        [-100.0, 1.0]
    )

    assert result.has_extreme_magnitude is True


def test_value_below_threshold_is_not_extreme():
    tester = StressTester(
        extreme_threshold=100.0
    )

    result = tester.analyze(
        [-99.9, 99.9]
    )

    assert result.has_extreme_magnitude is False


def test_large_sequence():
    values = [float(index) for index in range(10_000)]

    result = StressTester().analyze(values)

    assert result.sample_count == 10_000
    assert result.minimum == 0.0
    assert result.maximum == 9_999.0
    assert result.has_constant_values is False


def test_very_large_finite_values():
    values = [
        -1e100,
        0.0,
        1e100,
    ]

    result = StressTester().analyze(values)

    assert result.is_finite is True
    assert result.minimum == -1e100
    assert result.maximum == 1e100
    assert result.has_extreme_magnitude is True


# ============================================================
# Input Validation Tests
# ============================================================


def test_empty_values():
    with pytest.raises(ValueError):
        StressTester().analyze([])


@pytest.mark.parametrize(
    "values",
    [
        "invalid",
        b"invalid",
        123,
        {"a": 1},
    ],
)
def test_invalid_values_type(values):
    with pytest.raises(TypeError):
        StressTester().analyze(values)


@pytest.mark.parametrize(
    "value",
    [
        "invalid",
        None,
        True,
        [],
    ],
)
def test_invalid_value_type(value):
    with pytest.raises(TypeError):
        StressTester().analyze([1.0, value])


@pytest.mark.parametrize(
    "value",
    [
        float("nan"),
        float("inf"),
        -float("inf"),
    ],
)
def test_non_finite_values(value):
    with pytest.raises(ValueError):
        StressTester().analyze([1.0, value])


# ============================================================
# Helper Tests
# ============================================================


def test_mean_helper():
    assert StressTester._mean(
        [1.0, 2.0, 3.0]
    ) == 2.0


def test_standard_deviation_helper():
    result = (
        StressTester._sample_standard_deviation(
            [1.0, 3.0, 5.0],
            3.0,
        )
    )

    assert result == 2.0


def test_standard_deviation_single_value():
    result = (
        StressTester._sample_standard_deviation(
            [5.0],
            5.0,
        )
    )

    assert result == 0.0


def test_validate_values_returns_none():
    result = StressTester._validate_values(
        [1.0, 2.0]
    )

    assert result is None