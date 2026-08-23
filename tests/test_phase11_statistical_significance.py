import math

import pytest

from src.validation.statistical_tests import (
    PairedSignificanceTester,
    StatisticalTestResult,
)


# ============================================================
# Configuration Tests
# ============================================================


def test_default_alpha():
    tester = PairedSignificanceTester()

    assert tester.alpha == 0.05


def test_custom_alpha():
    tester = PairedSignificanceTester(alpha=0.01)

    assert tester.alpha == 0.01


@pytest.mark.parametrize(
    "alpha",
    [
        0,
        1,
        -0.1,
        1.1,
        "invalid",
        None,
        True,
        float("nan"),
        float("inf"),
    ],
)
def test_invalid_alpha(alpha):
    with pytest.raises((TypeError, ValueError)):
        PairedSignificanceTester(alpha=alpha)


# ============================================================
# Result Tests
# ============================================================


def test_returns_statistical_test_result():
    tester = PairedSignificanceTester()

    result = tester.test(
        model_a=[0.8, 0.9, 0.7],
        model_b=[0.7, 0.8, 0.6],
    )

    assert isinstance(result, StatisticalTestResult)


def test_result_sample_size():
    tester = PairedSignificanceTester()

    result = tester.test(
        model_a=[1.0, 2.0, 3.0],
        model_b=[0.0, 1.0, 2.0],
    )

    assert result.sample_size == 3


def test_mean_difference():
    tester = PairedSignificanceTester()

    result = tester.test(
        model_a=[0.9, 0.8, 0.7],
        model_b=[0.7, 0.6, 0.5],
    )

    assert math.isclose(
        result.mean_difference,
        0.2,
        abs_tol=1e-12,
    )


def test_standard_deviation():
    tester = PairedSignificanceTester()

    result = tester.test(
        model_a=[2.0, 4.0, 6.0],
        model_b=[1.0, 1.0, 1.0],
    )

    # Differences = [1, 3, 5]
    # Sample standard deviation = 2
    assert math.isclose(
        result.standard_deviation,
        2.0,
        abs_tol=1e-12,
    )


def test_standard_error():
    tester = PairedSignificanceTester()

    result = tester.test(
        model_a=[2.0, 4.0, 6.0],
        model_b=[1.0, 1.0, 1.0],
    )

    expected = 2.0 / math.sqrt(3)

    assert math.isclose(
        result.standard_error,
        expected,
        abs_tol=1e-12,
    )


def test_test_statistic():
    tester = PairedSignificanceTester()

    result = tester.test(
        model_a=[2.0, 4.0, 6.0],
        model_b=[1.0, 1.0, 1.0],
    )

    # Mean difference = 3
    # SE = 2 / sqrt(3)
    expected = 3.0 / (2.0 / math.sqrt(3))

    assert math.isclose(
        result.test_statistic,
        expected,
        abs_tol=1e-12,
    )


def test_p_value_range():
    tester = PairedSignificanceTester()

    result = tester.test(
        model_a=[0.9, 0.8, 0.7, 0.85],
        model_b=[0.7, 0.6, 0.5, 0.65],
    )

    assert 0.0 <= result.p_value <= 1.0


# ============================================================
# Significance Tests
# ============================================================


def test_identical_models_are_not_significant():
    tester = PairedSignificanceTester()

    result = tester.test(
        model_a=[0.8, 0.7, 0.9, 0.6],
        model_b=[0.8, 0.7, 0.9, 0.6],
    )

    assert result.mean_difference == 0.0
    assert result.standard_error == 0.0
    assert result.test_statistic == 0.0
    assert result.p_value == 1.0
    assert result.is_significant is False


def test_consistent_difference_is_significant():
    tester = PairedSignificanceTester()

    result = tester.test(
        model_a=[0.90, 0.91, 0.92, 0.93],
        model_b=[0.70, 0.71, 0.72, 0.73],
    )

    assert result.mean_difference > 0.0
    assert math.isinf(result.test_statistic)
    assert result.p_value == 0.0
    assert result.is_significant is True


def test_alpha_controls_significance():
    model_a = [0.90, 0.80, 0.85, 0.95, 0.88]
    model_b = [0.80, 0.78, 0.75, 0.85, 0.80]

    result_05 = PairedSignificanceTester(
        alpha=0.05
    ).test(model_a, model_b)

    result_01 = PairedSignificanceTester(
        alpha=0.01
    ).test(model_a, model_b)

    assert result_05.alpha == 0.05
    assert result_01.alpha == 0.01

    if result_05.is_significant:
        assert (
            result_01.p_value == result_05.p_value
        )


def test_negative_difference_is_supported():
    tester = PairedSignificanceTester()

    result = tester.test(
        model_a=[0.5, 0.6, 0.55],
        model_b=[0.8, 0.9, 0.85],
    )

    assert result.mean_difference < 0.0


# ============================================================
# Input Validation Tests
# ============================================================


@pytest.mark.parametrize(
    "model_a, model_b",
    [
        ([], []),
        ([], [1.0]),
        ([1.0], []),
    ],
)
def test_empty_inputs(model_a, model_b):
    tester = PairedSignificanceTester()

    with pytest.raises(ValueError):
        tester.test(model_a, model_b)


def test_mismatched_lengths():
    tester = PairedSignificanceTester()

    with pytest.raises(ValueError):
        tester.test(
            model_a=[1.0, 2.0],
            model_b=[1.0],
        )


@pytest.mark.parametrize(
    "model_a",
    [
        "invalid",
        b"invalid",
        123,
        {"a": 1},
    ],
)
def test_invalid_model_a_type(model_a):
    tester = PairedSignificanceTester()

    with pytest.raises(TypeError):
        tester.test(
            model_a=model_a,
            model_b=[1.0, 2.0],
        )


@pytest.mark.parametrize(
    "model_b",
    [
        "invalid",
        b"invalid",
        123,
        {"a": 1},
    ],
)
def test_invalid_model_b_type(model_b):
    tester = PairedSignificanceTester()

    with pytest.raises(TypeError):
        tester.test(
            model_a=[1.0, 2.0],
            model_b=model_b,
        )


@pytest.mark.parametrize(
    "model_a, model_b",
    [
        ([1.0], [2.0]),
        ([1.0], [1.0]),
    ],
)
def test_requires_at_least_two_observations(
    model_a,
    model_b,
):
    tester = PairedSignificanceTester()

    with pytest.raises(ValueError):
        tester.test(model_a, model_b)


@pytest.mark.parametrize(
    "value",
    [
        "invalid",
        None,
        True,
        [],
    ],
)
def test_invalid_model_value_type(value):
    tester = PairedSignificanceTester()

    with pytest.raises(TypeError):
        tester.test(
            model_a=[1.0, value],
            model_b=[1.0, 2.0],
        )


@pytest.mark.parametrize(
    "value",
    [
        float("nan"),
        float("inf"),
        -float("inf"),
    ],
)
def test_non_finite_model_values(value):
    tester = PairedSignificanceTester()

    with pytest.raises(ValueError):
        tester.test(
            model_a=[1.0, value],
            model_b=[1.0, 2.0],
        )


# ============================================================
# Helper Tests
# ============================================================


def test_mean_helper():
    assert PairedSignificanceTester._mean(
        [1.0, 2.0, 3.0]
    ) == 2.0


def test_sample_standard_deviation_helper():
    result = (
        PairedSignificanceTester
        ._sample_standard_deviation(
            [1.0, 3.0, 5.0],
            3.0,
        )
    )

    assert result == 2.0


def test_sample_standard_deviation_single_value():
    result = (
        PairedSignificanceTester
        ._sample_standard_deviation(
            [1.0],
            1.0,
        )
    )

    assert result == 0.0