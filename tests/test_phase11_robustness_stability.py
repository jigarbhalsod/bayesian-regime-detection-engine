import math

import pytest

from src.validation.robustness import (
    RobustnessAnalyzer,
    RobustnessResult,
)


# ============================================================
# Configuration Tests
# ============================================================


def test_default_stability_threshold():
    analyzer = RobustnessAnalyzer()

    assert analyzer.stability_threshold == 0.10


def test_custom_stability_threshold():
    analyzer = RobustnessAnalyzer(
        stability_threshold=0.25
    )

    assert analyzer.stability_threshold == 0.25


@pytest.mark.parametrize(
    "threshold",
    [
        -0.1,
        "invalid",
        None,
        True,
        float("nan"),
        float("inf"),
        -float("inf"),
    ],
)
def test_invalid_stability_threshold(threshold):
    with pytest.raises((TypeError, ValueError)):
        RobustnessAnalyzer(
            stability_threshold=threshold
        )


def test_zero_stability_threshold_is_valid():
    analyzer = RobustnessAnalyzer(
        stability_threshold=0.0
    )

    assert analyzer.stability_threshold == 0.0


# ============================================================
# Result Tests
# ============================================================


def test_analyze_returns_robustness_result():
    analyzer = RobustnessAnalyzer()

    result = analyzer.analyze(
        [0.80, 0.82, 0.79]
    )

    assert isinstance(result, RobustnessResult)


def test_window_count():
    analyzer = RobustnessAnalyzer()

    result = analyzer.analyze(
        [0.80, 0.82, 0.79, 0.81]
    )

    assert result.window_count == 4


def test_mean_performance():
    analyzer = RobustnessAnalyzer()

    result = analyzer.analyze(
        [0.70, 0.80, 0.90]
    )

    assert math.isclose(
        result.mean_performance,
        0.80,
        abs_tol=1e-12,
    )


def test_standard_deviation():
    analyzer = RobustnessAnalyzer()

    result = analyzer.analyze(
        [1.0, 3.0, 5.0]
    )

    assert math.isclose(
        result.standard_deviation,
        2.0,
        abs_tol=1e-12,
    )


def test_minimum_and_maximum_performance():
    analyzer = RobustnessAnalyzer()

    result = analyzer.analyze(
        [0.75, 0.90, 0.60, 0.80]
    )

    assert result.minimum_performance == 0.60
    assert result.maximum_performance == 0.90


def test_performance_range():
    analyzer = RobustnessAnalyzer()

    result = analyzer.analyze(
        [0.75, 0.90, 0.60, 0.80]
    )

    assert math.isclose(
        result.performance_range,
        0.30,
        abs_tol=1e-12,
    )


def test_coefficient_of_variation():
    analyzer = RobustnessAnalyzer()

    result = analyzer.analyze(
        [1.0, 3.0, 5.0]
    )

    # Standard deviation = 2
    # Mean = 3
    expected = 2.0 / 3.0

    assert math.isclose(
        result.coefficient_of_variation,
        expected,
        abs_tol=1e-12,
    )


def test_result_preserves_threshold():
    analyzer = RobustnessAnalyzer(
        stability_threshold=0.20
    )

    result = analyzer.analyze(
        [0.80, 0.81, 0.79]
    )

    assert result.stability_threshold == 0.20


# ============================================================
# Stability Decision Tests
# ============================================================


def test_constant_performance_is_stable():
    analyzer = RobustnessAnalyzer(
        stability_threshold=0.10
    )

    result = analyzer.analyze(
        [0.80, 0.80, 0.80, 0.80]
    )

    assert result.standard_deviation == 0.0
    assert result.coefficient_of_variation == 0.0
    assert result.is_stable is True


def test_low_variation_is_stable():
    analyzer = RobustnessAnalyzer(
        stability_threshold=0.05
    )

    result = analyzer.analyze(
        [0.80, 0.81, 0.79, 0.80]
    )

    assert result.is_stable is True


def test_high_variation_is_not_stable():
    analyzer = RobustnessAnalyzer(
        stability_threshold=0.05
    )

    result = analyzer.analyze(
        [0.40, 0.90, 0.50, 0.95]
    )

    assert result.is_stable is False


def test_threshold_boundary_is_stable():
    analyzer = RobustnessAnalyzer(
        stability_threshold=0.0
    )

    result = analyzer.analyze(
        [1.0, 1.0, 1.0]
    )

    assert result.coefficient_of_variation == 0.0
    assert result.is_stable is True


def test_single_window_is_stable():
    analyzer = RobustnessAnalyzer(
        stability_threshold=0.0
    )

    result = analyzer.analyze([0.75])

    assert result.standard_deviation == 0.0
    assert result.coefficient_of_variation == 0.0
    assert result.is_stable is True


def test_zero_mean_with_variation_has_infinite_cv():
    analyzer = RobustnessAnalyzer()

    result = analyzer.analyze(
        [-1.0, 1.0]
    )

    assert result.mean_performance == 0.0
    assert result.standard_deviation > 0.0
    assert math.isinf(
        result.coefficient_of_variation
    )
    assert result.is_stable is False


def test_zero_mean_constant_performance_has_zero_cv():
    analyzer = RobustnessAnalyzer()

    result = analyzer.analyze(
        [0.0, 0.0, 0.0]
    )

    assert result.coefficient_of_variation == 0.0
    assert result.is_stable is True


# ============================================================
# Input Validation Tests
# ============================================================


def test_empty_performances():
    analyzer = RobustnessAnalyzer()

    with pytest.raises(ValueError):
        analyzer.analyze([])


@pytest.mark.parametrize(
    "performances",
    [
        "invalid",
        b"invalid",
        123,
        {"a": 1},
    ],
)
def test_invalid_performances_type(performances):
    analyzer = RobustnessAnalyzer()

    with pytest.raises(TypeError):
        analyzer.analyze(performances)


@pytest.mark.parametrize(
    "value",
    [
        "invalid",
        None,
        True,
        [],
    ],
)
def test_invalid_performance_value_type(value):
    analyzer = RobustnessAnalyzer()

    with pytest.raises(TypeError):
        analyzer.analyze([0.8, value])


@pytest.mark.parametrize(
    "value",
    [
        float("nan"),
        float("inf"),
        -float("inf"),
    ],
)
def test_non_finite_performance_values(value):
    analyzer = RobustnessAnalyzer()

    with pytest.raises(ValueError):
        analyzer.analyze([0.8, value])


# ============================================================
# Helper Tests
# ============================================================


def test_mean_helper():
    assert RobustnessAnalyzer._mean(
        [1.0, 2.0, 3.0]
    ) == 2.0


def test_sample_standard_deviation_helper():
    result = (
        RobustnessAnalyzer
        ._sample_standard_deviation(
            [1.0, 3.0, 5.0],
            3.0,
        )
    )

    assert result == 2.0


def test_sample_standard_deviation_single_value():
    result = (
        RobustnessAnalyzer
        ._sample_standard_deviation(
            [1.0],
            1.0,
        )
    )

    assert result == 0.0