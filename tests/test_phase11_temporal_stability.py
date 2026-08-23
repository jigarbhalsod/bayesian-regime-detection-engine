import math

import pytest

from src.validation.temporal_stability import (
    TemporalStabilityAnalyzer,
    TemporalStabilityResult,
)


# ============================================================
# Configuration Tests
# ============================================================


def test_default_stability_tolerance():
    analyzer = TemporalStabilityAnalyzer()

    assert analyzer.stability_tolerance == 0.01


def test_custom_stability_tolerance():
    analyzer = TemporalStabilityAnalyzer(
        stability_tolerance=0.05
    )

    assert analyzer.stability_tolerance == 0.05


@pytest.mark.parametrize(
    "tolerance",
    [
        -0.01,
        "invalid",
        None,
        True,
        float("nan"),
        float("inf"),
        -float("inf"),
    ],
)
def test_invalid_stability_tolerance(tolerance):
    with pytest.raises((TypeError, ValueError)):
        TemporalStabilityAnalyzer(
            stability_tolerance=tolerance
        )


def test_zero_stability_tolerance_is_valid():
    analyzer = TemporalStabilityAnalyzer(
        stability_tolerance=0.0
    )

    assert analyzer.stability_tolerance == 0.0


# ============================================================
# Result Tests
# ============================================================


def test_analyze_returns_result():
    result = TemporalStabilityAnalyzer().analyze(
        [0.80, 0.81, 0.82]
    )

    assert isinstance(result, TemporalStabilityResult)


def test_window_count():
    result = TemporalStabilityAnalyzer().analyze(
        [0.80, 0.81, 0.82, 0.83]
    )

    assert result.window_count == 4


def test_mean_performance():
    result = TemporalStabilityAnalyzer().analyze(
        [0.70, 0.80, 0.90]
    )

    assert math.isclose(
        result.mean_performance,
        0.80,
        abs_tol=1e-12,
    )


def test_standard_deviation():
    result = TemporalStabilityAnalyzer().analyze(
        [1.0, 3.0, 5.0]
    )

    assert math.isclose(
        result.standard_deviation,
        2.0,
        abs_tol=1e-12,
    )


def test_first_and_last_performance():
    result = TemporalStabilityAnalyzer().analyze(
        [0.70, 0.80, 0.90]
    )

    assert result.first_performance == 0.70
    assert result.last_performance == 0.90


def test_absolute_change():
    result = TemporalStabilityAnalyzer().analyze(
        [0.70, 0.80, 0.90]
    )

    assert math.isclose(
        result.absolute_change,
        0.20,
        abs_tol=1e-12,
    )


def test_relative_change():
    result = TemporalStabilityAnalyzer().analyze(
        [0.50, 0.60, 0.75]
    )

    assert math.isclose(
        result.relative_change,
        0.50,
        abs_tol=1e-12,
    )


# ============================================================
# Trend Tests
# ============================================================


def test_improving_trend():
    analyzer = TemporalStabilityAnalyzer(
        stability_tolerance=0.01
    )

    result = analyzer.analyze(
        [0.50, 0.60, 0.70, 0.80]
    )

    assert result.trend_slope > 0.01
    assert result.is_improving is True
    assert result.is_deteriorating is False
    assert result.is_stable is False


def test_deteriorating_trend():
    analyzer = TemporalStabilityAnalyzer(
        stability_tolerance=0.01
    )

    result = analyzer.analyze(
        [0.80, 0.70, 0.60, 0.50]
    )

    assert result.trend_slope < -0.01
    assert result.is_improving is False
    assert result.is_deteriorating is True
    assert result.is_stable is False


def test_constant_performance_is_stable():
    analyzer = TemporalStabilityAnalyzer(
        stability_tolerance=0.01
    )

    result = analyzer.analyze(
        [0.80, 0.80, 0.80, 0.80]
    )

    assert result.trend_slope == 0.0
    assert result.is_improving is False
    assert result.is_deteriorating is False
    assert result.is_stable is True


def test_small_trend_within_tolerance_is_stable():
    analyzer = TemporalStabilityAnalyzer(
        stability_tolerance=0.05
    )

    result = analyzer.analyze(
        [0.80, 0.81, 0.82, 0.83]
    )

    assert result.trend_slope < 0.05
    assert result.is_stable is True


def test_threshold_boundary_is_stable():
    analyzer = TemporalStabilityAnalyzer(
        stability_tolerance=0.10
    )

    result = analyzer.analyze(
        [1.0, 1.1, 1.2]
    )

    assert math.isclose(
        result.trend_slope,
        0.10,
        abs_tol=1e-12,
    )

    assert result.is_stable is True


def test_single_window_is_stable():
    result = TemporalStabilityAnalyzer(
        stability_tolerance=0.0
    ).analyze([0.75])

    assert result.trend_slope == 0.0
    assert result.is_stable is True


# ============================================================
# Relative Change Edge Cases
# ============================================================


def test_zero_first_value_with_positive_change():
    result = TemporalStabilityAnalyzer().analyze(
        [0.0, 1.0]
    )

    assert math.isinf(result.relative_change)
    assert result.relative_change > 0.0


def test_zero_first_value_with_negative_change():
    result = TemporalStabilityAnalyzer().analyze(
        [0.0, -1.0]
    )

    assert math.isinf(result.relative_change)
    assert result.relative_change < 0.0


def test_zero_first_value_with_no_change():
    result = TemporalStabilityAnalyzer().analyze(
        [0.0, 0.0]
    )

    assert result.relative_change == 0.0


def test_negative_first_value_uses_absolute_denominator():
    result = TemporalStabilityAnalyzer().analyze(
        [-2.0, -1.0]
    )

    assert result.absolute_change == 1.0
    assert result.relative_change == 0.5


# ============================================================
# Input Validation Tests
# ============================================================


def test_empty_performances():
    with pytest.raises(ValueError):
        TemporalStabilityAnalyzer().analyze([])


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
    with pytest.raises(TypeError):
        TemporalStabilityAnalyzer().analyze(
            performances
        )


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
    with pytest.raises(TypeError):
        TemporalStabilityAnalyzer().analyze(
            [0.8, value]
        )


@pytest.mark.parametrize(
    "value",
    [
        float("nan"),
        float("inf"),
        -float("inf"),
    ],
)
def test_non_finite_performance_values(value):
    with pytest.raises(ValueError):
        TemporalStabilityAnalyzer().analyze(
            [0.8, value]
        )


# ============================================================
# Helper Tests
# ============================================================


def test_mean_helper():
    assert TemporalStabilityAnalyzer._mean(
        [1.0, 2.0, 3.0]
    ) == 2.0


def test_sample_standard_deviation_helper():
    result = (
        TemporalStabilityAnalyzer
        ._sample_standard_deviation(
            [1.0, 3.0, 5.0],
            3.0,
        )
    )

    assert result == 2.0


def test_sample_standard_deviation_single_value():
    result = (
        TemporalStabilityAnalyzer
        ._sample_standard_deviation(
            [1.0],
            1.0,
        )
    )

    assert result == 0.0


def test_relative_change_helper():
    assert (
        TemporalStabilityAnalyzer._relative_change(
            2.0,
            1.0,
        )
        == 0.5
    )


def test_trend_slope_helper():
    result = (
        TemporalStabilityAnalyzer
        ._calculate_trend_slope(
            [1.0, 2.0, 3.0]
        )
    )

    assert math.isclose(
        result,
        1.0,
        abs_tol=1e-12,
    )


def test_trend_slope_single_value():
    assert (
        TemporalStabilityAnalyzer
        ._calculate_trend_slope(
            [5.0]
        )
        == 0.0
    )