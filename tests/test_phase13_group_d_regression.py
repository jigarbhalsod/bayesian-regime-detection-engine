import math

import pytest

from src.monitoring.config import MonitoringConfig
from src.monitoring.models import MonitoringStatus
from src.monitoring.regression_performance import (
    RegressionPerformanceMonitor,
)


def create_monitor(
    thresholds=None,
):
    return RegressionPerformanceMonitor(
        MonitoringConfig(
            name="regression_performance",
            thresholds=thresholds or {},
        )
    )


def get_metric(
    result,
    name,
):
    for metric in result.metrics:
        if metric.name == name:
            return metric.value

    raise AssertionError(
        f"Metric not found: {name}"
    )


def test_regression_monitor_perfect_predictions():
    monitor = create_monitor()

    result = monitor.execute(
        {
            "y_true": [1.0, 2.0, 3.0],
            "y_pred": [1.0, 2.0, 3.0],
        }
    )

    assert get_metric(result, "mae") == 0.0
    assert get_metric(result, "mse") == 0.0
    assert get_metric(result, "rmse") == 0.0
    assert get_metric(result, "r2_score") == 1.0
    assert result.status == MonitoringStatus.OK


def test_regression_monitor_calculates_mae():
    monitor = create_monitor()

    result = monitor.execute(
        {
            "y_true": [1.0, 3.0, 5.0],
            "y_pred": [2.0, 1.0, 8.0],
        }
    )

    assert get_metric(
        result,
        "mae",
    ) == pytest.approx(2.0)


def test_regression_monitor_calculates_mse():
    monitor = create_monitor()

    result = monitor.execute(
        {
            "y_true": [1.0, 3.0, 5.0],
            "y_pred": [2.0, 1.0, 8.0],
        }
    )

    assert get_metric(
        result,
        "mse",
    ) == pytest.approx(14 / 3)


def test_regression_monitor_calculates_rmse():
    monitor = create_monitor()

    result = monitor.execute(
        {
            "y_true": [1.0, 3.0, 5.0],
            "y_pred": [2.0, 1.0, 8.0],
        }
    )

    assert get_metric(
        result,
        "rmse",
    ) == pytest.approx(
        math.sqrt(14 / 3)
    )


def test_regression_monitor_calculates_r2_score():
    monitor = create_monitor()

    result = monitor.execute(
        {
            "y_true": [1.0, 2.0, 3.0],
            "y_pred": [1.0, 2.0, 2.0],
        }
    )

    assert get_metric(
        result,
        "r2_score",
    ) == pytest.approx(0.5)


def test_regression_monitor_handles_negative_r2():
    monitor = create_monitor()

    result = monitor.execute(
        {
            "y_true": [1.0, 2.0, 3.0],
            "y_pred": [10.0, 10.0, 10.0],
        }
    )

    assert get_metric(
        result,
        "r2_score",
    ) < 0.0


def test_regression_monitor_handles_constant_target_perfect_predictions():
    monitor = create_monitor()

    result = monitor.execute(
        {
            "y_true": [5.0, 5.0, 5.0],
            "y_pred": [5.0, 5.0, 5.0],
        }
    )

    assert get_metric(
        result,
        "r2_score",
    ) == 1.0


def test_regression_monitor_handles_constant_target_imperfect_predictions():
    monitor = create_monitor()

    result = monitor.execute(
        {
            "y_true": [5.0, 5.0, 5.0],
            "y_pred": [4.0, 5.0, 6.0],
        }
    )

    assert get_metric(
        result,
        "r2_score",
    ) == 0.0


@pytest.mark.parametrize(
    "threshold_name",
    [
        "mae",
        "mse",
        "rmse",
    ],
)
def test_regression_monitor_warns_when_error_threshold_exceeded(
    threshold_name,
):
    monitor = create_monitor(
        thresholds={
            threshold_name: 0.5,
        }
    )

    result = monitor.execute(
        {
            "y_true": [1.0, 2.0],
            "y_pred": [2.0, 4.0],
        }
    )

    assert result.status == MonitoringStatus.WARNING


def test_regression_monitor_warns_when_r2_falls_below_threshold():
    monitor = create_monitor(
        thresholds={
            "r2_score": 0.9,
        }
    )

    result = monitor.execute(
        {
            "y_true": [1.0, 2.0, 3.0],
            "y_pred": [1.0, 2.0, 2.0],
        }
    )

    assert result.status == MonitoringStatus.WARNING


def test_regression_monitor_uses_ok_at_exact_threshold():
    monitor = create_monitor(
        thresholds={
            "mae": 1.0,
        }
    )

    result = monitor.execute(
        {
            "y_true": [1.0, 2.0],
            "y_pred": [2.0, 3.0],
        }
    )

    assert result.status == MonitoringStatus.OK


@pytest.mark.parametrize(
    "invalid_values",
    [
        ["a", "b"],
        [float("nan"), 1.0],
        [float("inf"), 1.0],
        [float("-inf"), 1.0],
    ],
)
def test_regression_monitor_rejects_invalid_numeric_values(
    invalid_values,
):
    monitor = create_monitor()

    with pytest.raises(
        (TypeError, ValueError),
    ):
        monitor.execute(
            {
                "y_true": invalid_values,
                "y_pred": [1.0, 2.0],
            }
        )