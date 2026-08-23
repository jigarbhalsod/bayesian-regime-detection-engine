import pytest

from src.monitoring.config import MonitoringConfig
from src.monitoring.models import MonitoringStatus
from src.monitoring.performance_degradation import (
    PerformanceDegradationMonitor,
)


def create_monitor(
    thresholds=None,
):
    return PerformanceDegradationMonitor(
        MonitoringConfig(
            name="performance_degradation",
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


def test_degradation_monitor_returns_ok_for_identical_performance():
    monitor = create_monitor()

    result = monitor.execute(
        {
            "reference": {
                "accuracy": 0.90,
                "mae": 0.20,
            },
            "current": {
                "accuracy": 0.90,
                "mae": 0.20,
            },
        }
    )

    assert (
        get_metric(
            result,
            "accuracy_degradation",
        )
        == 0.0
    )

    assert (
        get_metric(
            result,
            "mae_degradation",
        )
        == 0.0
    )

    assert result.status == MonitoringStatus.OK


def test_degradation_monitor_detects_accuracy_degradation():
    monitor = create_monitor()

    result = monitor.execute(
        {
            "reference": {
                "accuracy": 0.90,
            },
            "current": {
                "accuracy": 0.80,
            },
        }
    )

    assert get_metric(
        result,
        "accuracy_degradation",
    ) == pytest.approx(0.10)


def test_degradation_monitor_detects_accuracy_improvement():
    monitor = create_monitor()

    result = monitor.execute(
        {
            "reference": {
                "accuracy": 0.80,
            },
            "current": {
                "accuracy": 0.90,
            },
        }
    )

    assert get_metric(
        result,
        "accuracy_degradation",
    ) == pytest.approx(-0.10)


@pytest.mark.parametrize(
    "metric_name",
    [
        "mae",
        "mse",
        "rmse",
        "loss",
    ],
)
def test_degradation_monitor_detects_lower_is_better_metric_degradation(
    metric_name,
):
    monitor = create_monitor()

    result = monitor.execute(
        {
            "reference": {
                metric_name: 0.20,
            },
            "current": {
                metric_name: 0.50,
            },
        }
    )

    assert get_metric(
        result,
        f"{metric_name}_degradation",
    ) == pytest.approx(0.30)


def test_degradation_monitor_handles_multiple_metrics():
    monitor = create_monitor()

    result = monitor.execute(
        {
            "reference": {
                "accuracy": 0.90,
                "precision": 0.85,
                "mae": 0.20,
            },
            "current": {
                "accuracy": 0.80,
                "precision": 0.80,
                "mae": 0.30,
            },
        }
    )

    assert len(result.metrics) == 3
    assert result.metadata["metric_count"] == 3


def test_degradation_monitor_warns_when_threshold_exceeded():
    monitor = create_monitor(
        thresholds={
            "accuracy": 0.05,
        }
    )

    result = monitor.execute(
        {
            "reference": {
                "accuracy": 0.90,
            },
            "current": {
                "accuracy": 0.80,
            },
        }
    )

    assert result.status == MonitoringStatus.WARNING


def test_degradation_monitor_uses_ok_at_exact_threshold():
    monitor = create_monitor(
        thresholds={
            "accuracy": 0.10,
        }
    )

    result = monitor.execute(
        {
            "reference": {
                "accuracy": 0.90,
            },
            "current": {
                "accuracy": 0.80,
            },
        }
    )

    assert result.status == MonitoringStatus.OK


def test_degradation_monitor_does_not_warn_without_threshold():
    monitor = create_monitor()

    result = monitor.execute(
        {
            "reference": {
                "accuracy": 0.90,
            },
            "current": {
                "accuracy": 0.10,
            },
        }
    )

    assert result.status == MonitoringStatus.OK


def test_degradation_monitor_rejects_non_dictionary_data():
    monitor = create_monitor()

    with pytest.raises(TypeError):
        monitor.execute(
            ["invalid"]
        )


@pytest.mark.parametrize(
    "invalid_data",
    [
        {},
        {"reference": {}},
        {"current": {}},
        {
            "reference": {},
            "current": {},
            "extra": {},
        },
    ],
)
def test_degradation_monitor_rejects_missing_or_extra_keys(
    invalid_data,
):
    monitor = create_monitor()

    with pytest.raises(
        (TypeError, ValueError),
    ):
        monitor.execute(
            invalid_data
        )


def test_degradation_monitor_rejects_non_dictionary_reference():
    monitor = create_monitor()

    with pytest.raises(TypeError):
        monitor.execute(
            {
                "reference": [0.9],
                "current": {
                    "accuracy": 0.8,
                },
            }
        )


def test_degradation_monitor_rejects_empty_metrics():
    monitor = create_monitor()

    with pytest.raises(ValueError):
        monitor.execute(
            {
                "reference": {},
                "current": {},
            }
        )


def test_degradation_monitor_rejects_different_metric_names():
    monitor = create_monitor()

    with pytest.raises(ValueError):
        monitor.execute(
            {
                "reference": {
                    "accuracy": 0.9,
                },
                "current": {
                    "precision": 0.8,
                },
            }
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        "0.9",
        None,
        True,
        False,
        [],
    ],
)
def test_degradation_monitor_rejects_non_numeric_metric_values(
    invalid_value,
):
    monitor = create_monitor()

    with pytest.raises(TypeError):
        monitor.execute(
            {
                "reference": {
                    "accuracy": invalid_value,
                },
                "current": {
                    "accuracy": 0.8,
                },
            }
        )


@pytest.mark.parametrize(
    "invalid_name",
    [
        "",
        "   ",
        123,
    ],
)
def test_degradation_monitor_rejects_invalid_metric_names(
    invalid_name,
):
    monitor = create_monitor()

    with pytest.raises(
        (TypeError, ValueError),
    ):
        monitor.execute(
            {
                "reference": {
                    invalid_name: 0.9,
                },
                "current": {
                    invalid_name: 0.8,
                },
            }
        )


def test_degradation_monitor_includes_reference_and_current_metadata():
    monitor = create_monitor()

    reference = {
        "accuracy": 0.90,
    }

    current = {
        "accuracy": 0.80,
    }

    result = monitor.execute(
        {
            "reference": reference,
            "current": current,
        }
    )

    assert (
        result.metadata["reference"]
        == reference
    )

    assert (
        result.metadata["current"]
        == current
    )