import numpy as np
import pytest

from src.monitoring.config import MonitoringConfig
from src.monitoring.drift import PredictionDriftMonitor
from src.monitoring.models import MonitoringStatus


def create_monitor(
    reference_predictions,
    thresholds=None,
):
    return PredictionDriftMonitor(
        MonitoringConfig(
            name="prediction_drift",
            thresholds=thresholds or {},
            metadata={
                "reference_predictions": reference_predictions,
            },
        )
    )


def get_metric(result, name):
    return next(
        metric
        for metric in result.metrics
        if metric.name == name
    )


def test_prediction_drift_returns_ok_for_identical_predictions():
    reference = [0.2, 0.4, 0.6]

    monitor = create_monitor(reference)

    result = monitor.execute(reference)

    assert result.status == MonitoringStatus.OK
    assert (
        get_metric(
            result,
            "prediction_drift_score",
        ).value
        == 0.0
    )


def test_prediction_drift_detects_mean_shift():
    reference = [0.2, 0.2, 0.2]
    current = [0.3, 0.3, 0.3]

    monitor = create_monitor(reference)

    result = monitor.execute(current)

    assert (
        get_metric(
            result,
            "prediction_mean_difference",
        ).value
        == pytest.approx(0.1)
    )

    assert (
        get_metric(
            result,
            "prediction_drift_score",
        ).value
        == pytest.approx(0.5)
    )


def test_prediction_drift_detects_standard_deviation_change():
    reference = [0.4, 0.5, 0.6]
    current = [0.1, 0.5, 0.9]

    monitor = create_monitor(reference)

    result = monitor.execute(current)

    assert (
        get_metric(
            result,
            "prediction_std_difference",
        ).value
        > 0.0
    )


def test_prediction_drift_uses_reference_scale():
    reference = [10.0, 10.0]
    current = [15.0, 15.0]

    monitor = create_monitor(reference)

    result = monitor.execute(current)

    assert (
        get_metric(
            result,
            "prediction_drift_score",
        ).value
        == 0.5
    )


def test_prediction_drift_includes_metadata():
    reference = [0.1, 0.2, 0.3]
    current = [0.2, 0.3]

    monitor = create_monitor(reference)

    result = monitor.execute(current)

    assert result.metadata["reference_count"] == 3
    assert result.metadata["current_count"] == 2
    assert result.metadata["reference_mean"] == pytest.approx(
        0.2
    )
    assert result.metadata["current_mean"] == pytest.approx(
        0.25
    )


def test_prediction_drift_warns_when_score_threshold_exceeded():
    reference = [10.0, 10.0]
    current = [20.0, 20.0]

    monitor = create_monitor(
        reference,
        thresholds={
            "prediction_drift_score": 0.5,
        },
    )

    result = monitor.execute(current)

    assert result.status == MonitoringStatus.WARNING


def test_prediction_drift_uses_ok_at_exact_threshold():
    reference = [10.0, 10.0]
    current = [15.0, 15.0]

    monitor = create_monitor(
        reference,
        thresholds={
            "prediction_drift_score": 0.5,
        },
    )

    result = monitor.execute(current)

    assert result.status == MonitoringStatus.OK


@pytest.mark.parametrize(
    "invalid_reference",
    [
        [],
        [0.1, np.nan],
        [0.1, np.inf],
        "invalid",
        {"prediction": 0.1},
        [[0.1, 0.2]],
    ],
)
def test_prediction_drift_rejects_invalid_reference_predictions(
    invalid_reference,
):
    monitor = create_monitor(invalid_reference)

    with pytest.raises(
        (TypeError, ValueError),
    ):
        monitor.execute([0.1, 0.2])


@pytest.mark.parametrize(
    "invalid_data",
    [
        [],
        [0.1, np.nan],
        [0.1, -np.inf],
        "invalid",
        {"prediction": 0.1},
        [[0.1, 0.2]],
    ],
)
def test_prediction_drift_rejects_invalid_current_predictions(
    invalid_data,
):
    monitor = create_monitor([0.1, 0.2])

    with pytest.raises(
        (TypeError, ValueError),
    ):
        monitor.execute(invalid_data)