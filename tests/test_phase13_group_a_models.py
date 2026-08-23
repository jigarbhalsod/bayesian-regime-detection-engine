from datetime import datetime, timezone

import pytest

from src.monitoring.models import (
    MetricResult,
    MonitoringResult,
    MonitoringSnapshot,
    MonitoringStatus,
)


def test_monitoring_status_values():
    assert MonitoringStatus.OK.value == "ok"
    assert MonitoringStatus.WARNING.value == "warning"
    assert MonitoringStatus.CRITICAL.value == "critical"
    assert MonitoringStatus.ERROR.value == "error"
    assert MonitoringStatus.UNKNOWN.value == "unknown"


def test_metric_result_creates_successfully():
    metric = MetricResult(
        name="drift_score",
        value=0.25,
    )

    assert metric.name == "drift_score"
    assert metric.value == 0.25
    assert metric.threshold is None
    assert metric.status == MonitoringStatus.OK
    assert metric.metadata == {}


def test_metric_result_normalizes_name():
    metric = MetricResult(
        name="  drift_score  ",
        value=1,
    )

    assert metric.name == "drift_score"
    assert metric.value == 1.0


@pytest.mark.parametrize("name", ["", "   "])
def test_metric_result_rejects_empty_name(name):
    with pytest.raises(ValueError):
        MetricResult(name=name, value=1.0)


@pytest.mark.parametrize(
    "value",
    ["1.0", None, True, False, []],
)
def test_metric_result_rejects_invalid_value(value):
    with pytest.raises(TypeError):
        MetricResult(name="metric", value=value)


@pytest.mark.parametrize(
    "threshold",
    ["1.0", True, False, []],
)
def test_metric_result_rejects_invalid_threshold(threshold):
    with pytest.raises(TypeError):
        MetricResult(
            name="metric",
            value=1.0,
            threshold=threshold,
        )


def test_metric_result_rejects_invalid_status():
    with pytest.raises(TypeError):
        MetricResult(
            name="metric",
            value=1.0,
            status="ok",
        )


def test_metric_result_normalizes_metadata():
    metric = MetricResult(
        name="metric",
        value=1.0,
        metadata={" source ": "model"},
    )

    assert metric.metadata == {"source": "model"}


def test_metric_result_rejects_empty_metadata_key():
    with pytest.raises(ValueError):
        MetricResult(
            name="metric",
            value=1.0,
            metadata={" ": "value"},
        )


def test_metric_result_is_frozen():
    metric = MetricResult(
        name="metric",
        value=1.0,
    )

    with pytest.raises(AttributeError):
        metric.name = "changed"


def test_monitoring_result_creates_successfully():
    metric = MetricResult(
        name="score",
        value=0.2,
    )

    result = MonitoringResult(
        monitor_name="drift",
        status=MonitoringStatus.OK,
        metrics=(metric,),
    )

    assert result.monitor_name == "drift"
    assert result.status == MonitoringStatus.OK
    assert result.metrics == (metric,)
    assert result.timestamp.tzinfo is not None


def test_monitoring_result_normalizes_name_and_message():
    result = MonitoringResult(
        monitor_name="  monitor  ",
        status=MonitoringStatus.WARNING,
        message="  threshold exceeded  ",
    )

    assert result.monitor_name == "monitor"
    assert result.message == "threshold exceeded"


def test_monitoring_result_empty_message_becomes_none():
    result = MonitoringResult(
        monitor_name="monitor",
        status=MonitoringStatus.OK,
        message="   ",
    )

    assert result.message is None


@pytest.mark.parametrize("name", ["", "   "])
def test_monitoring_result_rejects_empty_name(name):
    with pytest.raises(ValueError):
        MonitoringResult(
            monitor_name=name,
            status=MonitoringStatus.OK,
        )


def test_monitoring_result_rejects_invalid_status():
    with pytest.raises(TypeError):
        MonitoringResult(
            monitor_name="monitor",
            status="ok",
        )


def test_monitoring_result_rejects_invalid_metric():
    with pytest.raises(TypeError):
        MonitoringResult(
            monitor_name="monitor",
            status=MonitoringStatus.OK,
            metrics=("invalid",),
        )


def test_monitoring_result_converts_metrics_to_tuple():
    metric = MetricResult(name="score", value=0.5)

    result = MonitoringResult(
        monitor_name="monitor",
        status=MonitoringStatus.OK,
        metrics=[metric],
    )

    assert isinstance(result.metrics, tuple)
    assert result.metrics == (metric,)


def test_monitoring_result_rejects_naive_timestamp():
    with pytest.raises(ValueError):
        MonitoringResult(
            monitor_name="monitor",
            status=MonitoringStatus.OK,
            timestamp=datetime(2026, 1, 1),
        )


def test_monitoring_result_rejects_non_datetime_timestamp():
    with pytest.raises(TypeError):
        MonitoringResult(
            monitor_name="monitor",
            status=MonitoringStatus.OK,
            timestamp="2026-01-01",
        )


def test_snapshot_creates_successfully():
    result = MonitoringResult(
        monitor_name="monitor",
        status=MonitoringStatus.OK,
    )

    snapshot = MonitoringSnapshot(
        name="snapshot",
        results=(result,),
    )

    assert snapshot.name == "snapshot"
    assert snapshot.results == (result,)
    assert snapshot.status == MonitoringStatus.OK


@pytest.mark.parametrize("name", ["", "   "])
def test_snapshot_rejects_empty_name(name):
    result = MonitoringResult(
        monitor_name="monitor",
        status=MonitoringStatus.OK,
    )

    with pytest.raises(ValueError):
        MonitoringSnapshot(
            name=name,
            results=(result,),
        )


def test_snapshot_rejects_empty_results():
    with pytest.raises(ValueError):
        MonitoringSnapshot(
            name="snapshot",
            results=(),
        )


def test_snapshot_rejects_invalid_result():
    with pytest.raises(TypeError):
        MonitoringSnapshot(
            name="snapshot",
            results=("invalid",),
        )


def test_snapshot_converts_results_to_tuple():
    result = MonitoringResult(
        monitor_name="monitor",
        status=MonitoringStatus.OK,
    )

    snapshot = MonitoringSnapshot(
        name="snapshot",
        results=[result],
    )

    assert isinstance(snapshot.results, tuple)


def test_snapshot_rejects_naive_timestamp():
    result = MonitoringResult(
        monitor_name="monitor",
        status=MonitoringStatus.OK,
    )

    with pytest.raises(ValueError):
        MonitoringSnapshot(
            name="snapshot",
            results=(result,),
            timestamp=datetime(2026, 1, 1),
        )


def test_snapshot_rejects_non_datetime_timestamp():
    result = MonitoringResult(
        monitor_name="monitor",
        status=MonitoringStatus.OK,
    )

    with pytest.raises(TypeError):
        MonitoringSnapshot(
            name="snapshot",
            results=(result,),
            timestamp="2026-01-01",
        )


@pytest.mark.parametrize(
    ("statuses", "expected"),
    [
        (
            [MonitoringStatus.OK],
            MonitoringStatus.OK,
        ),
        (
            [
                MonitoringStatus.OK,
                MonitoringStatus.WARNING,
            ],
            MonitoringStatus.WARNING,
        ),
        (
            [
                MonitoringStatus.OK,
                MonitoringStatus.CRITICAL,
                MonitoringStatus.WARNING,
            ],
            MonitoringStatus.CRITICAL,
        ),
        (
            [
                MonitoringStatus.CRITICAL,
                MonitoringStatus.ERROR,
            ],
            MonitoringStatus.ERROR,
        ),
    ],
)
def test_snapshot_returns_highest_severity_status(
    statuses,
    expected,
):
    results = tuple(
        MonitoringResult(
            monitor_name=f"monitor_{index}",
            status=status,
        )
        for index, status in enumerate(statuses)
    )

    snapshot = MonitoringSnapshot(
        name="snapshot",
        results=results,
    )

    assert snapshot.status == expected


def test_snapshot_is_frozen():
    result = MonitoringResult(
        monitor_name="monitor",
        status=MonitoringStatus.OK,
    )

    snapshot = MonitoringSnapshot(
        name="snapshot",
        results=(result,),
    )

    with pytest.raises(AttributeError):
        snapshot.name = "changed"