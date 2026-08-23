from src.monitoring import (
    BaseMonitor,
    MetricResult,
    MonitoringConfig,
    MonitoringResult,
    MonitoringSnapshot,
    MonitoringStatus,
)


def test_monitoring_public_api_exports():
    assert BaseMonitor is not None
    assert MetricResult is not None
    assert MonitoringConfig is not None
    assert MonitoringResult is not None
    assert MonitoringSnapshot is not None
    assert MonitoringStatus is not None


def test_monitoring_components_work_together():
    config = MonitoringConfig(
        name="integration_monitor",
        thresholds={"warning": 0.5},
    )

    metric = MetricResult(
        name="drift_score",
        value=0.25,
        threshold=config.get_threshold("warning"),
        status=MonitoringStatus.OK,
    )

    result = MonitoringResult(
        monitor_name=config.name,
        status=MonitoringStatus.OK,
        metrics=(metric,),
    )

    snapshot = MonitoringSnapshot(
        name="integration_snapshot",
        results=(result,),
    )

    assert snapshot.status == MonitoringStatus.OK
    assert snapshot.results[0].monitor_name == "integration_monitor"
    assert snapshot.results[0].metrics[0].name == "drift_score"
    assert snapshot.results[0].metrics[0].value == 0.25