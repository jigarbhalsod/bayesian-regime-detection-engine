import pandas as pd

from src.monitoring import (
    DataQualityMonitor,
    MonitoringConfig,
    MonitoringSnapshot,
    MonitoringStatus,
    SchemaMonitor,
    StatisticalDataMonitor,
)


def test_group_b_public_api_exports():
    assert DataQualityMonitor is not None
    assert SchemaMonitor is not None
    assert StatisticalDataMonitor is not None


def test_group_b_monitors_work_together():
    data = pd.DataFrame(
        {
            "price": [100.0, 101.0, 102.0],
            "volume": [1000, 1100, 1200],
        }
    )

    data_quality_monitor = DataQualityMonitor(
        MonitoringConfig(name="data_quality")
    )

    schema_monitor = SchemaMonitor(
        MonitoringConfig(
            name="schema",
            metadata={
                "expected_schema": {
                    "price": "float64",
                    "volume": "int64",
                }
            },
        )
    )

    statistics_monitor = StatisticalDataMonitor(
        MonitoringConfig(name="statistics")
    )

    results = (
        data_quality_monitor.execute(data),
        schema_monitor.execute(data),
        statistics_monitor.execute(data),
    )

    snapshot = MonitoringSnapshot(
        name="group_b_snapshot",
        results=results,
    )

    assert len(snapshot.results) == 3
    assert snapshot.status == MonitoringStatus.OK

    assert snapshot.results[0].monitor_name == "data_quality"
    assert snapshot.results[1].monitor_name == "schema"
    assert snapshot.results[2].monitor_name == "statistics"


def test_group_b_snapshot_aggregates_warning_status():
    data = pd.DataFrame(
        {
            "price": [100.0, None],
            "volume": [1000, 1100],
        }
    )

    data_quality_monitor = DataQualityMonitor(
        MonitoringConfig(
            name="data_quality",
            thresholds={"missing_ratio": 0.1},
        )
    )

    schema_monitor = SchemaMonitor(
        MonitoringConfig(
            name="schema",
            metadata={
                "expected_schema": {
                    "price": "float64",
                    "volume": "int64",
                }
            },
        )
    )

    statistics_monitor = StatisticalDataMonitor(
        MonitoringConfig(name="statistics")
    )

    snapshot = MonitoringSnapshot(
        name="warning_snapshot",
        results=(
            data_quality_monitor.execute(data),
            schema_monitor.execute(data),
            statistics_monitor.execute(data),
        ),
    )

    assert snapshot.status == MonitoringStatus.WARNING