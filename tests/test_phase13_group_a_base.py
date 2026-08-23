import pytest

from src.monitoring.base import BaseMonitor
from src.monitoring.config import MonitoringConfig
from src.monitoring.models import (
    MonitoringResult,
    MonitoringStatus,
)


class ValidMonitor(BaseMonitor):
    def monitor(self, data):
        return MonitoringResult(
            monitor_name=self.name,
            status=MonitoringStatus.OK,
            metadata={"received_data": data},
        )


class InvalidReturnMonitor(BaseMonitor):
    def monitor(self, data):
        return "invalid"


class WrongNameMonitor(BaseMonitor):
    def monitor(self, data):
        return MonitoringResult(
            monitor_name="different_monitor",
            status=MonitoringStatus.OK,
        )


def test_base_monitor_cannot_be_instantiated_directly():
    config = MonitoringConfig(name="monitor")

    with pytest.raises(TypeError):
        BaseMonitor(config)


def test_base_monitor_rejects_invalid_config():
    with pytest.raises(TypeError):
        ValidMonitor("invalid")


def test_base_monitor_stores_config():
    config = MonitoringConfig(
        name="test_monitor",
        interval_seconds=30,
    )

    monitor = ValidMonitor(config)

    assert monitor.config is config
    assert monitor.name == "test_monitor"
    assert monitor.enabled is True


def test_base_monitor_reports_disabled_status():
    config = MonitoringConfig(
        name="test_monitor",
        enabled=False,
    )

    monitor = ValidMonitor(config)

    assert monitor.enabled is False


def test_execute_returns_monitoring_result():
    monitor = ValidMonitor(
        MonitoringConfig(name="test_monitor")
    )

    result = monitor.execute({"value": 10})

    assert isinstance(result, MonitoringResult)
    assert result.monitor_name == "test_monitor"
    assert result.status == MonitoringStatus.OK


def test_execute_passes_data_to_monitor():
    monitor = ValidMonitor(
        MonitoringConfig(name="test_monitor")
    )

    data = {"market": "NIFTY", "value": 100}

    result = monitor.execute(data)

    assert result.metadata["received_data"] == data


def test_execute_rejects_disabled_monitor():
    monitor = ValidMonitor(
        MonitoringConfig(
            name="test_monitor",
            enabled=False,
        )
    )

    with pytest.raises(
        RuntimeError,
        match="disabled",
    ):
        monitor.execute({"value": 10})


def test_execute_rejects_invalid_monitor_return_type():
    monitor = InvalidReturnMonitor(
        MonitoringConfig(name="invalid_monitor")
    )

    with pytest.raises(
        TypeError,
        match="MonitoringResult",
    ):
        monitor.execute({"value": 10})


def test_execute_rejects_wrong_monitor_name():
    monitor = WrongNameMonitor(
        MonitoringConfig(name="correct_monitor")
    )

    with pytest.raises(
        ValueError,
        match="monitor_name",
    ):
        monitor.execute({"value": 10})


def test_monitor_is_abstract_method():
    assert getattr(
        BaseMonitor.monitor,
        "__isabstractmethod__",
        False,
    ) is True