import pytest

from src.monitoring.config import MonitoringConfig


def test_config_creates_with_defaults():
    config = MonitoringConfig(name="data_quality")

    assert config.name == "data_quality"
    assert config.enabled is True
    assert config.interval_seconds == 60
    assert config.thresholds == {}
    assert config.metadata == {}


def test_config_normalizes_name():
    config = MonitoringConfig(name="  drift_monitor  ")

    assert config.name == "drift_monitor"


@pytest.mark.parametrize(
    "name",
    [
        "",
        "   ",
    ],
)
def test_config_rejects_empty_name(name):
    with pytest.raises(ValueError):
        MonitoringConfig(name=name)


@pytest.mark.parametrize(
    "interval",
    [
        0,
        -1,
        -100,
    ],
)
def test_config_rejects_invalid_interval(interval):
    with pytest.raises(ValueError):
        MonitoringConfig(
            name="monitor",
            interval_seconds=interval,
        )


def test_config_normalizes_thresholds():
    config = MonitoringConfig(
        name="monitor",
        thresholds={
            " warning ": 0.5,
            "critical": 0.9,
        },
    )

    assert config.thresholds == {
        "warning": 0.5,
        "critical": 0.9,
    }


def test_config_rejects_empty_threshold_name():
    with pytest.raises(ValueError):
        MonitoringConfig(
            name="monitor",
            thresholds={" ": 0.5},
        )


@pytest.mark.parametrize(
    "value",
    [
        "0.5",
        None,
        True,
        False,
        [],
    ],
)
def test_config_rejects_non_numeric_threshold(value):
    with pytest.raises(TypeError):
        MonitoringConfig(
            name="monitor",
            thresholds={"warning": value},
        )


def test_get_threshold_returns_configured_value():
    config = MonitoringConfig(
        name="monitor",
        thresholds={"warning": 0.5},
    )

    assert config.get_threshold("warning") == 0.5


def test_get_threshold_returns_default():
    config = MonitoringConfig(name="monitor")

    assert config.get_threshold(
        "missing",
        1.0,
    ) == 1.0


def test_config_rejects_empty_metadata_key():
    with pytest.raises(ValueError):
        MonitoringConfig(
            name="monitor",
            metadata={" ": "value"},
        )


def test_config_is_frozen():
    config = MonitoringConfig(name="monitor")

    with pytest.raises(
        AttributeError
    ):
        config.name = "changed"