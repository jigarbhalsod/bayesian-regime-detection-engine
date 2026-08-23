import pytest

from src.monitoring.alerts import (
    Alert,
    AlertRule,
    AlertSeverity,
)
from src.monitoring.models import (
    MonitoringResult,
    MonitoringStatus,
)


def create_result(status=MonitoringStatus.OK):
    return MonitoringResult(
        monitor_name="test_monitor",
        status=status,
        metrics=[],
        metadata={},
    )


# ============================================================
# AlertSeverity
# ============================================================


def test_alert_severity_values():
    assert AlertSeverity.INFO.value == "info"
    assert AlertSeverity.WARNING.value == "warning"
    assert AlertSeverity.CRITICAL.value == "critical"


# ============================================================
# Alert
# ============================================================


def test_alert_creates_successfully():
    alert = Alert(
        source="data_quality",
        severity=AlertSeverity.WARNING,
        message="Missing values exceeded threshold.",
    )

    assert alert.source == "data_quality"
    assert alert.severity == AlertSeverity.WARNING
    assert (
        alert.message
        == "Missing values exceeded threshold."
    )
    assert alert.metadata == {}


@pytest.mark.parametrize(
    "source",
    [
        "",
        "   ",
    ],
)
def test_alert_rejects_empty_source(source):
    with pytest.raises(ValueError):
        Alert(
            source=source,
            severity=AlertSeverity.WARNING,
            message="Test message",
        )


def test_alert_rejects_invalid_severity():
    with pytest.raises(TypeError):
        Alert(
            source="test",
            severity="warning",
            message="Test message",
        )


@pytest.mark.parametrize(
    "message",
    [
        "",
        "   ",
    ],
)
def test_alert_rejects_empty_message(message):
    with pytest.raises(ValueError):
        Alert(
            source="test",
            severity=AlertSeverity.WARNING,
            message=message,
        )


def test_alert_rejects_non_dictionary_metadata():
    with pytest.raises(TypeError):
        Alert(
            source="test",
            severity=AlertSeverity.WARNING,
            message="Test",
            metadata=[],
        )


def test_alert_normalizes_values():
    alert = Alert(
        source="  data_quality  ",
        severity=AlertSeverity.WARNING,
        message="  Threshold exceeded.  ",
        metadata={
            " metric ": "missing_ratio",
        },
    )

    assert alert.source == "data_quality"
    assert alert.message == "Threshold exceeded."
    assert alert.metadata == {
        "metric": "missing_ratio",
    }


def test_alert_rejects_empty_metadata_key():
    with pytest.raises(ValueError):
        Alert(
            source="test",
            severity=AlertSeverity.WARNING,
            message="Test",
            metadata={
                "   ": "value",
            },
        )


# ============================================================
# AlertRule
# ============================================================


def test_alert_rule_creates_with_defaults():
    rule = AlertRule(
        source="data_quality",
        severity=AlertSeverity.WARNING,
    )

    assert rule.source == "data_quality"
    assert rule.severity == AlertSeverity.WARNING
    assert rule.statuses == (
        MonitoringStatus.WARNING,
    )


@pytest.mark.parametrize(
    "source",
    [
        "",
        "   ",
    ],
)
def test_alert_rule_rejects_empty_source(source):
    with pytest.raises(ValueError):
        AlertRule(
            source=source,
            severity=AlertSeverity.WARNING,
        )


def test_alert_rule_rejects_invalid_severity():
    with pytest.raises(TypeError):
        AlertRule(
            source="test",
            severity="warning",
        )


def test_alert_rule_rejects_non_tuple_statuses():
    with pytest.raises(TypeError):
        AlertRule(
            source="test",
            severity=AlertSeverity.WARNING,
            statuses=[
                MonitoringStatus.WARNING,
            ],
        )


def test_alert_rule_rejects_empty_statuses():
    with pytest.raises(ValueError):
        AlertRule(
            source="test",
            severity=AlertSeverity.WARNING,
            statuses=(),
        )


def test_alert_rule_rejects_invalid_status():
    with pytest.raises(TypeError):
        AlertRule(
            source="test",
            severity=AlertSeverity.WARNING,
            statuses=("warning",),
        )


@pytest.mark.parametrize(
    "template",
    [
        "",
        "   ",
    ],
)
def test_alert_rule_rejects_empty_message_template(template):
    with pytest.raises(ValueError):
        AlertRule(
            source="test",
            severity=AlertSeverity.WARNING,
            message_template=template,
        )


# ============================================================
# Rule matching
# ============================================================


def test_alert_rule_matches_configured_status():
    rule = AlertRule(
        source="test",
        severity=AlertSeverity.WARNING,
        statuses=(
            MonitoringStatus.WARNING,
        ),
    )

    assert rule.matches(
        create_result(MonitoringStatus.WARNING)
    )


def test_alert_rule_does_not_match_other_status():
    rule = AlertRule(
        source="test",
        severity=AlertSeverity.WARNING,
    )

    assert not rule.matches(
        create_result(MonitoringStatus.OK)
    )


def test_alert_rule_supports_multiple_statuses():
    rule = AlertRule(
        source="test",
        severity=AlertSeverity.CRITICAL,
        statuses=(
            MonitoringStatus.OK,
            MonitoringStatus.WARNING,
        ),
    )

    assert rule.matches(
        create_result(MonitoringStatus.OK)
    )
    assert rule.matches(
        create_result(MonitoringStatus.WARNING)
    )


def test_alert_rule_rejects_invalid_result():
    rule = AlertRule(
        source="test",
        severity=AlertSeverity.WARNING,
    )

    with pytest.raises(TypeError):
        rule.matches("invalid")


# ============================================================
# Alert creation
# ============================================================


def test_create_alert_returns_none_when_not_matched():
    rule = AlertRule(
        source="test",
        severity=AlertSeverity.WARNING,
    )

    alert = rule.create_alert(
        create_result(MonitoringStatus.OK)
    )

    assert alert is None


def test_create_alert_returns_alert_when_matched():
    rule = AlertRule(
        source="data_quality",
        severity=AlertSeverity.WARNING,
    )

    alert = rule.create_alert(
        create_result(MonitoringStatus.WARNING)
    )

    assert isinstance(alert, Alert)
    assert alert.source == "data_quality"
    assert alert.severity == AlertSeverity.WARNING
    assert (
        alert.metadata["monitor_status"]
        == "warning"
    )


def test_create_alert_formats_message():
    rule = AlertRule(
        source="performance",
        severity=AlertSeverity.CRITICAL,
        message_template=(
            "Monitor {source} is {status}."
        ),
    )

    alert = rule.create_alert(
        create_result(MonitoringStatus.WARNING)
    )

    assert (
        alert.message
        == "Monitor performance is warning."
    )


def test_create_alert_rejects_invalid_result():
    rule = AlertRule(
        source="test",
        severity=AlertSeverity.WARNING,
    )

    with pytest.raises(TypeError):
        rule.create_alert("invalid")