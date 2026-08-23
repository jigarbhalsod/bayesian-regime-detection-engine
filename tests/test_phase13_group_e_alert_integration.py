from __future__ import annotations

import pytest

from src.monitoring.alert_manager import AlertManager
from src.monitoring.alerts import (
    AlertRule,
    AlertSeverity,
)
from src.monitoring.models import (
    MonitoringResult,
    MonitoringStatus,
)
from src.monitoring.alert_integration import AlertIntegration


def create_result(
    name: str = "test_monitor",
    status: MonitoringStatus = MonitoringStatus.OK,
) -> MonitoringResult:
    return MonitoringResult(
        monitor_name=name,
        status=status,
        metrics={},
        metadata={},
    )


def create_alert_manager() -> AlertManager:
    rule = AlertRule(
        source="test_monitor",
        statuses=(MonitoringStatus.WARNING,),
        severity=AlertSeverity.WARNING,
        message_template="Warning from {source}",
    )

    return AlertManager(rules=[rule])


def create_integration(
    alert_manager: AlertManager | None = None,
) -> AlertIntegration:
    if alert_manager is None:
        alert_manager = create_alert_manager()

    return AlertIntegration(alert_manager=alert_manager)


def test_integration_creates_successfully():
    integration = create_integration()

    assert isinstance(integration, AlertIntegration)
    assert integration.alert_manager is not None


def test_integration_rejects_invalid_alert_manager():
    with pytest.raises(TypeError):
        AlertIntegration(alert_manager=None)

    with pytest.raises(TypeError):
        AlertIntegration(alert_manager="invalid")


def test_integration_returns_combined_report():
    integration = create_integration()

    results = [
        create_result(
            status=MonitoringStatus.OK,
        ),
    ]

    report = integration.execute(results)

    assert isinstance(report, dict)
    assert "results" in report
    assert "alerts" in report


def test_integration_handles_empty_results():
    integration = create_integration()

    report = integration.execute([])

    assert report["results"] == ()
    assert report["alerts"] == ()
    assert report["result_count"] == 0
    assert report["alert_count"] == 0


def test_integration_generates_matching_alerts():
    integration = create_integration()

    results = [
        create_result(
            name="test_monitor",
            status=MonitoringStatus.WARNING,
        ),
    ]

    report = integration.execute(results)

    assert report["result_count"] == 1
    assert report["alert_count"] == 1

    alert = report["alerts"][0]

    assert alert.source == "test_monitor"
    assert alert.severity == AlertSeverity.WARNING


def test_integration_returns_no_alerts_when_no_rules_match():
    integration = create_integration()

    results = [
        create_result(
            name="test_monitor",
            status=MonitoringStatus.OK,
        ),
    ]

    report = integration.execute(results)

    assert report["result_count"] == 1
    assert report["alert_count"] == 0
    assert report["alerts"] == ()


def test_integration_preserves_result_order():
    integration = create_integration()

    results = [
        create_result(
            name="first_monitor",
            status=MonitoringStatus.OK,
        ),
        create_result(
            name="second_monitor",
            status=MonitoringStatus.WARNING,
        ),
        create_result(
            name="third_monitor",
            status=MonitoringStatus.OK,
        ),
    ]

    report = integration.execute(results)

    returned_names = tuple(
        result.monitor_name
        for result in report["results"]
    )

    assert returned_names == (
        "first_monitor",
        "second_monitor",
        "third_monitor",
    )


def test_integration_preserves_alert_order():
    rule_one = AlertRule(
        source="first_monitor",
        statuses=(MonitoringStatus.WARNING,),
        severity=AlertSeverity.WARNING,
        message_template="First warning",
    )

    rule_two = AlertRule(
        source="second_monitor",
        statuses=(MonitoringStatus.WARNING,),
        severity=AlertSeverity.WARNING,
        message_template="Second warning",
    )

    alert_manager = AlertManager(
        rules=[
            rule_one,
            rule_two,
        ]
    )

    integration = create_integration(
        alert_manager=alert_manager,
    )

    results = [
        create_result(
            name="first_monitor",
            status=MonitoringStatus.WARNING,
        ),
        create_result(
            name="second_monitor",
            status=MonitoringStatus.WARNING,
        ),
    ]

    report = integration.execute(results)

    alert_sources = tuple(
        alert.source
        for alert in report["alerts"]
    )

    assert alert_sources == (
        "first_monitor",
        "second_monitor",
    )


def test_integration_includes_result_count():
    integration = create_integration()

    results = [
        create_result(name="one"),
        create_result(name="two"),
        create_result(name="three"),
    ]

    report = integration.execute(results)

    assert report["result_count"] == 3


def test_integration_includes_alert_count():
    integration = create_integration()

    results = [
        create_result(
            name="test_monitor",
            status=MonitoringStatus.WARNING,
        ),
        create_result(
            name="test_monitor",
            status=MonitoringStatus.WARNING,
        ),
    ]

    report = integration.execute(results)

    assert report["alert_count"] == 2


@pytest.mark.parametrize(
    "invalid_results",
    [
        None,
        123,
        "invalid",
    ],
)
def test_integration_rejects_non_iterable_results(
    invalid_results,
):
    integration = create_integration()

    with pytest.raises(TypeError):
        integration.execute(invalid_results)


@pytest.mark.parametrize(
    "invalid_results",
    [
        [None],
        ["invalid"],
        [123],
    ],
)
def test_integration_rejects_invalid_result_instances(
    invalid_results,
):
    integration = create_integration()

    with pytest.raises(TypeError):
        integration.execute(invalid_results)