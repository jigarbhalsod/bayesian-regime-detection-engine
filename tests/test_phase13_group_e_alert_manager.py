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


def create_result(
    monitor_name="test_monitor",
    status=MonitoringStatus.OK,
):
    return MonitoringResult(
        monitor_name=monitor_name,
        status=status,
        metrics=[],
        metadata={},
    )


def create_rule(
    source="test_monitor",
    severity=AlertSeverity.WARNING,
    statuses=(MonitoringStatus.WARNING,),
):
    return AlertRule(
        source=source,
        severity=severity,
        statuses=statuses,
    )


# ============================================================
# Initialization
# ============================================================


def test_alert_manager_creates_successfully():
    rule = create_rule()

    manager = AlertManager([rule])

    assert manager.rules == (rule,)


def test_alert_manager_accepts_empty_rules():
    manager = AlertManager([])

    assert manager.rules == ()


def test_alert_manager_rejects_non_iterable_rules():
    with pytest.raises(TypeError):
        AlertManager(None)


@pytest.mark.parametrize(
    "rules",
    [
        [create_rule(), "invalid"],
        ["invalid"],
        [None],
    ],
)
def test_alert_manager_rejects_invalid_rule_instances(
    rules,
):
    with pytest.raises(TypeError):
        AlertManager(rules)


def test_alert_manager_stores_rules_as_tuple():
    manager = AlertManager(
        [
            create_rule(source="monitor_a"),
            create_rule(source="monitor_b"),
        ]
    )

    assert isinstance(manager.rules, tuple)
    assert len(manager.rules) == 2


# ============================================================
# Multiple result evaluation
# ============================================================


def test_evaluate_returns_empty_for_no_results():
    manager = AlertManager([create_rule()])

    alerts = manager.evaluate([])

    assert alerts == []


def test_evaluate_returns_empty_when_no_rules_match():
    manager = AlertManager(
        [
            create_rule(source="other_monitor"),
        ]
    )

    alerts = manager.evaluate(
        [
            create_result(
                monitor_name="test_monitor",
                status=MonitoringStatus.WARNING,
            )
        ]
    )

    assert alerts == []


def test_evaluate_returns_empty_for_non_matching_status():
    manager = AlertManager(
        [
            create_rule(
                statuses=(MonitoringStatus.WARNING,),
            )
        ]
    )

    alerts = manager.evaluate(
        [
            create_result(
                status=MonitoringStatus.OK,
            )
        ]
    )

    assert alerts == []


def test_evaluate_creates_alert_for_matching_result():
    manager = AlertManager([create_rule()])

    alerts = manager.evaluate(
        [
            create_result(
                status=MonitoringStatus.WARNING,
            )
        ]
    )

    assert len(alerts) == 1
    assert alerts[0].source == "test_monitor"
    assert alerts[0].severity == AlertSeverity.WARNING


def test_evaluate_handles_multiple_results():
    manager = AlertManager(
        [
            create_rule(source="monitor_a"),
            create_rule(source="monitor_b"),
        ]
    )

    alerts = manager.evaluate(
        [
            create_result(
                monitor_name="monitor_a",
                status=MonitoringStatus.WARNING,
            ),
            create_result(
                monitor_name="monitor_b",
                status=MonitoringStatus.WARNING,
            ),
            create_result(
                monitor_name="monitor_c",
                status=MonitoringStatus.WARNING,
            ),
        ]
    )

    assert len(alerts) == 2
    assert alerts[0].source == "monitor_a"
    assert alerts[1].source == "monitor_b"


def test_evaluate_handles_multiple_matching_rules():
    manager = AlertManager(
        [
            create_rule(
                severity=AlertSeverity.WARNING,
            ),
            create_rule(
                severity=AlertSeverity.CRITICAL,
            ),
        ]
    )

    alerts = manager.evaluate(
        [
            create_result(
                status=MonitoringStatus.WARNING,
            )
        ]
    )

    assert len(alerts) == 2
    assert alerts[0].severity == AlertSeverity.WARNING
    assert alerts[1].severity == AlertSeverity.CRITICAL


def test_evaluate_preserves_result_order():
    manager = AlertManager(
        [
            create_rule(source="monitor_a"),
            create_rule(source="monitor_b"),
        ]
    )

    alerts = manager.evaluate(
        [
            create_result(
                monitor_name="monitor_b",
                status=MonitoringStatus.WARNING,
            ),
            create_result(
                monitor_name="monitor_a",
                status=MonitoringStatus.WARNING,
            ),
        ]
    )

    assert alerts[0].source == "monitor_b"
    assert alerts[1].source == "monitor_a"


# ============================================================
# Input validation
# ============================================================


@pytest.mark.parametrize(
    "results",
    [
        None,
        123,
        "invalid",
    ],
)
def test_evaluate_rejects_non_iterable_results(results):
    manager = AlertManager([create_rule()])

    with pytest.raises(TypeError):
        manager.evaluate(results)


@pytest.mark.parametrize(
    "results",
    [
        ["invalid"],
        [None],
        [create_result(), "invalid"],
    ],
)
def test_evaluate_rejects_invalid_result_instances(
    results,
):
    manager = AlertManager([create_rule()])

    with pytest.raises(TypeError):
        manager.evaluate(results)


def test_evaluate_accepts_tuple_results():
    manager = AlertManager([create_rule()])

    alerts = manager.evaluate(
        (
            create_result(
                status=MonitoringStatus.WARNING,
            ),
        )
    )

    assert len(alerts) == 1


def test_evaluate_accepts_generator_results():
    manager = AlertManager([create_rule()])

    results = (
        create_result(status=MonitoringStatus.WARNING)
        for _ in range(2)
    )

    alerts = manager.evaluate(results)

    assert len(alerts) == 2


# ============================================================
# Single result evaluation
# ============================================================


def test_evaluate_one_creates_matching_alert():
    manager = AlertManager([create_rule()])

    alerts = manager.evaluate_one(
        create_result(
            status=MonitoringStatus.WARNING,
        )
    )

    assert len(alerts) == 1
    assert alerts[0].source == "test_monitor"


def test_evaluate_one_returns_empty_for_non_matching_result():
    manager = AlertManager([create_rule()])

    alerts = manager.evaluate_one(
        create_result(
            status=MonitoringStatus.OK,
        )
    )

    assert alerts == []


def test_evaluate_one_rejects_invalid_result():
    manager = AlertManager([create_rule()])

    with pytest.raises(TypeError):
        manager.evaluate_one("invalid")