import pytest

from src.validation.base import ValidationResult
from src.validation.reporting import (
    ValidationReporter,
    ValidationSummary,
)


def make_result(
    name: str,
    is_valid: bool,
    metrics: dict[str, float] | None = None,
) -> ValidationResult:
    return ValidationResult(
        validator_name=name,
        is_valid=is_valid,
        metrics={} if metrics is None else metrics,
    )


# ============================================================
# ValidationSummary Tests
# ============================================================


def test_summary_values():
    summary = ValidationSummary(
        total_validators=3,
        passed_validators=2,
        failed_validators=1,
        overall_valid=False,
        failed_validator_names=("third",),
        aggregated_metrics={"accuracy": 0.9},
    )

    assert summary.total_validators == 3
    assert summary.passed_validators == 2
    assert summary.failed_validators == 1
    assert summary.overall_valid is False
    assert summary.failed_validator_names == ("third",)
    assert summary.aggregated_metrics == {"accuracy": 0.9}


def test_summary_is_immutable():
    summary = ValidationSummary(
        total_validators=1,
        passed_validators=1,
        failed_validators=0,
        overall_valid=True,
        failed_validator_names=(),
        aggregated_metrics={},
    )

    with pytest.raises(Exception):
        summary.total_validators = 10


# ============================================================
# Basic Summary Tests
# ============================================================


def test_summarize_returns_summary():
    result = ValidationReporter().summarize(
        [make_result("first", True)]
    )

    assert isinstance(result, ValidationSummary)


def test_all_valid_results():
    results = [
        make_result("first", True),
        make_result("second", True),
        make_result("third", True),
    ]

    summary = ValidationReporter().summarize(results)

    assert summary.total_validators == 3
    assert summary.passed_validators == 3
    assert summary.failed_validators == 0
    assert summary.overall_valid is True
    assert summary.failed_validator_names == ()


def test_all_invalid_results():
    results = [
        make_result("first", False),
        make_result("second", False),
    ]

    summary = ValidationReporter().summarize(results)

    assert summary.total_validators == 2
    assert summary.passed_validators == 0
    assert summary.failed_validators == 2
    assert summary.overall_valid is False
    assert summary.failed_validator_names == (
        "first",
        "second",
    )


def test_mixed_results():
    results = [
        make_result("first", True),
        make_result("second", False),
        make_result("third", True),
    ]

    summary = ValidationReporter().summarize(results)

    assert summary.total_validators == 3
    assert summary.passed_validators == 2
    assert summary.failed_validators == 1
    assert summary.overall_valid is False
    assert summary.failed_validator_names == ("second",)


def test_failed_validator_order_is_preserved():
    results = [
        make_result("first", False),
        make_result("second", True),
        make_result("third", False),
        make_result("fourth", False),
    ]

    summary = ValidationReporter().summarize(results)

    assert summary.failed_validator_names == (
        "first",
        "third",
        "fourth",
    )


# ============================================================
# Metric Aggregation Tests
# ============================================================


def test_no_metrics_returns_empty_dictionary():
    summary = ValidationReporter().summarize(
        [
            make_result("first", True),
            make_result("second", True),
        ]
    )

    assert summary.aggregated_metrics == {}


def test_single_metric():
    summary = ValidationReporter().summarize(
        [
            make_result(
                "first",
                True,
                {"accuracy": 0.8},
            ),
            make_result(
                "second",
                True,
                {"accuracy": 1.0},
            ),
        ]
    )

    assert summary.aggregated_metrics["accuracy"] == 0.9


def test_multiple_metrics():
    summary = ValidationReporter().summarize(
        [
            make_result(
                "first",
                True,
                {
                    "accuracy": 0.8,
                    "precision": 0.6,
                },
            ),
            make_result(
                "second",
                True,
                {
                    "accuracy": 1.0,
                    "precision": 0.8,
                },
            ),
        ]
    )

    assert summary.aggregated_metrics == {
        "accuracy": 0.9,
        "precision": 0.7,
    }


def test_metric_present_in_only_one_result():
    summary = ValidationReporter().summarize(
        [
            make_result(
                "first",
                True,
                {"accuracy": 0.8},
            ),
            make_result(
                "second",
                True,
                {"precision": 0.6},
            ),
        ]
    )

    assert summary.aggregated_metrics == {
        "accuracy": 0.8,
        "precision": 0.6,
    }


def test_metric_aggregation_includes_failed_validators():
    summary = ValidationReporter().summarize(
        [
            make_result(
                "passed",
                True,
                {"accuracy": 1.0},
            ),
            make_result(
                "failed",
                False,
                {"accuracy": 0.5},
            ),
        ]
    )

    assert summary.aggregated_metrics["accuracy"] == 0.75


def test_aggregate_metrics_returns_mean_values():
    results = [
        make_result("first", True, {"score": 1.0}),
        make_result("second", True, {"score": 2.0}),
        make_result("third", True, {"score": 3.0}),
    ]

    metrics = ValidationReporter._aggregate_metrics(results)

    assert metrics == {"score": 2.0}


# ============================================================
# Dictionary Export Tests
# ============================================================


def test_to_dict():
    summary = ValidationSummary(
        total_validators=2,
        passed_validators=1,
        failed_validators=1,
        overall_valid=False,
        failed_validator_names=("second",),
        aggregated_metrics={"accuracy": 0.75},
    )

    exported = summary.to_dict()

    assert exported == {
        "total_validators": 2,
        "passed_validators": 1,
        "failed_validators": 1,
        "overall_valid": False,
        "failed_validator_names": ("second",),
        "aggregated_metrics": {"accuracy": 0.75},
    }


def test_to_dict_returns_new_dictionary():
    summary = ValidationSummary(
        total_validators=1,
        passed_validators=1,
        failed_validators=0,
        overall_valid=True,
        failed_validator_names=(),
        aggregated_metrics={},
    )

    first = summary.to_dict()
    second = summary.to_dict()

    assert first == second
    assert first is not second


# ============================================================
# Input Validation Tests
# ============================================================


def test_empty_results():
    with pytest.raises(ValueError):
        ValidationReporter().summarize([])


@pytest.mark.parametrize(
    "results",
    [
        "invalid",
        b"invalid",
        123,
        {"result": "invalid"},
    ],
)
def test_invalid_results_type(results):
    with pytest.raises(TypeError):
        ValidationReporter().summarize(results)


def test_invalid_result_object():
    with pytest.raises(TypeError):
        ValidationReporter().summarize(
            ["invalid"]
        )


def test_mixed_valid_and_invalid_object_types():
    results = [
        make_result("valid", True),
        "invalid",
    ]

    with pytest.raises(TypeError):
        ValidationReporter().summarize(results)


def test_validate_results_returns_none():
    results = [
        make_result("first", True),
    ]

    value = ValidationReporter._validate_results(results)

    assert value is None