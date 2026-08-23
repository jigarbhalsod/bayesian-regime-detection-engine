import math

import pytest

from src.validation.metrics import (
    ClassMetrics,
    PredictionMetrics,
    PredictionMetricsResult,
)


# ============================================================
# Basic Result Tests
# ============================================================


def test_perfect_predictions():
    metrics = PredictionMetrics()

    result = metrics.evaluate(
        actual=["risk_on", "risk_off", "transition"],
        predicted=["risk_on", "risk_off", "transition"],
    )

    assert isinstance(result, PredictionMetricsResult)
    assert result.accuracy == 1.0
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.f1_score == 1.0
    assert result.balanced_accuracy == 1.0


def test_completely_incorrect_predictions():
    metrics = PredictionMetrics()

    result = metrics.evaluate(
        actual=["risk_on", "risk_on"],
        predicted=["risk_off", "risk_off"],
    )

    assert result.accuracy == 0.0
    assert result.per_class["risk_on"].recall == 0.0
    assert result.per_class["risk_off"].precision == 0.0


def test_result_contains_class_metrics():
    metrics = PredictionMetrics()

    result = metrics.evaluate(
        actual=["a", "a", "b", "b"],
        predicted=["a", "b", "b", "b"],
    )

    assert isinstance(result.per_class["a"], ClassMetrics)
    assert isinstance(result.per_class["b"], ClassMetrics)


# ============================================================
# Accuracy Tests
# ============================================================


def test_accuracy():
    metrics = PredictionMetrics()

    result = metrics.evaluate(
        actual=["a", "a", "b", "b"],
        predicted=["a", "b", "b", "b"],
    )

    assert result.accuracy == 0.75


# ============================================================
# Per-Class Metric Tests
# ============================================================


def test_per_class_metrics():
    metrics = PredictionMetrics()

    result = metrics.evaluate(
        actual=["a", "a", "b", "b"],
        predicted=["a", "b", "b", "b"],
    )

    class_a = result.per_class["a"]
    class_b = result.per_class["b"]

    assert class_a.precision == 1.0
    assert class_a.recall == 0.5
    assert math.isclose(
        class_a.f1_score,
        2 / 3,
    )
    assert class_a.support == 2

    assert math.isclose(
        class_b.precision,
        2 / 3,
    )
    assert class_b.recall == 1.0
    assert math.isclose(
        class_b.f1_score,
        0.8,
    )
    assert class_b.support == 2


# ============================================================
# Macro Metrics Tests
# ============================================================


def test_macro_metrics():
    metrics = PredictionMetrics()

    result = metrics.evaluate(
        actual=["a", "a", "b", "b"],
        predicted=["a", "b", "b", "b"],
    )

    assert math.isclose(
        result.precision,
        (1.0 + (2 / 3)) / 2,
    )

    assert math.isclose(
        result.recall,
        (0.5 + 1.0) / 2,
    )

    assert math.isclose(
        result.f1_score,
        ((2 / 3) + 0.8) / 2,
    )


def test_balanced_accuracy_equals_macro_recall():
    metrics = PredictionMetrics()

    result = metrics.evaluate(
        actual=["a", "a", "b", "b"],
        predicted=["a", "b", "b", "b"],
    )

    assert result.balanced_accuracy == result.recall


# ============================================================
# Predicted-Only Class Tests
# ============================================================


def test_predicted_only_class_is_included():
    metrics = PredictionMetrics()

    result = metrics.evaluate(
        actual=["a", "a"],
        predicted=["a", "c"],
    )

    assert "c" in result.per_class
    assert result.per_class["c"].support == 0
    assert result.per_class["c"].precision == 0.0
    assert result.per_class["c"].recall == 0.0
    assert result.per_class["c"].f1_score == 0.0


def test_actual_only_class_is_included():
    metrics = PredictionMetrics()

    result = metrics.evaluate(
        actual=["a", "b"],
        predicted=["a", "a"],
    )

    assert "b" in result.per_class
    assert result.per_class["b"].support == 1
    assert result.per_class["b"].precision == 0.0
    assert result.per_class["b"].recall == 0.0


# ============================================================
# Input Validation Tests
# ============================================================


@pytest.mark.parametrize(
    "actual, predicted",
    [
        ([], []),
        ([], ["a"]),
        (["a"], []),
    ],
)
def test_empty_inputs(actual, predicted):
    metrics = PredictionMetrics()

    with pytest.raises(ValueError):
        metrics.evaluate(actual, predicted)


def test_mismatched_lengths():
    metrics = PredictionMetrics()

    with pytest.raises(ValueError):
        metrics.evaluate(
            actual=["a", "b"],
            predicted=["a"],
        )


@pytest.mark.parametrize(
    "actual",
    [
        "invalid",
        b"invalid",
        123,
        {"a": 1},
    ],
)
def test_invalid_actual_type(actual):
    metrics = PredictionMetrics()

    with pytest.raises(TypeError):
        metrics.evaluate(
            actual=actual,
            predicted=["a"],
        )


@pytest.mark.parametrize(
    "predicted",
    [
        "invalid",
        b"invalid",
        123,
        {"a": 1},
    ],
)
def test_invalid_predicted_type(predicted):
    metrics = PredictionMetrics()

    with pytest.raises(TypeError):
        metrics.evaluate(
            actual=["a"],
            predicted=predicted,
        )


# ============================================================
# Helper Method Tests
# ============================================================


def test_safe_divide():
    assert PredictionMetrics._safe_divide(1, 2) == 0.5
    assert PredictionMetrics._safe_divide(1, 0) == 0.0


def test_calculate_f1():
    assert PredictionMetrics._calculate_f1(1.0, 1.0) == 1.0
    assert PredictionMetrics._calculate_f1(0.0, 0.0) == 0.0
    assert math.isclose(
        PredictionMetrics._calculate_f1(1.0, 0.5),
        2 / 3,
    )


def test_mean():
    assert PredictionMetrics._mean([1, 2, 3]) == 2.0
    assert PredictionMetrics._mean([]) == 0.0