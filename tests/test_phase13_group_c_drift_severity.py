import pytest

from src.monitoring.drift_severity import (
    DriftSeverity,
    DriftSeverityEvaluator,
)


def create_evaluator():
    return DriftSeverityEvaluator(
        {
            "medium": 0.1,
            "high": 0.3,
            "critical": 0.5,
        }
    )


def test_evaluator_stores_thresholds():
    evaluator = create_evaluator()

    assert evaluator.thresholds == {
        "medium": 0.1,
        "high": 0.3,
        "critical": 0.5,
    }


def test_score_below_medium_is_low():
    result = create_evaluator().evaluate(0.05)

    assert result.severity == DriftSeverity.LOW


def test_score_at_medium_is_medium():
    result = create_evaluator().evaluate(0.1)

    assert result.severity == DriftSeverity.MEDIUM


def test_score_between_medium_and_high_is_medium():
    result = create_evaluator().evaluate(0.2)

    assert result.severity == DriftSeverity.MEDIUM


def test_score_at_high_is_high():
    result = create_evaluator().evaluate(0.3)

    assert result.severity == DriftSeverity.HIGH


def test_score_between_high_and_critical_is_high():
    result = create_evaluator().evaluate(0.4)

    assert result.severity == DriftSeverity.HIGH


def test_score_at_critical_is_critical():
    result = create_evaluator().evaluate(0.5)

    assert result.severity == DriftSeverity.CRITICAL


def test_score_above_critical_is_critical():
    result = create_evaluator().evaluate(1.0)

    assert result.severity == DriftSeverity.CRITICAL


def test_result_contains_score_and_thresholds():
    result = create_evaluator().evaluate(0.35)

    assert result.score == 0.35
    assert result.thresholds == {
        "medium": 0.1,
        "high": 0.3,
        "critical": 0.5,
    }


@pytest.mark.parametrize(
    "invalid_thresholds",
    [
        {},
        {"medium": 0.1},
        {
            "medium": 0.1,
            "high": 0.3,
        },
        {
            "medium": 0.1,
            "high": 0.3,
            "critical": 0.5,
            "extra": 1.0,
        },
    ],
)
def test_evaluator_rejects_missing_or_extra_thresholds(
    invalid_thresholds,
):
    with pytest.raises(ValueError):
        DriftSeverityEvaluator(
            invalid_thresholds
        )


@pytest.mark.parametrize(
    "invalid_thresholds",
    [
        None,
        [],
        "invalid",
    ],
)
def test_evaluator_rejects_non_dictionary_thresholds(
    invalid_thresholds,
):
    with pytest.raises(TypeError):
        DriftSeverityEvaluator(
            invalid_thresholds
        )


@pytest.mark.parametrize(
    "invalid_thresholds",
    [
        {
            "medium": "0.1",
            "high": 0.3,
            "critical": 0.5,
        },
        {
            "medium": True,
            "high": 0.3,
            "critical": 0.5,
        },
        {
            "medium": 0.1,
            "high": None,
            "critical": 0.5,
        },
    ],
)
def test_evaluator_rejects_non_numeric_thresholds(
    invalid_thresholds,
):
    with pytest.raises(TypeError):
        DriftSeverityEvaluator(
            invalid_thresholds
        )


@pytest.mark.parametrize(
    "invalid_thresholds",
    [
        {
            "medium": -0.1,
            "high": 0.3,
            "critical": 0.5,
        },
        {
            "medium": 0.1,
            "high": -0.3,
            "critical": 0.5,
        },
    ],
)
def test_evaluator_rejects_negative_thresholds(
    invalid_thresholds,
):
    with pytest.raises(ValueError):
        DriftSeverityEvaluator(
            invalid_thresholds
        )


@pytest.mark.parametrize(
    "invalid_thresholds",
    [
        {
            "medium": 0.4,
            "high": 0.3,
            "critical": 0.5,
        },
        {
            "medium": 0.1,
            "high": 0.6,
            "critical": 0.5,
        },
    ],
)
def test_evaluator_rejects_unordered_thresholds(
    invalid_thresholds,
):
    with pytest.raises(ValueError):
        DriftSeverityEvaluator(
            invalid_thresholds
        )


@pytest.mark.parametrize(
    "invalid_score",
    [
        "0.1",
        None,
        True,
        [],
    ],
)
def test_evaluator_rejects_non_numeric_scores(
    invalid_score,
):
    evaluator = create_evaluator()

    with pytest.raises(TypeError):
        evaluator.evaluate(invalid_score)


@pytest.mark.parametrize(
    "invalid_score",
    [
        -0.1,
        -1.0,
    ],
)
def test_evaluator_rejects_negative_scores(
    invalid_score,
):
    evaluator = create_evaluator()

    with pytest.raises(ValueError):
        evaluator.evaluate(invalid_score)