import math

import pytest

from src.validation.regime_transitions import (
    RegimeTransitionResult,
    RegimeTransitionValidator,
)


# ============================================================
# Basic Result Tests
# ============================================================


def test_validate_returns_result():
    result = RegimeTransitionValidator().validate(
        ["risk_on", "risk_on", "risk_off"],
        ["risk_on", "risk_on", "risk_off"],
    )

    assert isinstance(result, RegimeTransitionResult)


def test_no_transitions():
    result = RegimeTransitionValidator().validate(
        ["risk_on", "risk_on", "risk_on"],
        ["risk_on", "risk_on", "risk_on"],
    )

    assert result.actual_transition_count == 0
    assert result.predicted_transition_count == 0
    assert result.matched_transition_count == 0
    assert result.missed_transition_count == 0
    assert result.false_transition_count == 0
    assert result.transition_precision == 0.0
    assert result.transition_recall == 0.0
    assert result.transition_f1 == 0.0
    assert result.mean_detection_delay is None


def test_perfect_transition_detection():
    result = RegimeTransitionValidator().validate(
        ["risk_on", "risk_on", "risk_off", "risk_off"],
        ["risk_on", "risk_on", "risk_off", "risk_off"],
    )

    assert result.actual_transition_count == 1
    assert result.predicted_transition_count == 1
    assert result.matched_transition_count == 1
    assert result.missed_transition_count == 0
    assert result.false_transition_count == 0
    assert result.transition_precision == 1.0
    assert result.transition_recall == 1.0
    assert result.transition_f1 == 1.0
    assert result.mean_detection_delay == 0.0


def test_multiple_perfect_transitions():
    result = RegimeTransitionValidator().validate(
        ["a", "a", "b", "b", "c", "c"],
        ["a", "a", "b", "b", "c", "c"],
    )

    assert result.actual_transition_count == 2
    assert result.predicted_transition_count == 2
    assert result.matched_transition_count == 2
    assert result.transition_f1 == 1.0


def test_missed_transition():
    result = RegimeTransitionValidator().validate(
        ["a", "a", "b", "b"],
        ["a", "a", "a", "a"],
    )

    assert result.actual_transition_count == 1
    assert result.predicted_transition_count == 0
    assert result.matched_transition_count == 0
    assert result.missed_transition_count == 1
    assert result.false_transition_count == 0
    assert result.transition_precision == 0.0
    assert result.transition_recall == 0.0
    assert result.transition_f1 == 0.0


def test_false_transition():
    result = RegimeTransitionValidator().validate(
        ["a", "a", "a", "a"],
        ["a", "a", "b", "b"],
    )

    assert result.actual_transition_count == 0
    assert result.predicted_transition_count == 1
    assert result.matched_transition_count == 0
    assert result.missed_transition_count == 0
    assert result.false_transition_count == 1
    assert result.transition_precision == 0.0
    assert result.transition_recall == 0.0
    assert result.transition_f1 == 0.0


def test_partial_transition_detection():
    result = RegimeTransitionValidator().validate(
        ["a", "a", "b", "b", "c", "c"],
        ["a", "a", "b", "b", "b", "b"],
    )

    assert result.actual_transition_count == 2
    assert result.predicted_transition_count == 1
    assert result.matched_transition_count == 1
    assert result.missed_transition_count == 1
    assert result.false_transition_count == 0
    assert result.transition_precision == 1.0
    assert result.transition_recall == 0.5
    assert math.isclose(
        result.transition_f1,
        2.0 / 3.0,
        abs_tol=1e-12,
    )


def test_extra_predicted_transition():
    result = RegimeTransitionValidator().validate(
        ["a", "a", "b", "b", "b", "b"],
        ["a", "a", "b", "b", "c", "c"],
    )

    assert result.actual_transition_count == 1
    assert result.predicted_transition_count == 2
    assert result.matched_transition_count == 1
    assert result.missed_transition_count == 0
    assert result.false_transition_count == 1
    assert result.transition_precision == 0.5
    assert result.transition_recall == 1.0
    assert math.isclose(
        result.transition_f1,
        2.0 / 3.0,
        abs_tol=1e-12,
    )


# ============================================================
# Transition Position Tests
# ============================================================


def test_transition_indices():
    transitions = (
        RegimeTransitionValidator._find_transitions(
            ["a", "a", "b", "b", "c"]
        )
    )

    assert transitions == [2, 4]


def test_single_observation_has_no_transition():
    transitions = (
        RegimeTransitionValidator._find_transitions(
            ["risk_on"]
        )
    )

    assert transitions == []


def test_different_position_is_not_matched():
    result = RegimeTransitionValidator().validate(
        ["a", "a", "b", "b"],
        ["a", "b", "b", "b"],
    )

    assert result.actual_transition_count == 1
    assert result.predicted_transition_count == 1
    assert result.matched_transition_count == 0
    assert result.missed_transition_count == 1
    assert result.false_transition_count == 1


# ============================================================
# Detection Delay Tests
# ============================================================


def test_exact_transition_has_zero_delay():
    pairs = [(2, 2), (5, 5)]

    result = (
        RegimeTransitionValidator
        ._mean_detection_delay(pairs)
    )

    assert result == 0.0


def test_no_matched_transitions_has_no_delay():
    result = (
        RegimeTransitionValidator
        ._mean_detection_delay([])
    )

    assert result is None


def test_transition_matching_returns_exact_pairs():
    pairs = (
        RegimeTransitionValidator
        ._match_transitions(
            [2, 5, 8],
            [2, 8, 10],
        )
    )

    assert pairs == [(2, 2), (8, 8)]


# ============================================================
# Input Validation Tests
# ============================================================


def test_empty_actual():
    with pytest.raises(ValueError):
        RegimeTransitionValidator().validate(
            [],
            ["a"],
        )


def test_empty_predicted():
    with pytest.raises(ValueError):
        RegimeTransitionValidator().validate(
            ["a"],
            [],
        )


def test_mismatched_lengths():
    with pytest.raises(ValueError):
        RegimeTransitionValidator().validate(
            ["a", "b"],
            ["a"],
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
    with pytest.raises(TypeError):
        RegimeTransitionValidator().validate(
            actual,
            ["a"],
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
    with pytest.raises(TypeError):
        RegimeTransitionValidator().validate(
            ["a"],
            predicted,
        )


@pytest.mark.parametrize(
    "value",
    [
        123,
        None,
        True,
        [],
    ],
)
def test_invalid_actual_label_type(value):
    with pytest.raises(TypeError):
        RegimeTransitionValidator().validate(
            [value],
            ["a"],
        )


@pytest.mark.parametrize(
    "value",
    [
        123,
        None,
        True,
        [],
    ],
)
def test_invalid_predicted_label_type(value):
    with pytest.raises(TypeError):
        RegimeTransitionValidator().validate(
            ["a"],
            [value],
        )


@pytest.mark.parametrize(
    "value",
    [
        "",
        "   ",
        "\t",
    ],
)
def test_empty_actual_label(value):
    with pytest.raises(ValueError):
        RegimeTransitionValidator().validate(
            [value],
            ["a"],
        )


@pytest.mark.parametrize(
    "value",
    [
        "",
        "   ",
        "\n",
    ],
)
def test_empty_predicted_label(value):
    with pytest.raises(ValueError):
        RegimeTransitionValidator().validate(
            ["a"],
            [value],
        )


# ============================================================
# Helper Tests
# ============================================================


def test_safe_divide():
    assert (
        RegimeTransitionValidator._safe_divide(1, 2)
        == 0.5
    )
    assert (
        RegimeTransitionValidator._safe_divide(1, 0)
        == 0.0
    )


def test_calculate_f1():
    assert (
        RegimeTransitionValidator._calculate_f1(
            1.0,
            1.0,
        )
        == 1.0
    )

    assert (
        RegimeTransitionValidator._calculate_f1(
            0.0,
            0.0,
        )
        == 0.0
    )


def test_validate_sequences_returns_none():
    result = (
        RegimeTransitionValidator
        ._validate_sequences(
            ["a", "b"],
            ["a", "b"],
        )
    )

    assert result is None