import math

import pytest

from src.validation.calibration import (
    CalibrationBin,
    CalibrationValidationResult,
    ProbabilityCalibrationValidator,
)


# ============================================================
# Configuration Tests
# ============================================================


def test_default_configuration():
    validator = ProbabilityCalibrationValidator()

    assert validator.num_bins == 10
    assert validator.tolerance == 1e-9


def test_custom_configuration():
    validator = ProbabilityCalibrationValidator(
        num_bins=5,
        tolerance=1e-6,
    )

    assert validator.num_bins == 5
    assert validator.tolerance == 1e-6


@pytest.mark.parametrize(
    "value",
    [
        0,
        -1,
        1.5,
        "invalid",
        True,
    ],
)
def test_invalid_num_bins(value):
    with pytest.raises((TypeError, ValueError)):
        ProbabilityCalibrationValidator(num_bins=value)


@pytest.mark.parametrize(
    "value",
    [
        -1.0,
        "invalid",
        True,
    ],
)
def test_invalid_tolerance(value):
    with pytest.raises((TypeError, ValueError)):
        ProbabilityCalibrationValidator(tolerance=value)


# ============================================================
# Perfect Prediction Tests
# ============================================================


def test_perfect_predictions():
    validator = ProbabilityCalibrationValidator(
        num_bins=5,
    )

    result = validator.evaluate(
        actual=["a", "b"],
        probabilities=[
            {"a": 1.0, "b": 0.0},
            {"a": 0.0, "b": 1.0},
        ],
    )

    assert isinstance(result, CalibrationValidationResult)
    assert result.brier_score == 0.0
    assert result.log_loss >= 0.0
    assert result.log_loss < 1e-10
    assert result.expected_calibration_error == 0.0
    assert result.maximum_calibration_error == 0.0


def test_result_contains_bins():
    validator = ProbabilityCalibrationValidator(
        num_bins=4,
    )

    result = validator.evaluate(
        actual=["a", "b"],
        probabilities=[
            {"a": 0.8, "b": 0.2},
            {"a": 0.1, "b": 0.9},
        ],
    )

    assert len(result.bins) == 4
    assert all(
        isinstance(bin_, CalibrationBin)
        for bin_ in result.bins
    )


# ============================================================
# Metric Tests
# ============================================================


def test_brier_score():
    validator = ProbabilityCalibrationValidator()

    result = validator.evaluate(
        actual=["a"],
        probabilities=[
            {"a": 0.8, "b": 0.2},
        ],
    )

    expected = (0.2 ** 2) + (0.2 ** 2)

    assert math.isclose(
        result.brier_score,
        expected,
    )


def test_log_loss():
    validator = ProbabilityCalibrationValidator()

    result = validator.evaluate(
        actual=["a"],
        probabilities=[
            {"a": 0.8, "b": 0.2},
        ],
    )

    assert math.isclose(
        result.log_loss,
        -math.log(0.8),
    )


def test_brier_score_penalizes_bad_predictions():
    validator = ProbabilityCalibrationValidator()

    good = validator.evaluate(
        actual=["a"],
        probabilities=[
            {"a": 0.9, "b": 0.1},
        ],
    )

    bad = validator.evaluate(
        actual=["a"],
        probabilities=[
            {"a": 0.1, "b": 0.9},
        ],
    )

    assert good.brier_score < bad.brier_score
    assert good.log_loss < bad.log_loss


# ============================================================
# Calibration Tests
# ============================================================


def test_calibration_bin_bounds():
    validator = ProbabilityCalibrationValidator(
        num_bins=4,
    )

    result = validator.evaluate(
        actual=["a"],
        probabilities=[
            {"a": 0.75, "b": 0.25},
        ],
    )

    assert result.bins[0].lower_bound == 0.0
    assert result.bins[0].upper_bound == 0.25

    assert result.bins[-1].lower_bound == 0.75
    assert result.bins[-1].upper_bound == 1.0


def test_confidence_one_goes_to_last_bin():
    validator = ProbabilityCalibrationValidator(
        num_bins=5,
    )

    result = validator.evaluate(
        actual=["a"],
        probabilities=[
            {"a": 1.0, "b": 0.0},
        ],
    )

    assert result.bins[-1].count == 1


def test_expected_calibration_error():
    validator = ProbabilityCalibrationValidator(
        num_bins=2,
    )

    result = validator.evaluate(
        actual=["a", "b"],
        probabilities=[
            {"a": 0.8, "b": 0.2},
            {"a": 0.8, "b": 0.2},
        ],
    )

    # Both predictions have confidence 0.8.
    # Accuracy is 0.5, so calibration gap is 0.3.
    assert math.isclose(
        result.expected_calibration_error,
        0.3,
    )


def test_maximum_calibration_error():
    validator = ProbabilityCalibrationValidator(
        num_bins=2,
    )

    result = validator.evaluate(
        actual=["a", "b"],
        probabilities=[
            {"a": 0.8, "b": 0.2},
            {"a": 0.8, "b": 0.2},
        ],
    )

    assert math.isclose(
        result.maximum_calibration_error,
        0.3,
    )


# ============================================================
# Label Handling Tests
# ============================================================


def test_labels_can_vary_between_distributions():
    validator = ProbabilityCalibrationValidator()

    result = validator.evaluate(
        actual=["a", "b"],
        probabilities=[
            {"a": 0.8, "b": 0.2},
            {"b": 0.7, "c": 0.3},
        ],
    )

    assert result.brier_score >= 0.0
    assert result.log_loss >= 0.0


# ============================================================
# Input Validation Tests
# ============================================================


@pytest.mark.parametrize(
    "actual, probabilities",
    [
        ([], []),
        ([], [{"a": 1.0}]),
        (["a"], []),
    ],
)
def test_empty_inputs(actual, probabilities):
    validator = ProbabilityCalibrationValidator()

    with pytest.raises(ValueError):
        validator.evaluate(actual, probabilities)


def test_mismatched_lengths():
    validator = ProbabilityCalibrationValidator()

    with pytest.raises(ValueError):
        validator.evaluate(
            actual=["a", "b"],
            probabilities=[
                {"a": 1.0},
            ],
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
    validator = ProbabilityCalibrationValidator()

    with pytest.raises(TypeError):
        validator.evaluate(
            actual=actual,
            probabilities=[{"a": 1.0}],
        )


@pytest.mark.parametrize(
    "probabilities",
    [
        "invalid",
        b"invalid",
        123,
        {"a": 1.0},
    ],
)
def test_invalid_probabilities_type(probabilities):
    validator = ProbabilityCalibrationValidator()

    with pytest.raises(TypeError):
        validator.evaluate(
            actual=["a"],
            probabilities=probabilities,
        )


def test_distribution_must_be_dictionary():
    validator = ProbabilityCalibrationValidator()

    with pytest.raises(TypeError):
        validator.evaluate(
            actual=["a"],
            probabilities=[
                [0.8, 0.2],
            ],
        )


def test_distribution_cannot_be_empty():
    validator = ProbabilityCalibrationValidator()

    with pytest.raises(ValueError):
        validator.evaluate(
            actual=["a"],
            probabilities=[{}],
        )


def test_actual_label_must_exist_in_distribution():
    validator = ProbabilityCalibrationValidator()

    with pytest.raises(ValueError):
        validator.evaluate(
            actual=["a"],
            probabilities=[
                {"b": 1.0},
            ],
        )


@pytest.mark.parametrize(
    "probability",
    [
        -0.1,
        1.1,
        float("nan"),
        float("inf"),
        -float("inf"),
    ],
)
def test_invalid_probability_values(probability):
    validator = ProbabilityCalibrationValidator()

    with pytest.raises(ValueError):
        validator.evaluate(
            actual=["a"],
            probabilities=[
                {"a": probability, "b": 0.0},
            ],
        )


@pytest.mark.parametrize(
    "probability",
    [
        "0.5",
        None,
        True,
        [],
    ],
)
def test_invalid_probability_types(probability):
    validator = ProbabilityCalibrationValidator()

    with pytest.raises(TypeError):
        validator.evaluate(
            actual=["a"],
            probabilities=[
                {"a": probability, "b": 0.0},
            ],
        )


def test_probabilities_must_sum_to_one():
    validator = ProbabilityCalibrationValidator()

    with pytest.raises(ValueError):
        validator.evaluate(
            actual=["a"],
            probabilities=[
                {"a": 0.6, "b": 0.2},
            ],
        )


def test_probability_sum_within_tolerance():
    validator = ProbabilityCalibrationValidator(
        tolerance=1e-5,
    )

    result = validator.evaluate(
        actual=["a"],
        probabilities=[
            {"a": 0.500001, "b": 0.499999},
        ],
    )

    assert result.brier_score >= 0.0