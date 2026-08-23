import pytest

from src.validation.base import ValidationResult
from src.validation.consistency import (
    ConsistencyResult,
    CrossValidatorConsistency,
)


def make_result(
    name: str,
    is_valid: bool,
) -> ValidationResult:
    return ValidationResult(
        validator_name=name,
        is_valid=is_valid,
    )


# ============================================================
# ConsistencyResult Tests
# ============================================================


def test_consistency_result_values():
    result = ConsistencyResult(
        validator_count=3,
        valid_count=2,
        invalid_count=1,
        is_consistent=False,
        majority_valid=True,
        inconsistent_validators=("third",),
    )

    assert result.validator_count == 3
    assert result.valid_count == 2
    assert result.invalid_count == 1
    assert result.is_consistent is False
    assert result.majority_valid is True
    assert result.inconsistent_validators == ("third",)


def test_consistency_result_is_immutable():
    result = ConsistencyResult(
        validator_count=1,
        valid_count=1,
        invalid_count=0,
        is_consistent=True,
        majority_valid=True,
        inconsistent_validators=(),
    )

    with pytest.raises(Exception):
        result.valid_count = 10


# ============================================================
# Fully Consistent Tests
# ============================================================


def test_all_valid_is_consistent():
    results = [
        make_result("first", True),
        make_result("second", True),
        make_result("third", True),
    ]

    result = CrossValidatorConsistency().analyze(results)

    assert isinstance(result, ConsistencyResult)
    assert result.validator_count == 3
    assert result.valid_count == 3
    assert result.invalid_count == 0
    assert result.is_consistent is True
    assert result.majority_valid is True
    assert result.inconsistent_validators == ()


def test_all_invalid_is_consistent():
    results = [
        make_result("first", False),
        make_result("second", False),
        make_result("third", False),
    ]

    result = CrossValidatorConsistency().analyze(results)

    assert result.validator_count == 3
    assert result.valid_count == 0
    assert result.invalid_count == 3
    assert result.is_consistent is True
    assert result.majority_valid is False
    assert result.inconsistent_validators == ()


def test_single_valid_result_is_consistent():
    result = CrossValidatorConsistency().analyze(
        [make_result("only", True)]
    )

    assert result.is_consistent is True
    assert result.majority_valid is True
    assert result.inconsistent_validators == ()


def test_single_invalid_result_is_consistent():
    result = CrossValidatorConsistency().analyze(
        [make_result("only", False)]
    )

    assert result.is_consistent is True
    assert result.majority_valid is False
    assert result.inconsistent_validators == ()


# ============================================================
# Inconsistency Tests
# ============================================================


def test_one_invalid_against_valid_majority():
    results = [
        make_result("first", True),
        make_result("second", True),
        make_result("third", False),
    ]

    result = CrossValidatorConsistency().analyze(results)

    assert result.is_consistent is False
    assert result.majority_valid is True
    assert result.inconsistent_validators == ("third",)


def test_one_valid_against_invalid_majority():
    results = [
        make_result("first", False),
        make_result("second", False),
        make_result("third", True),
    ]

    result = CrossValidatorConsistency().analyze(results)

    assert result.is_consistent is False
    assert result.majority_valid is False
    assert result.inconsistent_validators == ("third",)


def test_multiple_inconsistent_validators():
    results = [
        make_result("first", True),
        make_result("second", True),
        make_result("third", False),
        make_result("fourth", False),
        make_result("fifth", True),
    ]

    result = CrossValidatorConsistency().analyze(results)

    assert result.is_consistent is False
    assert result.majority_valid is True
    assert result.inconsistent_validators == (
        "third",
        "fourth",
    )


def test_inconsistent_order_is_preserved():
    results = [
        make_result("valid_1", True),
        make_result("invalid_1", False),
        make_result("invalid_2", False),
        make_result("valid_2", True),
        make_result("valid_3", True),
    ]

    result = CrossValidatorConsistency().analyze(results)

    assert result.inconsistent_validators == (
        "invalid_1",
        "invalid_2",
    )


# ============================================================
# Tie Tests
# ============================================================


def test_even_tie_is_not_consistent():
    results = [
        make_result("first", True),
        make_result("second", False),
    ]

    result = CrossValidatorConsistency().analyze(results)

    assert result.is_consistent is False
    assert result.valid_count == 1
    assert result.invalid_count == 1
    assert result.majority_valid is False


def test_tie_marks_valid_results_as_inconsistent():
    results = [
        make_result("first", True),
        make_result("second", False),
    ]

    result = CrossValidatorConsistency().analyze(results)

    assert result.inconsistent_validators == ("first",)


def test_larger_even_tie():
    results = [
        make_result("first", True),
        make_result("second", False),
        make_result("third", True),
        make_result("fourth", False),
    ]

    result = CrossValidatorConsistency().analyze(results)

    assert result.is_consistent is False
    assert result.majority_valid is False
    assert result.inconsistent_validators == (
        "first",
        "third",
    )


# ============================================================
# Input Validation Tests
# ============================================================


def test_empty_results():
    with pytest.raises(ValueError):
        CrossValidatorConsistency().analyze([])


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
        CrossValidatorConsistency().analyze(results)


def test_invalid_result_object():
    with pytest.raises(TypeError):
        CrossValidatorConsistency().analyze(
            ["invalid"]
        )


def test_mixed_valid_and_invalid_object_types():
    results = [
        make_result("valid", True),
        "invalid",
    ]

    with pytest.raises(TypeError):
        CrossValidatorConsistency().analyze(results)


def test_validate_results_returns_none():
    results = [
        make_result("first", True),
    ]

    value = CrossValidatorConsistency._validate_results(
        results
    )

    assert value is None