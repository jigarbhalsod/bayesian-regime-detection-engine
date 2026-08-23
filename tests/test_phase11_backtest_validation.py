import pytest

from src.validation.backtest_validation import (
    BacktestFold,
    BacktestValidationResult,
    BacktestValidator,
)


# ============================================================
# BacktestFold Tests
# ============================================================


def test_valid_fold():
    fold = BacktestFold(
        train_start=0,
        train_end=3,
        test_start=3,
        test_end=5,
        actual=("a", "b"),
        predicted=("a", "b"),
    )

    assert fold.train_start == 0
    assert fold.train_end == 3
    assert fold.test_start == 3
    assert fold.test_end == 5


def test_fold_is_immutable():
    fold = BacktestFold(
        train_start=0,
        train_end=2,
        test_start=2,
        test_end=3,
        actual=("a",),
        predicted=("a",),
    )

    with pytest.raises(Exception):
        fold.train_end = 10


@pytest.mark.parametrize(
    "field_name",
    [
        "train_start",
        "train_end",
        "test_start",
        "test_end",
    ],
)
def test_boolean_index_is_invalid(field_name):
    values = {
        "train_start": 0,
        "train_end": 2,
        "test_start": 2,
        "test_end": 3,
    }
    values[field_name] = True

    with pytest.raises(TypeError):
        BacktestFold(
            actual=("a",),
            predicted=("a",),
            **values,
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "train_start",
        "train_end",
        "test_start",
        "test_end",
    ],
)
def test_non_integer_index_is_invalid(field_name):
    values = {
        "train_start": 0,
        "train_end": 2,
        "test_start": 2,
        "test_end": 3,
    }
    values[field_name] = 1.5

    with pytest.raises(TypeError):
        BacktestFold(
            actual=("a",),
            predicted=("a",),
            **values,
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "train_start",
        "train_end",
        "test_start",
        "test_end",
    ],
)
def test_negative_index_is_invalid(field_name):
    values = {
        "train_start": 0,
        "train_end": 2,
        "test_start": 2,
        "test_end": 3,
    }
    values[field_name] = -1

    with pytest.raises(ValueError):
        BacktestFold(
            actual=("a",),
            predicted=("a",),
            **values,
        )


def test_invalid_training_window():
    with pytest.raises(ValueError):
        BacktestFold(
            train_start=2,
            train_end=2,
            test_start=2,
            test_end=3,
            actual=("a",),
            predicted=("a",),
        )


def test_invalid_test_window():
    with pytest.raises(ValueError):
        BacktestFold(
            train_start=0,
            train_end=2,
            test_start=3,
            test_end=3,
            actual=("a",),
            predicted=("a",),
        )


def test_training_must_not_overlap_test():
    with pytest.raises(ValueError):
        BacktestFold(
            train_start=0,
            train_end=4,
            test_start=3,
            test_end=5,
            actual=("a", "b"),
            predicted=("a", "b"),
        )


def test_actual_must_be_tuple():
    with pytest.raises(TypeError):
        BacktestFold(
            train_start=0,
            train_end=2,
            test_start=2,
            test_end=3,
            actual=["a"],
            predicted=("a",),
        )


def test_predicted_must_be_tuple():
    with pytest.raises(TypeError):
        BacktestFold(
            train_start=0,
            train_end=2,
            test_start=2,
            test_end=3,
            actual=("a",),
            predicted=["a"],
        )


def test_empty_actual_is_invalid():
    with pytest.raises(ValueError):
        BacktestFold(
            train_start=0,
            train_end=2,
            test_start=2,
            test_end=3,
            actual=(),
            predicted=("a",),
        )


def test_empty_predicted_is_invalid():
    with pytest.raises(ValueError):
        BacktestFold(
            train_start=0,
            train_end=2,
            test_start=2,
            test_end=3,
            actual=("a",),
            predicted=(),
        )


def test_invalid_actual_label_type():
    with pytest.raises(TypeError):
        BacktestFold(
            train_start=0,
            train_end=2,
            test_start=2,
            test_end=3,
            actual=(123,),
            predicted=("a",),
        )


def test_invalid_predicted_label_type():
    with pytest.raises(TypeError):
        BacktestFold(
            train_start=0,
            train_end=2,
            test_start=2,
            test_end=3,
            actual=("a",),
            predicted=(123,),
        )


@pytest.mark.parametrize(
    "actual",
    [
        ("",),
        ("   ",),
    ],
)
def test_empty_actual_label(actual):
    with pytest.raises(ValueError):
        BacktestFold(
            train_start=0,
            train_end=2,
            test_start=2,
            test_end=3,
            actual=actual,
            predicted=("a",),
        )


@pytest.mark.parametrize(
    "predicted",
    [
        ("",),
        ("\t",),
    ],
)
def test_empty_predicted_label(predicted):
    with pytest.raises(ValueError):
        BacktestFold(
            train_start=0,
            train_end=2,
            test_start=2,
            test_end=3,
            actual=("a",),
            predicted=predicted,
        )


def test_mismatched_prediction_lengths():
    with pytest.raises(ValueError):
        BacktestFold(
            train_start=0,
            train_end=2,
            test_start=2,
            test_end=4,
            actual=("a", "b"),
            predicted=("a",),
        )


def test_prediction_length_matches_test_window():
    with pytest.raises(ValueError):
        BacktestFold(
            train_start=0,
            train_end=2,
            test_start=2,
            test_end=5,
            actual=("a", "b"),
            predicted=("a", "b"),
        )


# ============================================================
# Validator Tests
# ============================================================


def make_fold(
    start: int,
    actual: tuple[str, ...],
    predicted: tuple[str, ...],
) -> BacktestFold:
    test_size = len(actual)

    return BacktestFold(
        train_start=0,
        train_end=start,
        test_start=start,
        test_end=start + test_size,
        actual=actual,
        predicted=predicted,
    )


def test_validate_returns_result():
    fold = make_fold(
        2,
        ("a", "b"),
        ("a", "b"),
    )

    result = BacktestValidator().validate([fold])

    assert isinstance(result, BacktestValidationResult)


def test_single_fold_perfect_accuracy():
    fold = make_fold(
        2,
        ("a", "b"),
        ("a", "b"),
    )

    result = BacktestValidator().validate([fold])

    assert result.fold_count == 1
    assert result.valid_fold_count == 1
    assert result.invalid_fold_count == 0
    assert result.total_predictions == 2
    assert result.overall_accuracy == 1.0
    assert result.is_valid is True


def test_single_fold_partial_accuracy():
    fold = make_fold(
        2,
        ("a", "b"),
        ("a", "a"),
    )

    result = BacktestValidator().validate([fold])

    assert result.total_predictions == 2
    assert result.overall_accuracy == 0.5


def test_multiple_folds_aggregate_accuracy():
    first = make_fold(
        2,
        ("a", "b"),
        ("a", "b"),
    )

    second = BacktestFold(
        train_start=0,
        train_end=4,
        test_start=4,
        test_end=6,
        actual=("a", "b"),
        predicted=("a", "a"),
    )

    result = BacktestValidator().validate(
        [first, second]
    )

    assert result.fold_count == 2
    assert result.total_predictions == 4
    assert result.overall_accuracy == 0.75


def test_test_windows_must_not_overlap():
    first = BacktestFold(
        train_start=0,
        train_end=2,
        test_start=2,
        test_end=5,
        actual=("a", "b", "c"),
        predicted=("a", "b", "c"),
    )

    second = BacktestFold(
        train_start=0,
        train_end=3,
        test_start=3,
        test_end=5,
        actual=("a", "b"),
        predicted=("a", "b"),
    )

    with pytest.raises(ValueError):
        BacktestValidator().validate(
            [first, second]
        )


def test_empty_folds():
    with pytest.raises(ValueError):
        BacktestValidator().validate([])


@pytest.mark.parametrize(
    "folds",
    [
        "invalid",
        b"invalid",
        123,
        {"fold": "invalid"},
    ],
)
def test_invalid_folds_type(folds):
    with pytest.raises(TypeError):
        BacktestValidator().validate(folds)


def test_invalid_fold_object():
    with pytest.raises(TypeError):
        BacktestValidator().validate(
            ["invalid"]
        )


def test_validate_folds_returns_none():
    fold = make_fold(
        2,
        ("a",),
        ("a",),
    )

    result = BacktestValidator._validate_folds([fold])

    assert result is None