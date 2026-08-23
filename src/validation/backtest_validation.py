from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class BacktestFold:
    """
    A single chronological backtest fold.
    """

    train_start: int
    train_end: int
    test_start: int
    test_end: int
    actual: tuple[str, ...]
    predicted: tuple[str, ...]

    def __post_init__(self) -> None:
        self._validate_index("train_start", self.train_start)
        self._validate_index("train_end", self.train_end)
        self._validate_index("test_start", self.test_start)
        self._validate_index("test_end", self.test_end)

        if self.train_start >= self.train_end:
            raise ValueError(
                "train_start must be less than train_end"
            )

        if self.test_start >= self.test_end:
            raise ValueError(
                "test_start must be less than test_end"
            )

        if self.train_end > self.test_start:
            raise ValueError(
                "training data must not overlap future test data"
            )

        self._validate_labels("actual", self.actual)
        self._validate_labels("predicted", self.predicted)

        if len(self.actual) != len(self.predicted):
            raise ValueError(
                "actual and predicted must have equal length"
            )

        expected_test_size = self.test_end - self.test_start

        if len(self.actual) != expected_test_size:
            raise ValueError(
                "prediction length must match test window size"
            )

    @staticmethod
    def _validate_index(
        name: str,
        value: int,
    ) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(
                f"{name} must be an integer"
            )

        if value < 0:
            raise ValueError(
                f"{name} must be non-negative"
            )

    @staticmethod
    def _validate_labels(
        name: str,
        values: tuple[str, ...],
    ) -> None:
        if not isinstance(values, tuple):
            raise TypeError(
                f"{name} must be a tuple"
            )

        if len(values) == 0:
            raise ValueError(
                f"{name} cannot be empty"
            )

        for value in values:
            if not isinstance(value, str):
                raise TypeError(
                    f"{name} labels must be strings"
                )

            if not value.strip():
                raise ValueError(
                    f"{name} labels cannot be empty"
                )


@dataclass(frozen=True)
class BacktestValidationResult:
    """
    Summary of complete backtest validation.
    """

    fold_count: int
    valid_fold_count: int
    invalid_fold_count: int
    total_predictions: int
    overall_accuracy: float
    is_valid: bool


class BacktestValidator:
    """
    Validate chronological backtest folds and aggregate accuracy.
    """

    def validate(
        self,
        folds: Sequence[BacktestFold],
    ) -> BacktestValidationResult:
        self._validate_folds(folds)

        total_predictions = sum(
            len(fold.actual)
            for fold in folds
        )

        correct_predictions = sum(
            sum(
                actual == predicted
                for actual, predicted in zip(
                    fold.actual,
                    fold.predicted,
                )
            )
            for fold in folds
        )

        overall_accuracy = (
            correct_predictions / total_predictions
        )

        return BacktestValidationResult(
            fold_count=len(folds),
            valid_fold_count=len(folds),
            invalid_fold_count=0,
            total_predictions=total_predictions,
            overall_accuracy=overall_accuracy,
            is_valid=True,
        )

    @staticmethod
    def _validate_folds(
        folds: Sequence[BacktestFold],
    ) -> None:
        if isinstance(folds, (str, bytes)):
            raise TypeError(
                "folds must be a sequence, not a string or bytes"
            )

        if not isinstance(folds, Sequence):
            raise TypeError(
                "folds must be a sequence"
            )

        if len(folds) == 0:
            raise ValueError(
                "folds cannot be empty"
            )

        for fold in folds:
            if not isinstance(fold, BacktestFold):
                raise TypeError(
                    "folds must contain BacktestFold objects"
                )

        for previous, current in zip(
            folds,
            folds[1:],
        ):
            if previous.test_end > current.test_start:
                raise ValueError(
                    "backtest test windows must not overlap"
                )

            if current.train_end > current.test_start:
                raise ValueError(
                    "fold contains future data leakage"
                )