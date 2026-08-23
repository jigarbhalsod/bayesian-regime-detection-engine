from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Sequence, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class TimeSeriesSplit(Generic[T]):
    """
    Result of a chronological time-series split.

    Data is divided into train, validation, and test segments while
    preserving the original temporal order.
    """

    train: Sequence[T]
    validation: Sequence[T]
    test: Sequence[T]


class TimeSeriesSplitter:
    """
    Chronologically split sequential data into train, validation,
    and test segments without shuffling.
    """

    def __init__(
        self,
        train_ratio: float = 0.70,
        validation_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> None:
        self.train_ratio = self._validate_ratio(
            train_ratio,
            "train_ratio",
        )
        self.validation_ratio = self._validate_ratio(
            validation_ratio,
            "validation_ratio",
        )
        self.test_ratio = self._validate_ratio(
            test_ratio,
            "test_ratio",
        )

        total_ratio = (
            self.train_ratio
            + self.validation_ratio
            + self.test_ratio
        )

        if abs(total_ratio - 1.0) > 1e-9:
            raise ValueError(
                "train_ratio, validation_ratio, and test_ratio "
                "must sum to 1.0"
            )

    @staticmethod
    def _validate_ratio(value: float, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a numeric value")

        value = float(value)

        if value <= 0.0 or value >= 1.0:
            raise ValueError(
                f"{name} must be greater than 0 and less than 1"
            )

        return value

    def split(self, data: Sequence[T]) -> TimeSeriesSplit[T]:
        """
        Split data chronologically.

        The earliest observations are assigned to training,
        followed by validation, followed by test observations.
        """
        if isinstance(data, (str, bytes)):
            raise TypeError(
                "data must be a sequence, not a string or bytes"
            )

        if not isinstance(data, Sequence):
            raise TypeError("data must be a sequence")

        total_size = len(data)

        if total_size < 3:
            raise ValueError(
                "data must contain at least 3 observations"
            )

        train_end = int(total_size * self.train_ratio)
        validation_end = train_end + int(
            total_size * self.validation_ratio
        )

        if train_end == 0:
            raise ValueError(
                "train split cannot be empty"
            )

        if validation_end <= train_end:
            raise ValueError(
                "validation split cannot be empty"
            )

        if validation_end >= total_size:
            raise ValueError(
                "test split cannot be empty"
            )

        return TimeSeriesSplit(
            train=data[:train_end],
            validation=data[train_end:validation_end],
            test=data[validation_end:],
        )