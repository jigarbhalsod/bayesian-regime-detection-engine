from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class WalkForwardWindow:
    """
    Represents one chronological train/test validation window.

    The indices follow Python slicing semantics:
    - train: [train_start:train_end]
    - test:  [test_start:test_end]
    """

    train_start: int
    train_end: int
    test_start: int
    test_end: int

    @property
    def train_size(self) -> int:
        return self.train_end - self.train_start

    @property
    def test_size(self) -> int:
        return self.test_end - self.test_start


class WalkForwardValidator:
    """
    Generate chronological walk-forward validation windows.

    Supports:

    - expanding training windows
    - rolling training windows
    - configurable test size
    - configurable step size

    Future observations are never included in the corresponding
    training window.
    """

    def __init__(
        self,
        train_size: int,
        test_size: int,
        step_size: int | None = None,
        expanding: bool = True,
    ) -> None:
        self.train_size = self._validate_positive_integer(
            train_size,
            "train_size",
        )
        self.test_size = self._validate_positive_integer(
            test_size,
            "test_size",
        )

        if step_size is None:
            step_size = test_size

        self.step_size = self._validate_positive_integer(
            step_size,
            "step_size",
        )

        if not isinstance(expanding, bool):
            raise TypeError("expanding must be a boolean")

        self.expanding = expanding

    @staticmethod
    def _validate_positive_integer(
        value: int,
        name: str,
    ) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")

        if value <= 0:
            raise ValueError(f"{name} must be greater than 0")

        return value

    def generate_windows(
        self,
        data_size: int,
    ) -> Iterator[WalkForwardWindow]:
        """
        Generate valid chronological walk-forward windows.

        Args:
            data_size:
                Total number of sequential observations.

        Yields:
            WalkForwardWindow instances.
        """
        data_size = self._validate_positive_integer(
            data_size,
            "data_size",
        )

        if data_size < self.train_size + self.test_size:
            raise ValueError(
                "data_size must be at least "
                "train_size + test_size"
            )

        first_test_start = self.train_size

        test_start = first_test_start

        while test_start + self.test_size <= data_size:
            if self.expanding:
                train_start = 0
            else:
                train_start = test_start - self.train_size

            train_end = test_start
            test_end = test_start + self.test_size

            yield WalkForwardWindow(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )

            test_start += self.step_size

    def get_windows(
        self,
        data_size: int,
    ) -> list[WalkForwardWindow]:
        """
        Return all generated walk-forward windows as a list.
        """
        return list(self.generate_windows(data_size))