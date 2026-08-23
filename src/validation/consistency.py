from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .base import ValidationResult


@dataclass(frozen=True)
class ConsistencyResult:
    """Summary of consistency across validator results."""

    validator_count: int
    valid_count: int
    invalid_count: int
    is_consistent: bool
    majority_valid: bool
    inconsistent_validators: tuple[str, ...]


class CrossValidatorConsistency:
    """Analyze agreement across multiple ValidationResult objects."""

    def analyze(
        self,
        results: Sequence[ValidationResult],
    ) -> ConsistencyResult:
        self._validate_results(results)

        valid_count = sum(
            result.is_valid
            for result in results
        )
        invalid_count = len(results) - valid_count

        is_consistent = (
            valid_count == 0
            or invalid_count == 0
        )

        majority_valid = valid_count > invalid_count

        if is_consistent:
            inconsistent_validators: tuple[str, ...] = ()
        else:
            majority_state = majority_valid

            inconsistent_validators = tuple(
                result.validator_name
                for result in results
                if result.is_valid != majority_state
            )

        return ConsistencyResult(
            validator_count=len(results),
            valid_count=valid_count,
            invalid_count=invalid_count,
            is_consistent=is_consistent,
            majority_valid=majority_valid,
            inconsistent_validators=inconsistent_validators,
        )

    @staticmethod
    def _validate_results(
        results: Sequence[ValidationResult],
    ) -> None:
        if isinstance(results, (str, bytes)):
            raise TypeError(
                "results must be a sequence, not a string or bytes"
            )

        if not isinstance(results, Sequence):
            raise TypeError(
                "results must be a sequence"
            )

        if len(results) == 0:
            raise ValueError(
                "results cannot be empty"
            )

        for result in results:
            if not isinstance(result, ValidationResult):
                raise TypeError(
                    "results must contain ValidationResult objects"
                )