from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .base import ValidationResult


@dataclass(frozen=True)
class ValidationSummary:
    """Structured summary of multiple validation results."""

    total_validators: int
    passed_validators: int
    failed_validators: int
    overall_valid: bool
    failed_validator_names: tuple[str, ...]
    aggregated_metrics: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        """Export the summary as a structured dictionary."""
        return {
            "total_validators": self.total_validators,
            "passed_validators": self.passed_validators,
            "failed_validators": self.failed_validators,
            "overall_valid": self.overall_valid,
            "failed_validator_names": self.failed_validator_names,
            "aggregated_metrics": self.aggregated_metrics,
        }


class ValidationReporter:
    """Create structured summaries from validation results."""

    def summarize(
        self,
        results: Sequence[ValidationResult],
    ) -> ValidationSummary:
        self._validate_results(results)

        total_validators = len(results)
        passed_validators = sum(
            result.is_valid
            for result in results
        )
        failed_validators = (
            total_validators - passed_validators
        )

        failed_validator_names = tuple(
            result.validator_name
            for result in results
            if not result.is_valid
        )

        aggregated_metrics = self._aggregate_metrics(results)

        return ValidationSummary(
            total_validators=total_validators,
            passed_validators=passed_validators,
            failed_validators=failed_validators,
            overall_valid=(failed_validators == 0),
            failed_validator_names=failed_validator_names,
            aggregated_metrics=aggregated_metrics,
        )

    @staticmethod
    def _aggregate_metrics(
        results: Sequence[ValidationResult],
    ) -> dict[str, float]:
        metric_values: dict[str, list[float]] = {}

        for result in results:
            for name, value in result.metrics.items():
                metric_values.setdefault(name, []).append(value)

        return {
            name: sum(values) / len(values)
            for name, values in metric_values.items()
        }

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