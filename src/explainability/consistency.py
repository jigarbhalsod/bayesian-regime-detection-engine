from __future__ import annotations

import math
from typing import Any


class ExplanationConsistencyValidator:
    """
    Validates that an explanation summary is internally consistent.
    """

    REQUIRED_FIELDS = (
        "predictions",
        "confidences",
        "feature_drivers",
        "regimes",
        "metadata",
    )

    def validate(self, summary: dict[str, Any]) -> dict[str, Any]:
        """
        Validate an explanation summary and return consistency details.
        """

        self._validate_summary_type(summary)

        issues: list[str] = []

        self._validate_required_fields(summary, issues)
        self._validate_collection_fields(summary, issues)
        self._validate_confidences(summary, issues)
        self._validate_string_collection(
            summary.get("predictions", []),
            "predictions",
            issues,
        )
        self._validate_string_collection(
            summary.get("feature_drivers", []),
            "feature_drivers",
            issues,
        )
        self._validate_string_collection(
            summary.get("regimes", []),
            "regimes",
            issues,
        )
        self._validate_metadata(summary, issues)

        return {
            "is_consistent": len(issues) == 0,
            "issues": issues,
        }

    @staticmethod
    def _validate_summary_type(summary: Any) -> None:
        if summary is None:
            raise ValueError("summary must not be None.")

        if not isinstance(summary, dict):
            raise TypeError("summary must be a dictionary.")

    def _validate_required_fields(
        self,
        summary: dict[str, Any],
        issues: list[str],
    ) -> None:
        for field in self.REQUIRED_FIELDS:
            if field not in summary:
                issues.append(f"Missing required field: {field}.")

    def _validate_collection_fields(
        self,
        summary: dict[str, Any],
        issues: list[str],
    ) -> None:
        for field in (
            "predictions",
            "confidences",
            "feature_drivers",
            "regimes",
        ):
            if field in summary and not isinstance(summary[field], list):
                issues.append(f"{field} must be a list.")

    @staticmethod
    def _validate_confidences(
        summary: dict[str, Any],
        issues: list[str],
    ) -> None:
        confidences = summary.get("confidences", [])

        if not isinstance(confidences, list):
            return

        for confidence in confidences:
            if (
                isinstance(confidence, bool)
                or not isinstance(confidence, (int, float))
                or not math.isfinite(confidence)
                or confidence < 0
                or confidence > 1
            ):
                issues.append(
                    "Confidence values must be finite numbers "
                    "between 0 and 1."
                )

    @staticmethod
    def _validate_string_collection(
        values: Any,
        field_name: str,
        issues: list[str],
    ) -> None:
        if not isinstance(values, list):
            return

        for value in values:
            if (
                not isinstance(value, str)
                or not value.strip()
            ):
                issues.append(
                    f"{field_name} must contain non-empty strings."
                )

    @staticmethod
    def _validate_metadata(
        summary: dict[str, Any],
        issues: list[str],
    ) -> None:
        if (
            "metadata" in summary
            and not isinstance(summary["metadata"], dict)
        ):
            issues.append("metadata must be a dictionary.")