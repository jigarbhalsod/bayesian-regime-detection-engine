from collections.abc import Iterable

from .result import RegimeResult


class RegimeValidator:
    """Validate regime labels produced by regime detectors."""

    DEFAULT_ALLOWED_REGIMES = {
        "risk_on",
        "risk_off",
        "transitional",
        "unknown",
    }

    def __init__(self, allowed_regimes=None):
        self.allowed_regimes = (
            set(allowed_regimes)
            if allowed_regimes is not None
            else set(self.DEFAULT_ALLOWED_REGIMES)
        )

    @staticmethod
    def _normalize_label(label):
        """Normalize a regime label for validation."""
        if not isinstance(label, str):
            return None

        return label.strip().lower()

    def invalid_labels(self, result):
        """Return all invalid regime labels from a RegimeResult."""
        if not isinstance(result, RegimeResult):
            return []

        labels = result.regime_labels

        if isinstance(labels, str) or not isinstance(labels, Iterable):
            return []

        invalid = []

        for label in labels:
            normalized = self._normalize_label(label)

            if (
                normalized is None
                or normalized not in self.allowed_regimes
            ):
                invalid.append(label)

        return invalid

    def validate(self, result):
        """Return True when all regime labels are valid."""
        if not isinstance(result, RegimeResult):
            return False

        labels = result.regime_labels

        if isinstance(labels, str) or not isinstance(labels, Iterable):
            return False

        return len(self.invalid_labels(result)) == 0