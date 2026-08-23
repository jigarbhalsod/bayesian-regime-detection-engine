from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class RegimeTransitionResult:
    """
    Summary of actual and predicted regime transitions.
    """

    actual_transition_count: int
    predicted_transition_count: int
    matched_transition_count: int
    missed_transition_count: int
    false_transition_count: int
    transition_precision: float
    transition_recall: float
    transition_f1: float
    mean_detection_delay: float | None


class RegimeTransitionValidator:
    """
    Validate predicted regime transitions against actual regime transitions.

    A transition occurs whenever the regime label changes between two
    consecutive observations. Transitions are matched by their position.
    """

    def validate(
        self,
        actual: Sequence[str],
        predicted: Sequence[str],
    ) -> RegimeTransitionResult:
        self._validate_sequences(actual, predicted)

        actual_transitions = self._find_transitions(actual)
        predicted_transitions = self._find_transitions(predicted)

        matched_pairs = self._match_transitions(
            actual_transitions,
            predicted_transitions,
        )

        matched_transition_count = len(matched_pairs)

        missed_transition_count = (
            len(actual_transitions)
            - matched_transition_count
        )

        false_transition_count = (
            len(predicted_transitions)
            - matched_transition_count
        )

        transition_precision = self._safe_divide(
            matched_transition_count,
            len(predicted_transitions),
        )

        transition_recall = self._safe_divide(
            matched_transition_count,
            len(actual_transitions),
        )

        transition_f1 = self._calculate_f1(
            transition_precision,
            transition_recall,
        )

        mean_detection_delay = (
            self._mean_detection_delay(matched_pairs)
        )

        return RegimeTransitionResult(
            actual_transition_count=len(actual_transitions),
            predicted_transition_count=len(predicted_transitions),
            matched_transition_count=matched_transition_count,
            missed_transition_count=missed_transition_count,
            false_transition_count=false_transition_count,
            transition_precision=transition_precision,
            transition_recall=transition_recall,
            transition_f1=transition_f1,
            mean_detection_delay=mean_detection_delay,
        )

    @staticmethod
    def _validate_sequences(
        actual: Sequence[str],
        predicted: Sequence[str],
    ) -> None:
        if isinstance(actual, (str, bytes)):
            raise TypeError(
                "actual must be a sequence, not a string or bytes"
            )

        if isinstance(predicted, (str, bytes)):
            raise TypeError(
                "predicted must be a sequence, not a string or bytes"
            )

        if not isinstance(actual, Sequence):
            raise TypeError("actual must be a sequence")

        if not isinstance(predicted, Sequence):
            raise TypeError("predicted must be a sequence")

        if len(actual) == 0:
            raise ValueError("actual cannot be empty")

        if len(predicted) == 0:
            raise ValueError("predicted cannot be empty")

        if len(actual) != len(predicted):
            raise ValueError(
                "actual and predicted must have equal length"
            )

        for name, sequence in (
            ("actual", actual),
            ("predicted", predicted),
        ):
            for value in sequence:
                if not isinstance(value, str):
                    raise TypeError(
                        f"{name} regime labels must be strings"
                    )

                if not value.strip():
                    raise ValueError(
                        f"{name} regime labels cannot be empty"
                    )

    @staticmethod
    def _find_transitions(
        regimes: Sequence[str],
    ) -> list[int]:
        return [
            index
            for index in range(1, len(regimes))
            if regimes[index] != regimes[index - 1]
        ]

    @staticmethod
    def _match_transitions(
        actual_transitions: Sequence[int],
        predicted_transitions: Sequence[int],
    ) -> list[tuple[int, int]]:
        """
        Match transitions occurring at exactly the same position.

        Returns pairs of:
        (actual_transition_index, predicted_transition_index)
        """
        predicted_set = set(predicted_transitions)

        return [
            (index, index)
            for index in actual_transitions
            if index in predicted_set
        ]

    @staticmethod
    def _safe_divide(
        numerator: int,
        denominator: int,
    ) -> float:
        if denominator == 0:
            return 0.0

        return numerator / denominator

    @staticmethod
    def _calculate_f1(
        precision: float,
        recall: float,
    ) -> float:
        denominator = precision + recall

        if denominator == 0.0:
            return 0.0

        return (
            2.0
            * precision
            * recall
            / denominator
        )

    @staticmethod
    def _mean_detection_delay(
        matched_pairs: Sequence[tuple[int, int]],
    ) -> float | None:
        if not matched_pairs:
            return None

        delays = [
            predicted_index - actual_index
            for actual_index, predicted_index in matched_pairs
        ]

        return sum(delays) / len(delays)