from __future__ import annotations

import math
from collections import Counter
from typing import Any, Sequence

from src.ensemble.weighting import EnsembleWeighting


class EnsembleVoting:
    """
    Voting utilities for combining categorical model predictions.

    Supports:
    - majority voting
    - weighted voting
    - vote counts
    - vote shares
    """

    @staticmethod
    def majority_vote(
        predictions: Sequence[Any],
    ) -> Any:
        """
        Return the prediction with the highest number of votes.

        Ties are resolved deterministically by selecting the first
        prediction that reaches the highest vote count.
        """
        validated = EnsembleVoting._validate_predictions(
            predictions
        )

        counts = Counter(validated)
        max_count = max(counts.values())

        for prediction in validated:
            if counts[prediction] == max_count:
                return prediction

        raise RuntimeError(
            "Unable to determine majority vote."
        )

    @staticmethod
    def weighted_vote(
        predictions: Sequence[Any],
        weights: Sequence[Any] | None = None,
    ) -> Any:
        """
        Return the prediction with the highest total voting weight.

        Equal weighting is used when weights are not supplied.
        Ties are resolved by selecting the first prediction appearing
        in the original prediction sequence.
        """
        validated = EnsembleVoting._validate_predictions(
            predictions
        )

        if weights is None:
            normalized_weights = (
                EnsembleWeighting.equal_weights(
                    len(validated)
                )
            )
        else:
            normalized_weights = (
                EnsembleWeighting.normalize_weights(
                    weights,
                    n_models=len(validated),
                )
            )

        vote_weights: dict[Any, float] = {}

        for prediction, weight in zip(
            validated,
            normalized_weights,
        ):
            vote_weights[prediction] = (
                vote_weights.get(prediction, 0.0)
                + weight
            )

        max_weight = max(vote_weights.values())

        for prediction in validated:
            if (
                math.isclose(
                    vote_weights[prediction],
                    max_weight,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            ):
                return prediction

        raise RuntimeError(
            "Unable to determine weighted vote."
        )

    @staticmethod
    def vote_counts(
        predictions: Sequence[Any],
    ) -> dict[Any, int]:
        """
        Return the number of votes received by each prediction.

        Dictionary order follows first appearance order.
        """
        validated = EnsembleVoting._validate_predictions(
            predictions
        )

        counts: dict[Any, int] = {}

        for prediction in validated:
            counts[prediction] = (
                counts.get(prediction, 0)
                + 1
            )

        return counts

    @staticmethod
    def vote_shares(
        predictions: Sequence[Any],
    ) -> dict[Any, float]:
        """
        Return the fraction of total votes received by each prediction.
        """
        counts = EnsembleVoting.vote_counts(
            predictions
        )

        total = sum(counts.values())

        return {
            prediction: count / total
            for prediction, count in counts.items()
        }

    @staticmethod
    def weighted_vote_shares(
        predictions: Sequence[Any],
        weights: Sequence[Any] | None = None,
    ) -> dict[Any, float]:
        """
        Return normalized total voting weight for each prediction.
        """
        validated = EnsembleVoting._validate_predictions(
            predictions
        )

        if weights is None:
            normalized_weights = (
                EnsembleWeighting.equal_weights(
                    len(validated)
                )
            )
        else:
            normalized_weights = (
                EnsembleWeighting.normalize_weights(
                    weights,
                    n_models=len(validated),
                )
            )

        shares: dict[Any, float] = {}

        for prediction, weight in zip(
            validated,
            normalized_weights,
        ):
            shares[prediction] = (
                shares.get(prediction, 0.0)
                + weight
            )

        return shares

    @staticmethod
    def _validate_predictions(
        predictions: Sequence[Any],
    ) -> tuple[Any, ...]:
        """
        Validate predictions.

        Predictions must be a non-empty sequence of hashable values.
        Strings and bytes are rejected as the outer container because
        they represent a single scalar prediction, not a collection.
        """
        if isinstance(
            predictions,
            (str, bytes),
        ) or not isinstance(
            predictions,
            Sequence,
        ):
            raise TypeError(
                "predictions must be a sequence."
            )

        if len(predictions) == 0:
            raise ValueError(
                "predictions cannot be empty."
            )

        validated = tuple(predictions)

        for index, prediction in enumerate(validated):
            if prediction is None:
                raise TypeError(
                    f"prediction at index {index} cannot be None."
                )

            try:
                hash(prediction)
            except TypeError as exc:
                raise TypeError(
                    f"prediction at index {index} "
                    "must be hashable."
                ) from exc

        return validated