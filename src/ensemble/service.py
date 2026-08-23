from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from src.ensemble.aggregation import EnsembleAggregator
from src.ensemble.config import EnsembleConfig
from src.ensemble.result import EnsembleResult
from src.ensemble.voting import EnsembleVoting
from src.ensemble.weighting import EnsembleWeighting


class EnsembleService:
    """
    High-level service for combining predictions from multiple models.
    """

    def __init__(
        self,
        config: EnsembleConfig | None = None,
    ) -> None:
        self.config = (
            config
            if config is not None
            else EnsembleConfig()
        )

    def aggregate(
        self,
        predictions: Sequence[Any],
        weights: Sequence[Any] | None = None,
    ) -> EnsembleResult:
        values = self._validate_scalar_predictions(
            predictions
        )

        resolved_weights = self._resolve_weights(
            n_models=len(values),
            weights=weights,
        )

        prediction = EnsembleAggregator.weighted_mean(
            values=values,
            weights=resolved_weights,
        )

        confidence = self._calculate_confidence(
            resolved_weights
        )

        uncertainty = 1.0 - confidence

        return EnsembleResult(
            prediction=prediction,
            confidence=confidence,
            uncertainty=uncertainty,
        )

    def aggregate_vectors(
        self,
        predictions: Sequence[Sequence[Any]],
        weights: Sequence[Any] | None = None,
    ) -> EnsembleResult:
        vectors = self._validate_vector_predictions(
            predictions
        )

        resolved_weights = self._resolve_weights(
            n_models=len(vectors),
            weights=weights,
        )

        prediction = EnsembleAggregator.aggregate_vectors(
            vectors=vectors,
            weights=resolved_weights,
        )

        confidence = self._calculate_confidence(
            resolved_weights
        )

        uncertainty = 1.0 - confidence

        return EnsembleResult(
            prediction=prediction,
            confidence=confidence,
            uncertainty=uncertainty,
        )

    def vote(
        self,
        predictions: Sequence[Any],
        weights: Sequence[Any] | None = None,
    ) -> EnsembleResult:
        validated = self._validate_vote_predictions(
            predictions
        )

        if weights is None:
            prediction = EnsembleVoting.majority_vote(
                validated
            )

            resolved_weights = self._resolve_weights(
                n_models=len(validated),
                weights=None,
            )
        else:
            resolved_weights = self._resolve_weights(
                n_models=len(validated),
                weights=weights,
            )

            prediction = EnsembleVoting.weighted_vote(
                predictions=validated,
                weights=resolved_weights,
            )

        confidence = self._calculate_confidence(
            resolved_weights
        )

        uncertainty = 1.0 - confidence

        return EnsembleResult(
            prediction=prediction,
            confidence=confidence,
            uncertainty=uncertainty,
        )

    def _resolve_weights(
        self,
        n_models: int,
        weights: Sequence[Any] | None,
    ) -> tuple[float, ...]:
        if weights is not None:
            return tuple(
                EnsembleWeighting.normalize_weights(
                    weights=weights,
                    n_models=n_models,
                )
            )

        return tuple(
            EnsembleWeighting.equal_weights(
                n_models
            )
        )

    @staticmethod
    def _calculate_confidence(
        weights: Sequence[float],
    ) -> float:
        """
        Calculate confidence from normalized model weights.

        Equal weights produce lower confidence than concentrated
        weights, while a single fully weighted model produces 1.0.
        """
        if len(weights) == 1:
            return 1.0

        maximum_weight = max(weights)
        equal_weight = 1.0 / len(weights)

        confidence = (
            maximum_weight - equal_weight
        ) / (
            1.0 - equal_weight
        )

        return float(
            min(
                1.0,
                max(0.0, confidence),
            )
        )

    @staticmethod
    def _validate_scalar_predictions(
        predictions: Any,
    ) -> tuple[float, ...]:
        if (
            isinstance(predictions, (str, bytes))
            or not isinstance(predictions, Sequence)
        ):
            raise TypeError(
                "predictions must be a sequence."
            )

        if len(predictions) == 0:
            raise ValueError(
                "predictions must not be empty."
            )

        validated: list[float] = []

        for value in predictions:
            if (
                isinstance(value, bool)
                or not isinstance(
                    value,
                    (int, float),
                )
            ):
                raise TypeError(
                    "prediction values must be numeric."
                )

            numeric_value = float(value)

            if not math.isfinite(numeric_value):
                raise ValueError(
                    "prediction values must be finite."
                )

            validated.append(numeric_value)

        return tuple(validated)

    @staticmethod
    def _validate_vector_predictions(
        predictions: Any,
    ) -> tuple[tuple[float, ...], ...]:
        if (
            isinstance(predictions, (str, bytes))
            or not isinstance(predictions, Sequence)
        ):
            raise TypeError(
                "predictions must be a sequence."
            )

        if len(predictions) == 0:
            raise ValueError(
                "predictions must not be empty."
            )

        validated_vectors: list[
            tuple[float, ...]
        ] = []

        vector_length: int | None = None

        for vector in predictions:
            if (
                isinstance(vector, (str, bytes))
                or not isinstance(vector, Sequence)
            ):
                raise TypeError(
                    "each prediction must be a sequence."
                )

            if len(vector) == 0:
                raise ValueError(
                    "prediction vectors must not be empty."
                )

            validated_vector: list[float] = []

            for value in vector:
                if (
                    isinstance(value, bool)
                    or not isinstance(
                        value,
                        (int, float),
                    )
                ):
                    raise TypeError(
                        "vector values must be numeric."
                    )

                numeric_value = float(value)

                if not math.isfinite(numeric_value):
                    raise ValueError(
                        "vector values must be finite."
                    )

                validated_vector.append(
                    numeric_value
                )

            current_length = len(validated_vector)

            if vector_length is None:
                vector_length = current_length

            elif current_length != vector_length:
                raise ValueError(
                    "all prediction vectors must have "
                    "the same length."
                )

            validated_vectors.append(
                tuple(validated_vector)
            )

        return tuple(validated_vectors)

    @staticmethod
    def _validate_vote_predictions(
        predictions: Any,
    ) -> tuple[Any, ...]:
        if (
            isinstance(predictions, (str, bytes))
            or not isinstance(predictions, Sequence)
        ):
            raise TypeError(
                "predictions must be a sequence."
            )

        if len(predictions) == 0:
            raise ValueError(
                "predictions must not be empty."
            )

        validated = tuple(predictions)

        for value in validated:
            if value is None:
                raise TypeError(
                    "prediction values must not be None."
                )

        return validated