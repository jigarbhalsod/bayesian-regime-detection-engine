from __future__ import annotations

from typing import Any, Sequence

from src.ensemble.weighting import EnsembleWeighting


class EnsembleAggregator:
    """
    Aggregation utilities for combining ensemble model outputs.

    Supports:
    - arithmetic mean aggregation
    - weighted mean aggregation
    - per-dimension aggregation for prediction vectors
    """

    @staticmethod
    def mean(
        values: Sequence[Any],
    ) -> float:
        """
        Compute the arithmetic mean of numeric values.
        """
        validated = EnsembleWeighting._validate_values(
            values
        )

        return sum(validated) / len(validated)

    @staticmethod
    def weighted_mean(
        values: Sequence[Any],
        weights: Sequence[Any] | None = None,
    ) -> float:
        """
        Compute a weighted mean.

        Equal weighting is used when weights are not supplied.
        """
        return EnsembleWeighting.aggregate(
            values=values,
            weights=weights,
        )

    @staticmethod
    def aggregate_vectors(
        vectors: Sequence[Sequence[Any]],
        weights: Sequence[Any] | None = None,
    ) -> tuple[float, ...]:
        """
        Aggregate equally sized numeric vectors element by element.

        Example
        -------
        vectors = [
            [1.0, 2.0],
            [3.0, 4.0],
        ]

        Result with equal weighting:
        (2.0, 3.0)
        """
        validated_vectors = (
            EnsembleAggregator._validate_vectors(
                vectors
            )
        )

        n_vectors = len(validated_vectors)

        if weights is None:
            normalized_weights = (
                EnsembleWeighting.equal_weights(
                    n_vectors
                )
            )
        else:
            normalized_weights = (
                EnsembleWeighting.normalize_weights(
                    weights,
                    n_models=n_vectors,
                )
            )

        vector_length = len(validated_vectors[0])

        aggregated: list[float] = []

        for index in range(vector_length):
            value = sum(
                vector[index] * weight
                for vector, weight in zip(
                    validated_vectors,
                    normalized_weights,
                )
            )

            aggregated.append(value)

        return tuple(aggregated)

    @staticmethod
    def _validate_vectors(
        vectors: Sequence[Sequence[Any]],
    ) -> tuple[tuple[float, ...], ...]:
        """
        Validate a non-empty collection of equally sized vectors.
        """
        if isinstance(
            vectors,
            (str, bytes),
        ) or not isinstance(
            vectors,
            Sequence,
        ):
            raise TypeError(
                "vectors must be a sequence."
            )

        if len(vectors) == 0:
            raise ValueError(
                "vectors cannot be empty."
            )

        validated: list[tuple[float, ...]] = []
        expected_length: int | None = None

        for index, vector in enumerate(vectors):
            try:
                numeric_vector = (
                    EnsembleWeighting._validate_values(
                        vector
                    )
                )
            except TypeError as exc:
                raise TypeError(
                    f"vector at index {index} is invalid."
                ) from exc
            except ValueError as exc:
                raise ValueError(
                    f"vector at index {index} is invalid."
                ) from exc

            if expected_length is None:
                expected_length = len(numeric_vector)

            elif len(numeric_vector) != expected_length:
                raise ValueError(
                    "all vectors must have the same length."
                )

            validated.append(numeric_vector)

        return tuple(validated)