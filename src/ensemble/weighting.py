from __future__ import annotations

from typing import Any, Sequence


class EnsembleWeighting:
    """
    Weight management utilities for ensemble models.

    Supports equal weighting, custom weighting, validation,
    normalization, and weighted aggregation.
    """

    @staticmethod
    def equal_weights(
        n_models: int,
    ) -> tuple[float, ...]:
        """
        Create equal weights for the specified number of models.
        """
        EnsembleWeighting._validate_model_count(
            n_models
        )

        weight = 1.0 / n_models

        return tuple(
            weight
            for _ in range(n_models)
        )

    @staticmethod
    def validate_weights(
        weights: Sequence[Any],
        n_models: int | None = None,
    ) -> tuple[float, ...]:
        """
        Validate a sequence of ensemble weights.

        Weights must:
        - be a non-empty sequence
        - contain numeric finite values
        - not contain negative values
        - contain at least one positive value
        - match n_models when provided
        """
        if isinstance(
            weights,
            (str, bytes),
        ) or not isinstance(
            weights,
            Sequence,
        ):
            raise TypeError(
                "weights must be a sequence."
            )

        if len(weights) == 0:
            raise ValueError(
                "weights cannot be empty."
            )

        if n_models is not None:
            EnsembleWeighting._validate_model_count(
                n_models
            )

            if len(weights) != n_models:
                raise ValueError(
                    "weights length must match n_models."
                )

        validated: list[float] = []

        for index, weight in enumerate(weights):
            validated.append(
                EnsembleWeighting._validate_weight(
                    weight,
                    index,
                )
            )

        if not any(
            weight > 0.0
            for weight in validated
        ):
            raise ValueError(
                "at least one weight must be greater than 0."
            )

        return tuple(validated)

    @staticmethod
    def normalize_weights(
        weights: Sequence[Any],
        n_models: int | None = None,
    ) -> tuple[float, ...]:
        """
        Validate and normalize weights so their sum equals 1.
        """
        validated = EnsembleWeighting.validate_weights(
            weights,
            n_models=n_models,
        )

        total = sum(validated)

        return tuple(
            weight / total
            for weight in validated
        )

    @staticmethod
    def aggregate(
        values: Sequence[Any],
        weights: Sequence[Any] | None = None,
    ) -> float:
        """
        Compute a weighted average.

        Equal weighting is used when weights are not supplied.
        """
        validated_values = (
            EnsembleWeighting._validate_values(
                values
            )
        )

        n_models = len(validated_values)

        if weights is None:
            normalized_weights = (
                EnsembleWeighting.equal_weights(
                    n_models
                )
            )
        else:
            normalized_weights = (
                EnsembleWeighting.normalize_weights(
                    weights,
                    n_models=n_models,
                )
            )

        return sum(
            value * weight
            for value, weight in zip(
                validated_values,
                normalized_weights,
            )
        )

    @staticmethod
    def _validate_model_count(
        value: Any,
    ) -> None:
        """
        Validate the number of ensemble models.
        """
        if isinstance(value, bool) or not isinstance(
            value,
            int,
        ):
            raise TypeError(
                "n_models must be an integer."
            )

        if value <= 0:
            raise ValueError(
                "n_models must be greater than 0."
            )

    @staticmethod
    def _validate_weight(
        value: Any,
        index: int,
    ) -> float:
        """
        Validate a single ensemble weight.
        """
        if isinstance(value, bool) or not isinstance(
            value,
            (int, float),
        ):
            raise TypeError(
                f"weight at index {index} must be numeric."
            )

        weight = float(value)

        if weight != weight:
            raise ValueError(
                f"weight at index {index} must be finite."
            )

        if weight in (
            float("inf"),
            float("-inf"),
        ):
            raise ValueError(
                f"weight at index {index} must be finite."
            )

        if weight < 0:
            raise ValueError(
                f"weight at index {index} cannot be negative."
            )

        return weight

    @staticmethod
    def _validate_values(
        values: Sequence[Any],
    ) -> tuple[float, ...]:
        """
        Validate values supplied for aggregation.
        """
        if isinstance(
            values,
            (str, bytes),
        ) or not isinstance(
            values,
            Sequence,
        ):
            raise TypeError(
                "values must be a sequence."
            )

        if len(values) == 0:
            raise ValueError(
                "values cannot be empty."
            )

        validated: list[float] = []

        for index, value in enumerate(values):
            if isinstance(value, bool) or not isinstance(
                value,
                (int, float),
            ):
                raise TypeError(
                    f"value at index {index} must be numeric."
                )

            numeric_value = float(value)

            if numeric_value != numeric_value:
                raise ValueError(
                    f"value at index {index} must be finite."
                )

            if numeric_value in (
                float("inf"),
                float("-inf"),
            ):
                raise ValueError(
                    f"value at index {index} must be finite."
                )

            validated.append(numeric_value)

        return tuple(validated)