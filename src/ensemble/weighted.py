from typing import Any, Dict, Iterable

from .base import BaseEnsembleStrategy
from .input import PreparedEnsembleInput
from .result import EnsembleResult


class WeightedEnsembleStrategy(BaseEnsembleStrategy):
    """
    Combine model probability distributions using configured weights.
    """

    STRATEGY_NAME = "weighted"

    @property
    def strategy_name(self) -> str:
        return self.STRATEGY_NAME

    def combine(
        self,
        model_outputs: Iterable[Any],
    ) -> EnsembleResult:
        """
        Combine prepared model outputs using weighted probability
        aggregation.
        """

        if isinstance(model_outputs, PreparedEnsembleInput):
            prepared = model_outputs
        else:
            raise TypeError(
                "WeightedEnsembleStrategy requires a "
                "PreparedEnsembleInput instance."
            )

        prepared.validate(min_models=self.config.min_models)

        participating_models = prepared.participating_models

        weights = self._resolve_weights(participating_models)

        combined_probabilities = self._combine_probabilities(
            prepared,
            weights,
        )

        prediction = self._select_prediction(combined_probabilities)

        result = EnsembleResult(
            prediction=prediction,
            probabilities=combined_probabilities,
            model_predictions={
                output.model_name: output.prediction
                for output in prepared.outputs
            },
            model_probabilities={
                output.model_name: dict(output.probabilities)
                for output in prepared.outputs
            },
            participating_models=participating_models,
            strategy=self.strategy_name,
            metadata={
                "weights": weights,
            },
        )

        return self.validate_result(result)

    def _resolve_weights(
        self,
        model_names: list[str],
    ) -> Dict[str, float]:
        """
        Resolve and validate weights for participating models.

        Models without explicitly configured weights receive a default
        weight of 1.0.
        """

        weights = {
            model_name: self.config.model_weights.get(
                model_name,
                1.0,
            )
            for model_name in model_names
        }

        total_weight = sum(weights.values())

        if total_weight <= 0:
            raise ValueError(
                "The total weight of participating models must be positive."
            )

        if self.config.normalize_weights:
            weights = {
                model_name: weight / total_weight
                for model_name, weight in weights.items()
            }

        return weights

    @staticmethod
    def _combine_probabilities(
        prepared: PreparedEnsembleInput,
        weights: Dict[str, float],
    ) -> Dict[Any, float]:
        """
        Combine probability distributions across all regime labels.
        """

        combined = {
            label: 0.0
            for label in prepared.regime_labels
        }

        for output in prepared.outputs:
            weight = weights[output.model_name]

            for label in prepared.regime_labels:
                probability = output.probabilities.get(label, 0.0)
                combined[label] += weight * probability

        total_probability = sum(combined.values())

        if total_probability <= 0:
            raise ValueError(
                "Combined probabilities must have a positive total."
            )

        return {
            label: probability / total_probability
            for label, probability in combined.items()
        }

    @staticmethod
    def _select_prediction(
        probabilities: Dict[Any, float],
    ) -> Any:
        """
        Select the regime with the highest combined probability.
        """

        if not probabilities:
            raise ValueError(
                "Cannot select a prediction from empty probabilities."
            )

        return max(
            probabilities,
            key=probabilities.get,
        )