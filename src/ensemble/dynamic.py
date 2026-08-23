from typing import Any, Dict, Iterable

from .config import EnsembleConfig
from .input import PreparedEnsembleInput
from .result import EnsembleResult
from .weighted import WeightedEnsembleStrategy


class DynamicWeightedEnsembleStrategy(WeightedEnsembleStrategy):
    """
    Ensemble strategy that derives model weights from performance scores.

    Performance scores are provided through EnsembleConfig.metadata
    using the key "performance_scores".

    Example:
        EnsembleConfig(
            metadata={
                "performance_scores": {
                    "hmm": 0.90,
                    "bayesian": 0.60,
                }
            }
        )
    """

    STRATEGY_NAME = "dynamic_weighted"

    @property
    def strategy_name(self) -> str:
        return self.STRATEGY_NAME

    def combine(
        self,
        model_outputs: Iterable[Any],
    ) -> EnsembleResult:
        """
        Combine model outputs using performance-derived dynamic weights.
        """

        if not isinstance(model_outputs, PreparedEnsembleInput):
            raise TypeError(
                "DynamicWeightedEnsembleStrategy requires a "
                "PreparedEnsembleInput instance."
            )

        model_outputs.validate(min_models=self.config.min_models)

        participating_models = model_outputs.participating_models

        weights = self._resolve_dynamic_weights(participating_models)

        combined_probabilities = self._combine_probabilities(
            model_outputs,
            weights,
        )

        prediction = self._select_prediction(combined_probabilities)

        result = EnsembleResult(
            prediction=prediction,
            probabilities=combined_probabilities,
            model_predictions={
                output.model_name: output.prediction
                for output in model_outputs.outputs
            },
            model_probabilities={
                output.model_name: dict(output.probabilities)
                for output in model_outputs.outputs
            },
            participating_models=participating_models,
            strategy=self.strategy_name,
            metadata={
                "weights": weights,
                "performance_scores": self._get_performance_scores(),
            },
        )

        return self.validate_result(result)

    def _get_performance_scores(self) -> Dict[str, float]:
        """
        Retrieve and validate configured performance scores.
        """

        performance_scores = self.config.metadata.get(
            "performance_scores"
        )

        if performance_scores is None:
            raise ValueError(
                "Dynamic weighting requires 'performance_scores' "
                "in EnsembleConfig.metadata."
            )

        if not isinstance(performance_scores, dict):
            raise TypeError(
                "performance_scores must be a dictionary."
            )

        return performance_scores

    def _resolve_dynamic_weights(
        self,
        model_names: list[str],
    ) -> Dict[str, float]:
        """
        Convert participating model performance scores into weights.
        """

        performance_scores = self._get_performance_scores()

        scores: Dict[str, float] = {}

        for model_name in model_names:
            if model_name not in performance_scores:
                raise ValueError(
                    f"Missing performance score for model "
                    f"'{model_name}'."
                )

            score = performance_scores[model_name]

            if not isinstance(score, (int, float)):
                raise TypeError(
                    f"Performance score for model '{model_name}' "
                    "must be numeric."
                )

            if score < 0:
                raise ValueError(
                    f"Performance score for model '{model_name}' "
                    "cannot be negative."
                )

            scores[model_name] = float(score)

        total_score = sum(scores.values())

        if total_score <= 0:
            raise ValueError(
                "The total performance score must be positive."
            )

        return {
            model_name: score / total_score
            for model_name, score in scores.items()
        }