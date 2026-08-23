from collections import Counter
from typing import Any, Dict, Iterable

from .base import BaseEnsembleStrategy
from .input import PreparedEnsembleInput
from .result import EnsembleResult


class VotingEnsembleStrategy(BaseEnsembleStrategy):
    """
    Combine model predictions using majority voting.

    The strategy counts valid model predictions and converts vote
    counts into a normalized probability distribution.
    """

    STRATEGY_NAME = "voting"

    @property
    def strategy_name(self) -> str:
        return self.STRATEGY_NAME

    def combine(
        self,
        model_outputs: Iterable[Any],
    ) -> EnsembleResult:
        """
        Combine prepared model outputs using majority voting.
        """

        if not isinstance(model_outputs, PreparedEnsembleInput):
            raise TypeError(
                "VotingEnsembleStrategy requires a "
                "PreparedEnsembleInput instance."
            )

        model_outputs.validate(min_models=self.config.min_models)

        predictions = self._extract_predictions(model_outputs)

        vote_counts = Counter(predictions.values())

        if not vote_counts:
            raise ValueError(
                "Voting requires at least one valid model prediction."
            )

        prediction = self._select_prediction(
            vote_counts,
            model_outputs.regime_labels,
        )

        probabilities = self._calculate_vote_probabilities(
            vote_counts,
            model_outputs.regime_labels,
        )

        result = EnsembleResult(
            prediction=prediction,
            probabilities=probabilities,
            model_predictions=predictions,
            model_probabilities={
                output.model_name: dict(output.probabilities)
                for output in model_outputs.outputs
            },
            participating_models=model_outputs.participating_models,
            strategy=self.strategy_name,
            metadata={
                "vote_counts": dict(vote_counts),
            },
        )

        return self.validate_result(result)

    @staticmethod
    def _extract_predictions(
        prepared: PreparedEnsembleInput,
    ) -> Dict[str, Any]:
        """
        Extract and validate predictions from participating models.
        """

        predictions: Dict[str, Any] = {}

        for output in prepared.outputs:
            if output.prediction is None:
                raise ValueError(
                    f"Model '{output.model_name}' does not have a prediction."
                )

            predictions[output.model_name] = output.prediction

        return predictions

    @staticmethod
    def _select_prediction(
        vote_counts: Counter,
        regime_labels: list[Any],
    ) -> Any:
        """
        Select the winner by highest vote count.

        Ties are resolved using regime_labels order. This makes
        tie-breaking deterministic and reproducible.
        """

        max_votes = max(vote_counts.values())

        tied_labels = {
            label
            for label, count in vote_counts.items()
            if count == max_votes
        }

        for label in regime_labels:
            if label in tied_labels:
                return label

        # Defensive fallback; normally unreachable.
        return next(iter(tied_labels))

    @staticmethod
    def _calculate_vote_probabilities(
        vote_counts: Counter,
        regime_labels: list[Any],
    ) -> Dict[Any, float]:
        """
        Convert vote counts into a normalized distribution across
        known regime labels.
        """

        total_votes = sum(vote_counts.values())

        if total_votes <= 0:
            raise ValueError(
                "Total vote count must be positive."
            )

        labels = list(regime_labels)

        for label in vote_counts:
            if label not in labels:
                labels.append(label)

        return {
            label: vote_counts.get(label, 0) / total_votes
            for label in labels
        }