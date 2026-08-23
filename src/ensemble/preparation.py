from typing import Any, Iterable, List

from .config import EnsembleConfig
from .input import (
    EnsembleModelOutput,
    PreparedEnsembleInput,
)


class EnsembleInputPreparer:
    """
    Prepares raw or standardized model outputs for ensemble strategies.

    The preparer validates outputs, handles unavailable models,
    aligns regime labels, and normalizes probability distributions.
    """

    def __init__(self, config: EnsembleConfig | None = None) -> None:
        self.config = config or EnsembleConfig()
        self.config.validate()

    def prepare(
        self,
        model_outputs: Iterable[EnsembleModelOutput],
    ) -> PreparedEnsembleInput:
        """
        Prepare model outputs for ensemble consumption.
        """

        if model_outputs is None:
            raise ValueError("model_outputs cannot be None.")

        valid_outputs: List[EnsembleModelOutput] = []
        skipped_models: List[str] = []

        for output in model_outputs:
            if not isinstance(output, EnsembleModelOutput):
                raise TypeError(
                    "All model outputs must be EnsembleModelOutput instances."
                )

            if not output.success:
                if self.config.allow_missing_models:
                    skipped_models.append(output.model_name)
                    continue

                raise ValueError(
                    f"Model '{output.model_name}' was unsuccessful and "
                    "allow_missing_models is False."
                )

            if (
                self.config.enabled_models is not None
                and output.model_name not in self.config.enabled_models
            ):
                continue

            output.validate()

            normalized_output = self._normalize_output(output)

            valid_outputs.append(normalized_output)

        prepared = PreparedEnsembleInput(
            outputs=valid_outputs,
            regime_labels=self._collect_regime_labels(valid_outputs),
            skipped_models=skipped_models,
        )

        prepared.validate(min_models=self.config.min_models)

        return prepared

    def _normalize_output(
        self,
        output: EnsembleModelOutput,
    ) -> EnsembleModelOutput:
        """
        Normalize the probability distribution of one model output.
        """

        probabilities = dict(output.probabilities)

        if not probabilities:
            return EnsembleModelOutput(
                model_name=output.model_name,
                prediction=output.prediction,
                probabilities={},
                success=output.success,
                metadata=dict(output.metadata),
            )

        total_probability = sum(probabilities.values())

        if total_probability <= 0:
            raise ValueError(
                f"Probabilities for model '{output.model_name}' "
                "must have a positive total."
            )

        normalized_probabilities = {
            label: probability / total_probability
            for label, probability in probabilities.items()
        }

        return EnsembleModelOutput(
            model_name=output.model_name,
            prediction=output.prediction,
            probabilities=normalized_probabilities,
            success=output.success,
            metadata=dict(output.metadata),
        )

    @staticmethod
    def _collect_regime_labels(
        outputs: List[EnsembleModelOutput],
    ) -> List[Any]:
        """
        Collect all unique regime labels across model outputs while
        preserving first-seen order.
        """

        labels: List[Any] = []

        for output in outputs:
            for label in output.probabilities:
                if label not in labels:
                    labels.append(label)

            if (
                output.prediction is not None
                and output.prediction not in labels
            ):
                labels.append(output.prediction)

        return labels