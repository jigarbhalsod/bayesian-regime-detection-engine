from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EnsembleModelOutput:
    """
    Standardized representation of one model output prepared
    for ensemble consumption.
    """

    model_name: str
    prediction: Optional[Any] = None
    probabilities: Dict[Any, float] = field(default_factory=dict)
    success: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """
        Validate a single ensemble model output.
        """

        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("model_name must be a non-empty string.")

        if not isinstance(self.probabilities, dict):
            raise TypeError("probabilities must be a dictionary.")

        for label, probability in self.probabilities.items():
            if not isinstance(probability, (int, float)):
                raise ValueError(
                    f"Probability for label '{label}' in model "
                    f"'{self.model_name}' must be numeric."
                )

            if probability < 0:
                raise ValueError(
                    f"Probability for label '{label}' in model "
                    f"'{self.model_name}' cannot be negative."
                )


@dataclass
class PreparedEnsembleInput:
    """
    Collection of validated and aligned model outputs ready for
    consumption by an ensemble strategy.
    """

    outputs: List[EnsembleModelOutput] = field(default_factory=list)

    regime_labels: List[Any] = field(default_factory=list)

    skipped_models: List[str] = field(default_factory=list)

    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self, min_models: int = 1) -> None:
        """
        Validate the prepared ensemble input.
        """

        if min_models < 1:
            raise ValueError("min_models must be at least 1.")

        if len(self.outputs) < min_models:
            raise ValueError(
                f"At least {min_models} valid model output(s) are required. "
                f"Received {len(self.outputs)}."
            )

        model_names = [output.model_name for output in self.outputs]

        if len(set(model_names)) != len(model_names):
            raise ValueError(
                "Prepared ensemble input must not contain duplicate "
                "model names."
            )

        for output in self.outputs:
            output.validate()

    @property
    def model_count(self) -> int:
        """
        Number of valid participating model outputs.
        """

        return len(self.outputs)

    @property
    def participating_models(self) -> List[str]:
        """
        Names of models participating in the ensemble.
        """

        return [output.model_name for output in self.outputs]