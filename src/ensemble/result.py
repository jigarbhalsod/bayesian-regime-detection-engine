from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EnsembleResult:
    """
    Standardized result produced by an ensemble strategy.
    """

    prediction: Optional[Any] = None

    probabilities: Dict[Any, float] = field(default_factory=dict)

    model_predictions: Dict[str, Any] = field(default_factory=dict)

    model_probabilities: Dict[str, Dict[Any, float]] = field(
        default_factory=dict
    )

    participating_models: List[str] = field(default_factory=list)

    strategy: str = "base"

    success: bool = True
    message: Optional[str] = None

    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """
        Validate the ensemble result structure.
        """

        if not self.strategy or not self.strategy.strip():
            raise ValueError(
                "Ensemble result strategy must be a non-empty string."
            )

        if len(set(self.participating_models)) != len(
            self.participating_models
        ):
            raise ValueError(
                "participating_models must not contain duplicates."
            )

        for label, probability in self.probabilities.items():
            if not isinstance(probability, (int, float)):
                raise ValueError(
                    f"Probability for label '{label}' must be numeric."
                )

            if probability < 0:
                raise ValueError(
                    f"Probability for label '{label}' cannot be negative."
                )

        for model_name, probabilities in self.model_probabilities.items():
            if not model_name or not model_name.strip():
                raise ValueError(
                    "model_probabilities contains an invalid model name."
                )

            for label, probability in probabilities.items():
                if not isinstance(probability, (int, float)):
                    raise ValueError(
                        f"Probability for model '{model_name}', "
                        f"label '{label}' must be numeric."
                    )

                if probability < 0:
                    raise ValueError(
                        f"Probability for model '{model_name}', "
                        f"label '{label}' cannot be negative."
                    )

    @property
    def model_count(self) -> int:
        """
        Number of models participating in the ensemble.
        """

        return len(self.participating_models)