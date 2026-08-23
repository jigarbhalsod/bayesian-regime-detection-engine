from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class EnsembleConfig:
    """
    Configuration for ensemble strategies.

    This configuration provides common settings shared across
    different ensemble implementations.
    """

    name: str = "default_ensemble"
    strategy: str = "base"

    enabled_models: Optional[List[str]] = None
    model_weights: Dict[str, float] = field(default_factory=dict)

    normalize_weights: bool = True
    allow_missing_models: bool = True

    min_models: int = 1

    metadata: Dict[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        """
        Validate ensemble configuration.
        """

        if not self.name or not self.name.strip():
            raise ValueError("Ensemble name must be a non-empty string.")

        if not self.strategy or not self.strategy.strip():
            raise ValueError("Ensemble strategy must be a non-empty string.")

        if self.min_models < 1:
            raise ValueError("min_models must be at least 1.")

        if self.enabled_models is not None:
            if len(self.enabled_models) == 0:
                raise ValueError(
                    "enabled_models cannot be an empty list when provided."
                )

            if len(set(self.enabled_models)) != len(self.enabled_models):
                raise ValueError(
                    "enabled_models must not contain duplicate model names."
                )

        for model_name, weight in self.model_weights.items():
            if not model_name or not model_name.strip():
                raise ValueError(
                    "Model weight keys must be non-empty model names."
                )

            if not isinstance(weight, (int, float)):
                raise ValueError(
                    f"Weight for model '{model_name}' must be numeric."
                )

            if weight < 0:
                raise ValueError(
                    f"Weight for model '{model_name}' cannot be negative."
                )