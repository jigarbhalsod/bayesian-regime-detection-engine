from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class UncertaintyLevel(str, Enum):
    """
    Qualitative classification of predictive uncertainty.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class UncertaintyResult(BaseModel):
    """
    Standardized result produced by uncertainty estimation components.

    This contract allows different uncertainty estimators to expose
    their results in one consistent format.
    """

    model_config = ConfigDict(extra="forbid")

    prediction: Optional[Any] = None

    uncertainty_score: float = Field(
        ge=0.0,
        le=1.0,
    )

    confidence_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    entropy: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    uncertainty_level: UncertaintyLevel

    probabilities: Dict[Any, float] = Field(
        default_factory=dict,
    )

    components: Dict[str, float] = Field(
        default_factory=dict,
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
    )

    @field_validator("probabilities")
    @classmethod
    def validate_probabilities(
        cls,
        probabilities: Dict[Any, float],
    ) -> Dict[Any, float]:
        """
        Ensure probability values are valid.
        """

        for label, probability in probabilities.items():
            if probability < 0.0:
                raise ValueError(
                    f"Probability for '{label}' cannot be negative."
                )

            if probability > 1.0:
                raise ValueError(
                    f"Probability for '{label}' cannot exceed 1.0."
                )

        return probabilities

    @field_validator("components")
    @classmethod
    def validate_components(
        cls,
        components: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Ensure uncertainty component values are non-negative.
        """

        for name, value in components.items():
            if value < 0.0:
                raise ValueError(
                    f"Uncertainty component '{name}' cannot be negative."
                )

        return components

    @model_validator(mode="after")
    def validate_probability_total(self):
        """
        If probabilities are provided, ensure they form a valid
        normalized distribution.
        """

        if self.probabilities:
            total = sum(self.probabilities.values())

            if abs(total - 1.0) > 1e-6:
                raise ValueError(
                    "Probabilities must sum to 1.0."
                )

        return self

    @property
    def is_low_uncertainty(self) -> bool:
        return self.uncertainty_level == UncertaintyLevel.LOW

    @property
    def is_medium_uncertainty(self) -> bool:
        return self.uncertainty_level == UncertaintyLevel.MEDIUM

    @property
    def is_high_uncertainty(self) -> bool:
        return self.uncertainty_level == UncertaintyLevel.HIGH

    @property
    def has_confidence(self) -> bool:
        return self.confidence_score is not None

    @property
    def has_entropy(self) -> bool:
        return self.entropy is not None

    @property
    def component_names(self) -> List[str]:
        return list(self.components.keys())