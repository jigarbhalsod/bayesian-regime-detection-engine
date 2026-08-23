from dataclasses import dataclass


@dataclass(frozen=True)
class UncertaintyConfig:
    """Configuration for the uncertainty pipeline."""

    low_uncertainty_threshold: float = 0.15
    high_uncertainty_threshold: float = 0.40
    abstain_on_high_uncertainty: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.low_uncertainty_threshold <= 1.0:
            raise ValueError(
                "low_uncertainty_threshold must be between 0.0 and 1.0."
            )

        if not 0.0 <= self.high_uncertainty_threshold <= 1.0:
            raise ValueError(
                "high_uncertainty_threshold must be between 0.0 and 1.0."
            )

        if self.low_uncertainty_threshold > self.high_uncertainty_threshold:
            raise ValueError(
                "low_uncertainty_threshold cannot exceed "
                "high_uncertainty_threshold."
            )