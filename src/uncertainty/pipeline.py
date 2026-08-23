from collections.abc import Sequence

from src.ensemble.service import EnsembleService
from src.uncertainty.config import UncertaintyConfig
from src.uncertainty.result import UncertaintyResult


class UncertaintyPipeline:
    """Combines ensemble predictions with disagreement-based uncertainty."""

    def __init__(
        self,
        config: UncertaintyConfig | None = None,
        ensemble_service: EnsembleService | None = None,
    ) -> None:
        self.config = config or UncertaintyConfig()
        self.ensemble_service = ensemble_service or EnsembleService()

    def predict(
        self,
        predictions: Sequence[float],
        weights: Sequence[float] | None = None,
    ) -> UncertaintyResult:
        """Generate an ensemble prediction with uncertainty information."""

        values = tuple(float(value) for value in predictions)

        if not values:
            raise ValueError("predictions cannot be empty.")

        ensemble_result = self.ensemble_service.aggregate(
            predictions=values,
            weights=weights,
        )

        prediction = float(ensemble_result.prediction)

        uncertainty = self._calculate_uncertainty(
            values=values,
            prediction=prediction,
        )

        confidence = max(0.0, min(1.0, 1.0 - uncertainty))
        level = self._determine_level(uncertainty)

        abstained = (
            self.config.abstain_on_high_uncertainty
            and level == "high"
        )

        return UncertaintyResult(
            prediction=prediction,
            uncertainty=uncertainty,
            confidence=confidence,
            level=level,
            abstained=abstained,
        )

    def _calculate_uncertainty(
        self,
        values: tuple[float, ...],
        prediction: float,
    ) -> float:
        """Calculate normalized mean absolute disagreement."""

        if len(values) == 1:
            return 0.0

        mean_absolute_deviation = sum(
            abs(value - prediction)
            for value in values
        ) / len(values)

        scale = max(
            abs(prediction),
            max(abs(value) for value in values),
            1.0,
        )

        uncertainty = mean_absolute_deviation / scale

        return max(0.0, min(1.0, uncertainty))

    def _determine_level(self, uncertainty: float) -> str:
        """Classify uncertainty into low, medium, or high."""

        if uncertainty <= self.config.low_uncertainty_threshold:
            return "low"

        if uncertainty >= self.config.high_uncertainty_threshold:
            return "high"

        return "medium"