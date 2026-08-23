import math
from typing import Any, Dict

from .result import EnsembleResult


class TemperatureCalibrator:
    """
    Calibrate ensemble probability distributions using
    temperature scaling.

    Temperature scaling transforms probabilities through:

        calibrated_i = exp(log(p_i) / T)
                       / sum(exp(log(p_j) / T))

    T = 1.0 preserves the original distribution.
    T > 1.0 softens the distribution.
    T < 1.0 sharpens the distribution.
    """

    def __init__(
        self,
        temperature: float = 1.0,
    ) -> None:
        self.temperature = temperature
        self._validate_temperature()

    def calibrate(
        self,
        result: EnsembleResult,
    ) -> EnsembleResult:
        """
        Return a calibrated EnsembleResult.
        """

        if not isinstance(result, EnsembleResult):
            raise TypeError(
                "TemperatureCalibrator requires an "
                "EnsembleResult instance."
            )

        probabilities = dict(result.probabilities)

        self._validate_probabilities(probabilities)

        calibrated_probabilities = (
            self._apply_temperature_scaling(probabilities)
        )

        return EnsembleResult(
            prediction=self._get_prediction(
                calibrated_probabilities
            ),
            probabilities=calibrated_probabilities,
            participating_models=list(
                result.participating_models
            ),
            strategy=result.strategy,
            metadata={
                **getattr(result, "metadata", {}),
                "calibrated": True,
                "calibration_method": "temperature_scaling",
                "temperature": self.temperature,
            },
        )

    def _validate_temperature(self) -> None:
        """
        Validate the calibration temperature.
        """

        if not isinstance(
            self.temperature,
            (int, float),
        ):
            raise TypeError(
                "temperature must be numeric."
            )

        if self.temperature <= 0.0:
            raise ValueError(
                "temperature must be greater than 0."
            )

    @staticmethod
    def _validate_probabilities(
        probabilities: Dict[Any, float],
    ) -> None:
        """
        Validate the input probability distribution.
        """

        if not probabilities:
            raise ValueError(
                "Cannot calibrate empty probabilities."
            )

        total = 0.0

        for label, probability in probabilities.items():
            if not isinstance(
                probability,
                (int, float),
            ):
                raise TypeError(
                    f"Probability for '{label}' must be numeric."
                )

            if probability < 0.0:
                raise ValueError(
                    f"Probability for '{label}' cannot be negative."
                )

            if probability > 1.0:
                raise ValueError(
                    f"Probability for '{label}' cannot exceed 1.0."
                )

            total += probability

        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                "Probabilities must sum to 1.0."
            )

    def _apply_temperature_scaling(
        self,
        probabilities: Dict[Any, float],
    ) -> Dict[Any, float]:
        """
        Apply temperature scaling and return a normalized
        probability distribution.
        """

        epsilon = 1e-12

        logits = {
            label: math.log(max(probability, epsilon))
            / self.temperature
            for label, probability in probabilities.items()
        }

        max_logit = max(logits.values())

        exponentials = {
            label: math.exp(logit - max_logit)
            for label, logit in logits.items()
        }

        total = sum(exponentials.values())

        return {
            label: value / total
            for label, value in exponentials.items()
        }

    @staticmethod
    def _get_prediction(
        probabilities: Dict[Any, float],
    ) -> Any:
        """
        Return the class with the highest calibrated probability.
        """

        return max(
            probabilities,
            key=probabilities.get,
        )