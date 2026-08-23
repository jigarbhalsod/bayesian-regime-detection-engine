from typing import Optional

from .calibration import TemperatureCalibrator
from .confidence import ConfidenceEntropyAnalyzer
from .result import EnsembleResult
from .uncertainty import UncertaintyResult
from .uncertainty_estimator import PredictiveUncertaintyEstimator


class EnsembleUncertaintyResult:
    """
    Unified result containing the ensemble output and its
    uncertainty analyses.
    """

    def __init__(
        self,
        ensemble_result: EnsembleResult,
        predictive_uncertainty: UncertaintyResult,
        confidence_entropy: UncertaintyResult,
        calibrated: bool,
    ) -> None:
        self.ensemble_result = ensemble_result
        self.predictive_uncertainty = predictive_uncertainty
        self.confidence_entropy = confidence_entropy
        self.calibrated = calibrated


class EnsembleUncertaintyIntegrator:
    """
    Integrate ensemble calibration, predictive uncertainty,
    and confidence/entropy analysis into one pipeline.
    """

    def __init__(
        self,
        predictive_estimator: Optional[
            PredictiveUncertaintyEstimator
        ] = None,
        confidence_analyzer: Optional[
            ConfidenceEntropyAnalyzer
        ] = None,
        calibrator: Optional[
            TemperatureCalibrator
        ] = None,
    ) -> None:
        self.predictive_estimator = (
            predictive_estimator
            or PredictiveUncertaintyEstimator()
        )

        self.confidence_analyzer = (
            confidence_analyzer
            or ConfidenceEntropyAnalyzer()
        )

        self.calibrator = calibrator

        self._validate_components()

    def integrate(
        self,
        result: EnsembleResult,
    ) -> EnsembleUncertaintyResult:
        """
        Run the complete ensemble-uncertainty integration pipeline.
        """

        if not isinstance(result, EnsembleResult):
            raise TypeError(
                "EnsembleUncertaintyIntegrator requires an "
                "EnsembleResult instance."
            )

        processed_result = result
        calibrated = False

        if self.calibrator is not None:
            processed_result = self.calibrator.calibrate(result)
            calibrated = True

        predictive_uncertainty = (
            self.predictive_estimator.estimate(processed_result)
        )

        confidence_entropy = (
            self.confidence_analyzer.analyze(processed_result)
        )

        return EnsembleUncertaintyResult(
            ensemble_result=processed_result,
            predictive_uncertainty=predictive_uncertainty,
            confidence_entropy=confidence_entropy,
            calibrated=calibrated,
        )

    def _validate_components(self) -> None:
        """
        Validate all injected integration components.
        """

        if not isinstance(
            self.predictive_estimator,
            PredictiveUncertaintyEstimator,
        ):
            raise TypeError(
                "predictive_estimator must be a "
                "PredictiveUncertaintyEstimator instance."
            )

        if not isinstance(
            self.confidence_analyzer,
            ConfidenceEntropyAnalyzer,
        ):
            raise TypeError(
                "confidence_analyzer must be a "
                "ConfidenceEntropyAnalyzer instance."
            )

        if (
            self.calibrator is not None
            and not isinstance(
                self.calibrator,
                TemperatureCalibrator,
            )
        ):
            raise TypeError(
                "calibrator must be a TemperatureCalibrator "
                "instance or None."
            )