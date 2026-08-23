from typing import Optional

from .calibration import TemperatureCalibrator
from .confidence import ConfidenceEntropyAnalyzer
from .integration import (
    EnsembleUncertaintyIntegrator,
    EnsembleUncertaintyResult,
)
from .result import EnsembleResult
from .uncertainty_estimator import PredictiveUncertaintyEstimator


class Phase10Pipeline:
    """
    Final Phase 10 orchestration pipeline.

    This pipeline accepts an EnsembleResult and runs the complete
    calibration and uncertainty workflow.
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
        self.integrator = EnsembleUncertaintyIntegrator(
            predictive_estimator=predictive_estimator,
            confidence_analyzer=confidence_analyzer,
            calibrator=calibrator,
        )

    def run(
        self,
        result: EnsembleResult,
    ) -> EnsembleUncertaintyResult:
        """
        Run the complete Phase 10 ensemble and uncertainty workflow.
        """

        if not isinstance(result, EnsembleResult):
            raise TypeError(
                "Phase10Pipeline requires an EnsembleResult instance."
            )

        return self.integrator.integrate(result)