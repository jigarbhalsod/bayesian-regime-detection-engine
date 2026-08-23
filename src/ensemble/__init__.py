"""
Ensemble and uncertainty components for the Bayesian Regime
Detection Engine.
"""

from .base import BaseEnsembleStrategy
from .calibration import TemperatureCalibrator
from .confidence import ConfidenceEntropyAnalyzer
from .config import EnsembleConfig
from .dynamic import DynamicWeightedEnsembleStrategy
from .input import (
    EnsembleModelOutput,
    PreparedEnsembleInput,
)
from .integration import (
    EnsembleUncertaintyIntegrator,
    EnsembleUncertaintyResult,
)
from .phase10_pipeline import Phase10Pipeline
from .preparation import EnsembleInputPreparer
from .registry import EnsembleRegistry
from .result import EnsembleResult
from .uncertainty import (
    UncertaintyLevel,
    UncertaintyResult,
)
from .uncertainty_estimator import PredictiveUncertaintyEstimator
from .voting import VotingEnsembleStrategy
from .weighted import WeightedEnsembleStrategy


__all__ = [
    # Base architecture
    "BaseEnsembleStrategy",
    "EnsembleConfig",
    "EnsembleResult",

    # Input preparation
    "EnsembleModelOutput",
    "PreparedEnsembleInput",
    "EnsembleInputPreparer",

    # Registry
    "EnsembleRegistry",

    # Ensemble strategies
    "WeightedEnsembleStrategy",
    "DynamicWeightedEnsembleStrategy",
    "VotingEnsembleStrategy",

    # Uncertainty
    "UncertaintyLevel",
    "UncertaintyResult",
    "PredictiveUncertaintyEstimator",
    "ConfidenceEntropyAnalyzer",

    # Calibration
    "TemperatureCalibrator",

    # Integration
    "EnsembleUncertaintyIntegrator",
    "EnsembleUncertaintyResult",

    # Final Phase 10 pipeline
    "Phase10Pipeline",
]