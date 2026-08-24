"""
Explainability package for the Bayesian Regime Detection Engine.

This package provides components for:
- Base explainability abstractions
- Feature attribution and importance
- Prediction explanations
- Explanation aggregation
- Regime contribution analysis
- Regime explanation generation
- Integrated regime explanation pipelines
- Explanation summaries
- Consistency validation
- High-level explainability service
"""

from src.explainability.config import ExplainabilityConfig

from src.explainability.models import (
    ExplanationResult,
    FeatureAttribution,
    FeatureImportance,
)

from src.explainability.base import BaseExplainer

from src.explainability.global_importance import (
    GlobalFeatureImportanceExplainer,
)

from src.explainability.prediction import (
    PredictionExplainer,
)

from src.explainability.aggregation import (
    ExplanationAggregator,
)

from src.explainability.regime_models import (
    RegimeContribution,
)

from src.explainability.regime_contribution import (
    RegimeContributionAnalyzer,
)

from src.explainability.regime_generator import (
    RegimeExplanationGenerator,
)

from src.explainability.regime_integration import (
    RegimeExplanationPipeline,
)

from src.explainability.summary import (
    ExplanationSummaryEngine,
)

from src.explainability.consistency import (
    ExplanationConsistencyValidator,
)

from src.explainability.service import (
    ExplainabilityService,
)


__all__ = [
    # Configuration
    "ExplainabilityConfig",

    # Core models
    "ExplanationResult",
    "FeatureAttribution",
    "FeatureImportance",

    # Base
    "BaseExplainer",

    # Group B components
    "GlobalFeatureImportanceExplainer",
    "PredictionExplainer",
    "ExplanationAggregator",

    # Group C components
    "RegimeContribution",
    "RegimeContributionAnalyzer",
    "RegimeExplanationGenerator",
    "RegimeExplanationPipeline",

    # Group D components
    "ExplanationSummaryEngine",
    "ExplanationConsistencyValidator",
    "ExplainabilityService",
]