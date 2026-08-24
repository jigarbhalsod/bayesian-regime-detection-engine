"""High-level explainability service."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.explainability.aggregation import ExplanationAggregator
from src.explainability.consistency import ExplanationConsistencyValidator
from src.explainability.global_importance import GlobalFeatureImportanceExplainer
from src.explainability.models import (
    ExplanationResult,
    FeatureAttribution,
    FeatureImportance,
)
from src.explainability.prediction import PredictionExplainer
from src.explainability.regime_contribution import RegimeContributionAnalyzer
from src.explainability.regime_generator import RegimeExplanationGenerator
from src.explainability.regime_models import (
    RegimeContribution,
    RegimeExplanation,
)
from src.explainability.summary import ExplanationSummaryEngine


class ExplainabilityService:
    """High-level orchestration service for the explainability pipeline."""

    def __init__(
        self,
        prediction_explainer: Optional[PredictionExplainer] = None,
        importance_explainer: Optional[
            GlobalFeatureImportanceExplainer
        ] = None,
        aggregator: Optional[ExplanationAggregator] = None,
        contribution_analyzer: Optional[
            RegimeContributionAnalyzer
        ] = None,
        regime_generator: Optional[
            RegimeExplanationGenerator
        ] = None,
        summary_engine: Optional[
            ExplanationSummaryEngine
        ] = None,
        consistency_validator: Optional[
            ExplanationConsistencyValidator
        ] = None,
    ) -> None:
        self.prediction_explainer = (
            prediction_explainer
            if prediction_explainer is not None
            else PredictionExplainer()
        )

        self.importance_explainer = (
            importance_explainer
            if importance_explainer is not None
            else GlobalFeatureImportanceExplainer()
        )

        self.aggregator = (
            aggregator
            if aggregator is not None
            else ExplanationAggregator()
        )

        self.contribution_analyzer = (
            contribution_analyzer
            if contribution_analyzer is not None
            else RegimeContributionAnalyzer()
        )

        self.regime_generator = (
            regime_generator
            if regime_generator is not None
            else RegimeExplanationGenerator()
        )

        self.summary_engine = (
            summary_engine
            if summary_engine is not None
            else ExplanationSummaryEngine()
        )

        self.consistency_validator = (
            consistency_validator
            if consistency_validator is not None
            else ExplanationConsistencyValidator()
        )

    def explain(
        self,
        prediction: Optional[str] = None,
        confidence: Optional[float] = None,
        attributions: Optional[Dict[str, float]] = None,
        importances: Optional[Dict[str, float]] = None,
        regime: Optional[str] = None,
        probability: Optional[float] = None,
        contributions: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run the complete explainability pipeline."""

        metadata = {} if metadata is None else metadata

        # Used by summary engine: accepts both explanation types.
        all_explanations: List[
            ExplanationResult | RegimeExplanation
        ] = []

        # Used by aggregator: accepts ONLY ExplanationResult.
        aggregation_results: List[ExplanationResult] = []

        # ---------------------------------------------------------
        # Prediction flow
        # ---------------------------------------------------------
        if prediction is not None:
            feature_attributions = self._build_attributions(
                attributions
            )
            feature_importances = self._build_importances(
                importances
            )

            prediction_result = self.prediction_explainer.explain(
                feature_attributions,
                prediction=prediction,
                confidence=confidence,
                importances=feature_importances,
                metadata=metadata,
            )

            all_explanations.append(prediction_result)
            aggregation_results.append(prediction_result)

        # ---------------------------------------------------------
        # Regime flow
        # ---------------------------------------------------------
        if regime is not None:
            regime_contributions = self._build_contributions(
                contributions
            )

            regime_result = self.regime_generator.generate(
                regime=regime,
                probability=probability,
                contributions=regime_contributions,
                metadata=metadata,
            )

            all_explanations.append(regime_result)

            # Aggregator only accepts ExplanationResult.
            # Convert the regime result into an aggregation-safe object.
            aggregation_results.append(
                ExplanationResult(
                    attributions=[],
                    importances=[],
                    prediction=None,
                    confidence=probability,
                    warnings=[],
                    metadata=dict(metadata),
                )
            )

        # ---------------------------------------------------------
        # Aggregate prediction-compatible results
        # ---------------------------------------------------------
        explanation = self.aggregator.aggregate(
            aggregation_results
        )

        # ---------------------------------------------------------
        # Build summary from the REAL prediction + regime objects
        # ---------------------------------------------------------
        summary = self.summary_engine.summarize(
            all_explanations
        )

        # ---------------------------------------------------------
        # Validate SUMMARY, not ExplanationResult
        # ---------------------------------------------------------
        consistency = self.consistency_validator.validate(
            summary
        )

        return {
            "explanation": explanation,
            "summary": summary,
            "consistency": consistency,
        }

    @staticmethod
    def _build_attributions(
        values: Optional[Dict[str, float]],
    ) -> List[FeatureAttribution]:
        """Convert attribution dictionary into model objects."""

        if values is None:
            return []

        return [
            FeatureAttribution(
                feature_name=str(name).strip(),
                attribution=value,
            )
            for name, value in values.items()
        ]

    @staticmethod
    def _build_importances(
        values: Optional[Dict[str, float]],
    ) -> List[FeatureImportance]:
        """Convert importance dictionary into model objects."""

        if values is None:
            return []

        return [
            FeatureImportance(
                feature_name=str(name).strip(),
                importance=value,
            )
            for name, value in values.items()
        ]

    def _build_contributions(
        self,
        values: Optional[Dict[str, float]],
    ) -> List[RegimeContribution]:
        """Analyze and convert regime contribution values."""

        if values is None:
            return []

        return self.contribution_analyzer.analyze(values)