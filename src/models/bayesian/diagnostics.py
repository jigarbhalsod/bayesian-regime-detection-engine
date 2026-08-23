from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from .model import BayesianPrediction


@dataclass(frozen=True)
class BayesianDiagnosticsResult:
    """
    Aggregated diagnostics for Bayesian regime predictions.
    """

    n_samples: int
    n_regimes: int
    regime_counts: Tensor
    regime_frequencies: Tensor
    mean_confidence: float
    mean_predictive_entropy: float
    mean_expected_entropy: float
    mean_mutual_information: float
    mean_epistemic_uncertainty: float
    mean_probability_std: float
    low_confidence_count: int
    low_confidence_fraction: float
    low_confidence_mask: Tensor


class BayesianDiagnostics:
    """
    Analyze prediction confidence and uncertainty produced by a
    BayesianRegimeModel.

    This class operates on BayesianPrediction objects returned by
    BayesianRegimeModel.predict_with_uncertainty().
    """

    def __init__(
        self,
        confidence_threshold: float = 0.50,
    ) -> None:
        self.confidence_threshold = (
            self._validate_confidence_threshold(
                confidence_threshold
            )
        )

    @staticmethod
    def _validate_confidence_threshold(
        value: Any,
    ) -> float:
        """Validate the low-confidence threshold."""
        if isinstance(value, bool) or not isinstance(
            value,
            (int, float),
        ):
            raise TypeError(
                "confidence_threshold must be a number."
            )

        value = float(value)

        if value <= 0 or value > 1:
            raise ValueError(
                "confidence_threshold must be in the range "
                "(0, 1]."
            )

        return value

    @staticmethod
    def _validate_prediction(
        prediction: BayesianPrediction,
    ) -> BayesianPrediction:
        """Validate the Bayesian prediction object."""
        if not isinstance(
            prediction,
            BayesianPrediction,
        ):
            raise TypeError(
                "prediction must be a BayesianPrediction instance."
            )

        predictions = prediction.predictions
        probabilities = prediction.probabilities
        confidence = prediction.confidence

        if not isinstance(predictions, Tensor):
            raise TypeError(
                "prediction.predictions must be a torch.Tensor."
            )

        if not isinstance(probabilities, Tensor):
            raise TypeError(
                "prediction.probabilities must be a torch.Tensor."
            )

        if not isinstance(confidence, Tensor):
            raise TypeError(
                "prediction.confidence must be a torch.Tensor."
            )

        if predictions.ndim != 1:
            raise ValueError(
                "prediction.predictions must be 1-dimensional."
            )

        if probabilities.ndim != 2:
            raise ValueError(
                "prediction.probabilities must be 2-dimensional."
            )

        if confidence.ndim != 1:
            raise ValueError(
                "prediction.confidence must be 1-dimensional."
            )

        n_samples = predictions.shape[0]

        if n_samples < 1:
            raise ValueError(
                "prediction must contain at least one sample."
            )

        if probabilities.shape[0] != n_samples:
            raise ValueError(
                "predictions and probabilities must contain "
                "the same number of samples."
            )

        if confidence.shape[0] != n_samples:
            raise ValueError(
                "predictions and confidence must contain "
                "the same number of samples."
            )

        if probabilities.shape[1] < 2:
            raise ValueError(
                "prediction.probabilities must contain at least "
                "two regimes."
            )

        tensors = (
            predictions,
            probabilities,
            confidence,
            prediction.uncertainty.predictive_entropy,
            prediction.uncertainty.expected_entropy,
            prediction.uncertainty.mutual_information,
            prediction.uncertainty.probability_std,
            prediction.uncertainty.epistemic_uncertainty,
        )

        for tensor in tensors:
            if not isinstance(tensor, Tensor):
                raise TypeError(
                    "All Bayesian prediction values must be "
                    "torch.Tensor instances."
                )

            if not torch.isfinite(tensor).all():
                raise ValueError(
                    "Bayesian prediction values must be finite."
                )

        if torch.any(confidence < 0) or torch.any(
            confidence > 1
        ):
            raise ValueError(
                "prediction.confidence must be within [0, 1]."
            )

        if torch.any(probabilities < 0) or torch.any(
            probabilities > 1
        ):
            raise ValueError(
                "prediction.probabilities must be within [0, 1]."
            )

        row_sums = probabilities.sum(dim=1)

        if not torch.allclose(
            row_sums,
            torch.ones_like(row_sums),
            atol=1e-5,
            rtol=1e-5,
        ):
            raise ValueError(
                "Each probability row must sum to 1."
            )

        if torch.any(predictions < 0):
            raise ValueError(
                "prediction.predictions must not be negative."
            )

        if torch.any(
            predictions >= probabilities.shape[1]
        ):
            raise ValueError(
                "prediction.predictions contains an invalid "
                "regime index."
            )

        uncertainty_tensors = (
            prediction.uncertainty.predictive_entropy,
            prediction.uncertainty.expected_entropy,
            prediction.uncertainty.mutual_information,
            prediction.uncertainty.probability_std,
            prediction.uncertainty.epistemic_uncertainty,
        )

        for tensor in uncertainty_tensors:
            if tensor.ndim != 1:
                raise ValueError(
                    "Each uncertainty metric must be "
                    "1-dimensional."
                )

            if tensor.shape[0] != n_samples:
                raise ValueError(
                    "Each uncertainty metric must contain one "
                    "value per sample."
                )

        return prediction

    def analyze(
        self,
        prediction: BayesianPrediction,
    ) -> BayesianDiagnosticsResult:
        """
        Calculate aggregate diagnostics for a Bayesian prediction.
        """
        prediction = self._validate_prediction(
            prediction
        )

        n_samples = prediction.predictions.shape[0]
        n_regimes = prediction.probabilities.shape[1]

        regime_counts = torch.bincount(
            prediction.predictions.long(),
            minlength=n_regimes,
        )

        regime_frequencies = (
            regime_counts.float() / float(n_samples)
        )

        low_confidence_mask = (
            prediction.confidence
            < self.confidence_threshold
        )

        low_confidence_count = int(
            low_confidence_mask.sum().item()
        )

        low_confidence_fraction = (
            float(low_confidence_count) / float(n_samples)
        )

        return BayesianDiagnosticsResult(
            n_samples=n_samples,
            n_regimes=n_regimes,
            regime_counts=regime_counts,
            regime_frequencies=regime_frequencies,
            mean_confidence=float(
                prediction.confidence.mean().item()
            ),
            mean_predictive_entropy=float(
                prediction.uncertainty.predictive_entropy
                .mean()
                .item()
            ),
            mean_expected_entropy=float(
                prediction.uncertainty.expected_entropy
                .mean()
                .item()
            ),
            mean_mutual_information=float(
                prediction.uncertainty.mutual_information
                .mean()
                .item()
            ),
            mean_epistemic_uncertainty=float(
                prediction.uncertainty.epistemic_uncertainty
                .mean()
                .item()
            ),
            mean_probability_std=float(
                prediction.uncertainty.probability_std
                .mean()
                .item()
            ),
            low_confidence_count=low_confidence_count,
            low_confidence_fraction=low_confidence_fraction,
            low_confidence_mask=low_confidence_mask.clone(),
        )

    def summary(
        self,
        prediction: BayesianPrediction,
    ) -> dict[str, Any]:
        """
        Return diagnostics as plain Python-friendly metadata.
        """
        result = self.analyze(prediction)

        return {
            "n_samples": result.n_samples,
            "n_regimes": result.n_regimes,
            "regime_counts": (
                result.regime_counts.tolist()
            ),
            "regime_frequencies": (
                result.regime_frequencies.tolist()
            ),
            "mean_confidence": result.mean_confidence,
            "mean_predictive_entropy": (
                result.mean_predictive_entropy
            ),
            "mean_expected_entropy": (
                result.mean_expected_entropy
            ),
            "mean_mutual_information": (
                result.mean_mutual_information
            ),
            "mean_epistemic_uncertainty": (
                result.mean_epistemic_uncertainty
            ),
            "mean_probability_std": (
                result.mean_probability_std
            ),
            "low_confidence_count": (
                result.low_confidence_count
            ),
            "low_confidence_fraction": (
                result.low_confidence_fraction
            ),
        }