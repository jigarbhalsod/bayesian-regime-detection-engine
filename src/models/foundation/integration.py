from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import Tensor

from .adapter import FoundationModelAdapter
from .config import FoundationModelConfig
from .sequence import SequenceDataset, SequencePreparer


@dataclass(frozen=True)
class FoundationIntegrationResult:
    """
    Result produced by the end-to-end foundation pipeline.
    """

    dataset: SequenceDataset
    predictions: Tensor
    metadata: dict[str, Any]


class FoundationModelIntegration:
    """
    End-to-end integration layer for the foundation model pipeline.

    Responsibilities
    ----------------
    1. Validate chronological feature data.
    2. Prepare sliding temporal sequences.
    3. Train the foundation model adapter.
    4. Generate forecasts.
    5. Return predictions and integration metadata.
    """

    def __init__(
        self,
        config: FoundationModelConfig | None = None,
        hidden_dim: int = 64,
        stride: int = 1,
    ) -> None:
        if config is None:
            config = FoundationModelConfig()

        if not isinstance(config, FoundationModelConfig):
            raise TypeError(
                "config must be a FoundationModelConfig instance."
            )

        self.config = config

        self.preparer = SequencePreparer(
            context_length=config.context_length,
            forecast_horizon=config.forecast_horizon,
            n_features=config.n_features,
            output_dim=config.output_dim,
            stride=stride,
        )

        self.model = FoundationModelAdapter(
            config=config,
            hidden_dim=hidden_dim,
        )

    @property
    def is_fitted(self) -> bool:
        """Return whether the underlying model is fitted."""
        return self.model.is_fitted

    def prepare(
        self,
        data: Tensor,
        targets: Tensor | None = None,
    ) -> SequenceDataset:
        """
        Prepare chronological observations as temporal sequences.
        """
        return self.preparer.prepare(
            data=data,
            targets=targets,
        )

    def fit(
        self,
        data: Tensor,
        targets: Tensor | None = None,
        epochs: int = 100,
        learning_rate: float = 1e-3,
        **kwargs: Any,
    ) -> "FoundationModelIntegration":
        """
        Prepare sequences and train the underlying foundation model.
        """
        dataset = self.prepare(
            data=data,
            targets=targets,
        )

        self.model.fit(
            dataset.inputs,
            dataset.targets,
            epochs=epochs,
            learning_rate=learning_rate,
            **kwargs,
        )

        return self

    def predict(
        self,
        data: Tensor,
        **kwargs: Any,
    ) -> Tensor:
        """
        Generate forecasts from already prepared temporal input.

        Expected shape:
            (batch_size, context_length, n_features)
        """
        return self.model.predict(
            data,
            **kwargs,
        )

    def fit_predict(
        self,
        data: Tensor,
        targets: Tensor | None = None,
        epochs: int = 100,
        learning_rate: float = 1e-3,
        **kwargs: Any,
    ) -> FoundationIntegrationResult:
        """
        Execute the complete pipeline:

        data -> sequence preparation -> training -> prediction.
        """
        dataset = self.prepare(
            data=data,
            targets=targets,
        )

        self.model.fit(
            dataset.inputs,
            dataset.targets,
            epochs=epochs,
            learning_rate=learning_rate,
            **kwargs,
        )

        predictions = self.model.predict(
            dataset.inputs,
        )

        return FoundationIntegrationResult(
            dataset=dataset,
            predictions=predictions,
            metadata=self.get_metadata(),
        )

    def get_metadata(self) -> dict[str, Any]:
        """
        Return complete integration metadata.
        """
        return {
            "config": {
                "model_name": self.config.model_name,
                "context_length": self.config.context_length,
                "forecast_horizon": self.config.forecast_horizon,
                "n_features": self.config.n_features,
                "output_dim": self.config.output_dim,
            },
            "sequence": self.preparer.get_metadata(),
            "model": self.model.get_metadata(),
            "is_fitted": self.is_fitted,
        }