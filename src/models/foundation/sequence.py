from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


@dataclass(frozen=True)
class SequenceDataset:
    """
    Prepared temporal sequences for foundation-model training.

    inputs shape:
        (n_sequences, context_length, n_features)

    targets shape:
        (n_sequences, forecast_horizon, output_dim)
    """

    inputs: Tensor
    targets: Tensor


class SequencePreparer:
    """
    Convert chronological observations into sliding context and
    forecast windows.

    Input data shape:
        (n_observations, n_features)

    Target data shape, when provided:
        (n_observations, output_dim)

    If targets are not provided, the first output_dim columns of
    data are used as forecasting targets.
    """

    def __init__(
        self,
        context_length: int,
        forecast_horizon: int,
        n_features: int,
        output_dim: int = 1,
        stride: int = 1,
    ) -> None:
        self.context_length = self._validate_positive_int(
            context_length,
            "context_length",
        )
        self.forecast_horizon = self._validate_positive_int(
            forecast_horizon,
            "forecast_horizon",
        )
        self.n_features = self._validate_positive_int(
            n_features,
            "n_features",
        )
        self.output_dim = self._validate_positive_int(
            output_dim,
            "output_dim",
        )
        self.stride = self._validate_positive_int(
            stride,
            "stride",
        )

        if self.output_dim > self.n_features:
            raise ValueError(
                "output_dim cannot be greater than n_features "
                "when default targets are derived from data."
            )

    @staticmethod
    def _validate_positive_int(
        value: Any,
        field_name: str,
    ) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(
                f"{field_name} must be an integer."
            )

        if value < 1:
            raise ValueError(
                f"{field_name} must be greater than or equal to 1."
            )

        return value

    def _validate_data(
        self,
        data: Tensor,
    ) -> Tensor:
        if not isinstance(data, Tensor):
            raise TypeError(
                "data must be a torch.Tensor."
            )

        if data.ndim != 2:
            raise ValueError(
                "data must be a 2-dimensional tensor with shape "
                "(n_observations, n_features)."
            )

        if data.shape[0] < 1:
            raise ValueError(
                "data must contain at least one observation."
            )

        if data.shape[1] != self.n_features:
            raise ValueError(
                f"Expected n_features={self.n_features}, "
                f"but received {data.shape[1]}."
            )

        if not torch.isfinite(data).all():
            raise ValueError(
                "data must contain only finite values."
            )

        return data.float()

    def _validate_targets(
        self,
        targets: Tensor,
        n_observations: int,
    ) -> Tensor:
        if not isinstance(targets, Tensor):
            raise TypeError(
                "targets must be a torch.Tensor."
            )

        if targets.ndim != 2:
            raise ValueError(
                "targets must be a 2-dimensional tensor with shape "
                "(n_observations, output_dim)."
            )

        if targets.shape[0] != n_observations:
            raise ValueError(
                "data and targets must contain the same number "
                "of observations."
            )

        if targets.shape[1] != self.output_dim:
            raise ValueError(
                f"Expected output_dim={self.output_dim}, "
                f"but received {targets.shape[1]}."
            )

        if not torch.isfinite(targets).all():
            raise ValueError(
                "targets must contain only finite values."
            )

        return targets.float()

    def _minimum_observations(self) -> int:
        return (
            self.context_length
            + self.forecast_horizon
        )

    def prepare(
        self,
        data: Tensor,
        targets: Tensor | None = None,
    ) -> SequenceDataset:
        """
        Build chronological sliding windows.

        For each sequence starting at index i:

            input:
                data[i : i + context_length]

            target:
                target_data[
                    i + context_length :
                    i + context_length + forecast_horizon
                ]
        """
        data = self._validate_data(data)

        n_observations = data.shape[0]

        if targets is None:
            target_data = data[:, : self.output_dim]
        else:
            target_data = self._validate_targets(
                targets,
                n_observations,
            )

        minimum = self._minimum_observations()

        if n_observations < minimum:
            raise ValueError(
                f"At least {minimum} observations are required, "
                f"but received {n_observations}."
            )

        max_start = (
            n_observations
            - self.context_length
            - self.forecast_horizon
        )

        start_indices = range(
            0,
            max_start + 1,
            self.stride,
        )

        inputs = []
        output_targets = []

        for start in start_indices:
            context_end = start + self.context_length
            forecast_end = (
                context_end + self.forecast_horizon
            )

            inputs.append(
                data[start:context_end]
            )

            output_targets.append(
                target_data[context_end:forecast_end]
            )

        return SequenceDataset(
            inputs=torch.stack(inputs),
            targets=torch.stack(output_targets),
        )

    def get_metadata(self) -> dict[str, int]:
        """Return sequence preparation configuration."""
        return {
            "context_length": self.context_length,
            "forecast_horizon": self.forecast_horizon,
            "n_features": self.n_features,
            "output_dim": self.output_dim,
            "stride": self.stride,
        }