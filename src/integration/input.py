from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class AdvancedModelInput:
    """
    Shared input contract for all advanced model families.

    Parameters
    ----------
    features:
        Sequence of feature observations supplied to models.
    timestamps:
        Optional timestamps aligned with feature observations.
    target:
        Optional target values or target metadata.
    regime_context:
        Optional context describing the current market regime.
    metadata:
        Additional immutable-compatible metadata for the execution.
    """

    features: Sequence[Any]
    timestamps: Sequence[Any] | None = None
    target: Any | None = None
    regime_context: Any | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.features, (str, bytes)):
            raise TypeError(
                "features must be a sequence of observations."
            )

        if not isinstance(self.features, Sequence):
            raise TypeError(
                "features must be a sequence."
            )

        if len(self.features) == 0:
            raise ValueError(
                "features cannot be empty."
            )

        if self.timestamps is not None:
            if isinstance(
                self.timestamps,
                (str, bytes),
            ):
                raise TypeError(
                    "timestamps must be a sequence or None."
                )

            if not isinstance(
                self.timestamps,
                Sequence,
            ):
                raise TypeError(
                    "timestamps must be a sequence or None."
                )

            if len(self.timestamps) != len(
                self.features
            ):
                raise ValueError(
                    "timestamps length must match "
                    "features length."
                )

        if not isinstance(
            self.metadata,
            Mapping,
        ):
            raise TypeError(
                "metadata must be a mapping."
            )

        object.__setattr__(
            self,
            "metadata",
            dict(self.metadata),
        )

    @property
    def n_observations(self) -> int:
        """
        Return the number of feature observations.
        """
        return len(self.features)

    @property
    def has_timestamps(self) -> bool:
        """
        Return whether timestamps were supplied.
        """
        return self.timestamps is not None

    @property
    def has_target(self) -> bool:
        """
        Return whether target information was supplied.
        """
        return self.target is not None

    @property
    def has_regime_context(self) -> bool:
        """
        Return whether regime context was supplied.
        """
        return self.regime_context is not None

    def to_dict(self) -> dict[str, Any]:
        """
        Return the shared model input as a dictionary.
        """
        return {
            "features": self.features,
            "timestamps": self.timestamps,
            "target": self.target,
            "regime_context": self.regime_context,
            "metadata": dict(self.metadata),
        }