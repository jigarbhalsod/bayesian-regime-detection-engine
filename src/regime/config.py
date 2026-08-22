from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RegimeConfig:
    """
    Immutable configuration for regime detection.
    """

    n_regimes: int = 3
    feature_columns: tuple[str, ...] = field(default_factory=tuple)

    lookback_window: int | None = None
    min_samples: int = 30

    random_state: int | None = 42

    model_parameters: dict[str, Any] = field(default_factory=dict)