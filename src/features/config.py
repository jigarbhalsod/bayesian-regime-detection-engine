"""Configuration for the Phase 6 feature engineering pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FeatureConfig:
    """Configuration shared across feature engineering components."""

    date_column: str = "date"
    target_dataset: str = "nifty_50"
    feature_separator: str = "__"

    return_periods: tuple[int, ...] = (1, 5, 10, 20)
    momentum_periods: tuple[int, ...] = (5, 10, 20)
    sma_periods: tuple[int, ...] = (5, 10, 20, 50)
    ema_periods: tuple[int, ...] = (5, 10, 20, 50)

    enabled_feature_groups: tuple[str, ...] = field(
        default_factory=lambda: (
            "returns",
            "price",
            "momentum",
        )
    )

    @property
    def target_close_column(self) -> str:
        """Return the integrated close-price column for the target dataset."""
        return (
            f"{self.target_dataset}"
            f"{self.feature_separator}"
            f"close"
        )

    @property
    def target_returns_column(self) -> str:
        """Return the integrated returns column for the target dataset."""
        return (
            f"{self.target_dataset}"
            f"{self.feature_separator}"
            f"returns"
        )