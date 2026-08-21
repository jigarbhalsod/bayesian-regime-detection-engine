"""Configuration for Phase 6 feature engineering."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureConfig:
    """Configuration for all Phase 6 feature transformers."""

    # ------------------------------------------------------------------
    # Phase 5 compatible source columns
    # ------------------------------------------------------------------
    target_close_column: str = "nifty_50__close"
    target_returns_column: str = "nifty_50__returns"
    target_volume_column: str = "nifty_50__volume"

    # ------------------------------------------------------------------
    # Return and price features
    #
    # return_periods is the compatibility-facing configuration used by
    # the existing Phase 6 tests.
    # ------------------------------------------------------------------
    return_periods: tuple[int, ...] = (
        1,
        5,
        10,
        20,
    )

    # Optional integration-facing aliases.
    price_periods: tuple[int, ...] | None = None
    price_change_periods: tuple[int, ...] | None = None
    price_rolling_periods: tuple[int, ...] | None = None

    # ------------------------------------------------------------------
    # Trend and momentum features
    # ------------------------------------------------------------------
    trend_periods: tuple[int, ...] = (
        5,
        10,
        20,
        50,
    )

    sma_periods: tuple[int, ...] | None = None
    ema_periods: tuple[int, ...] | None = None

    momentum_periods: tuple[int, ...] = (
        1,
        5,
        10,
        20,
    )

    # ------------------------------------------------------------------
    # Volatility features
    # ------------------------------------------------------------------
    volatility_periods: tuple[int, ...] = (
        5,
        10,
        20,
    )

    # ------------------------------------------------------------------
    # Volume and liquidity features
    # ------------------------------------------------------------------
    volume_periods: tuple[int, ...] = (
        5,
        10,
        20,
    )

    # ------------------------------------------------------------------
    # Technical indicators
    # ------------------------------------------------------------------
    technical_periods: tuple[int, ...] = (
        14,
    )

    bollinger_periods: tuple[int, ...] = (
        20,
    )

    rsi_period: int | None = None

    macd_fast_period: int = 12
    macd_slow_period: int = 26
    macd_signal_period: int = 9

    bollinger_period: int | None = None
    bollinger_std_multiplier: float = 2.0

    # ------------------------------------------------------------------
    # Cross-asset features
    # ------------------------------------------------------------------
    cross_asset_columns: tuple[str, ...] = (
        "bank_nifty__close",
        "india_vix__close",
    )

    cross_asset_periods: tuple[int, ...] = (
        1,
    )

    cross_asset_return_period: int | None = None

    cross_asset_correlation_periods: tuple[int, ...] = (
        5,
        10,
        20,
    )

    # ------------------------------------------------------------------
    # Macro features
    # ------------------------------------------------------------------
    macro_columns: tuple[str, ...] = (
        "repo_rate",
        "cpi_inflation",
        "usd_inr",
    )

    macro_periods: tuple[int, ...] = (
        1,
        3,
    )

    macro_change_periods: tuple[int, ...] | None = None

    macro_rolling_periods: tuple[int, ...] = (
        3,
        6,
        12,
    )

    # ------------------------------------------------------------------
    # Resolved compatibility properties
    # ------------------------------------------------------------------
    @property
    def resolved_price_change_periods(
        self,
    ) -> tuple[int, ...]:
        """Return the configured price change periods."""

        if self.price_change_periods is not None:
            return self.price_change_periods

        if self.price_periods is not None:
            return self.price_periods

        return self.return_periods

    @property
    def resolved_price_rolling_periods(
        self,
    ) -> tuple[int, ...]:
        """Return the configured rolling price periods."""

        if self.price_rolling_periods is not None:
            return self.price_rolling_periods

        if self.price_periods is not None:
            return self.price_periods

        return self.return_periods

    @property
    def resolved_sma_periods(
        self,
    ) -> tuple[int, ...]:
        """Return configured SMA periods."""

        if self.sma_periods is not None:
            return self.sma_periods

        return self.trend_periods

    @property
    def resolved_ema_periods(
        self,
    ) -> tuple[int, ...]:
        """Return configured EMA periods."""

        if self.ema_periods is not None:
            return self.ema_periods

        return self.trend_periods

    @property
    def resolved_rsi_period(
        self,
    ) -> int:
        """Return the configured RSI period."""

        if self.rsi_period is not None:
            return self.rsi_period

        return self.technical_periods[0]

    @property
    def resolved_bollinger_period(
        self,
    ) -> int:
        """Return the configured Bollinger period."""

        if self.bollinger_period is not None:
            return self.bollinger_period

        return self.bollinger_periods[0]

    @property
    def resolved_cross_asset_return_period(
        self,
    ) -> int:
        """Return the configured cross-asset return period."""

        if self.cross_asset_return_period is not None:
            return self.cross_asset_return_period

        return self.cross_asset_periods[0]

    @property
    def resolved_macro_change_periods(
        self,
    ) -> tuple[int, ...]:
        """Return configured macro change periods."""

        if self.macro_change_periods is not None:
            return self.macro_change_periods

        return self.macro_periods