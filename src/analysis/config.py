from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class AnalysisConfig:
    """
    Central configuration for Phase 7 financial analysis.
    """

    # ------------------------------------------------------------------
    # Source columns
    # ------------------------------------------------------------------

    return_column: str = "feature__return_1d"

    # ------------------------------------------------------------------
    # Annualization and risk
    # ------------------------------------------------------------------

    risk_free_rate: float = 0.0
    annualization_factor: int = 252

    # ------------------------------------------------------------------
    # Return analysis
    # ------------------------------------------------------------------

    return_periods: Sequence[int] = field(
        default_factory=lambda: (1, 5, 20)
    )

    rolling_windows: Sequence[int] = field(
        default_factory=lambda: (5, 20, 60)
    )

    # ------------------------------------------------------------------
    # Volatility analysis
    # ------------------------------------------------------------------

    volatility_windows: Sequence[int] = field(
        default_factory=lambda: (5, 20, 60)
    )

    # ------------------------------------------------------------------
    # Drawdown analysis
    # ------------------------------------------------------------------

    drawdown_threshold: float = -0.10

    # ------------------------------------------------------------------
    # Cross-asset analysis
    # ------------------------------------------------------------------

    correlation_windows: Sequence[int] = field(
        default_factory=lambda: (5, 20, 60)
    )

    # ------------------------------------------------------------------
    # Reserved for later Phase 7 groups
    # ------------------------------------------------------------------

    stress_enabled: bool = True