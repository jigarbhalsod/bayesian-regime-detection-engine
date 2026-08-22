from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from src.analysis.base import BaseFinancialAnalyzer
from src.analysis.result import AnalysisResult


class RiskAnalyzer(BaseFinancialAnalyzer):
    """
    Calculates core risk, volatility, and risk-adjusted metrics
    from return data.
    """

    @property
    def name(self) -> str:
        return "risk"

    def analyze(
        self,
        records: Sequence[dict[str, Any]],
    ) -> AnalysisResult:
        """
        Analyze return-based risk metrics, rolling volatility,
        and risk-adjusted performance metrics.
        """

        output_records = [dict(record) for record in records]

        valid_returns: list[float] = []
        parsed_returns: list[float | None] = []

        # --------------------------------------------------------------
        # Parse and validate returns
        # --------------------------------------------------------------

        for record in records:
            value = record.get(self.config.return_column)

            if value is None:
                parsed_returns.append(None)
                continue

            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                parsed_returns.append(None)
                continue

            if not math.isfinite(numeric_value):
                parsed_returns.append(None)
                continue

            parsed_returns.append(numeric_value)
            valid_returns.append(numeric_value)

        metadata = {
            "valid_return_count": len(valid_returns),
            "return_column": self.config.return_column,
            "volatility_windows": list(
                self.config.volatility_windows
            ),
        }

        # --------------------------------------------------------------
        # Add rolling volatility columns
        # --------------------------------------------------------------

        for window in self.config.volatility_windows:
            column = f"analysis__rolling_volatility_{window}"

            for index in range(len(output_records)):
                if index < window - 1:
                    output_records[index][column] = None
                    continue

                window_values = parsed_returns[
                    index - window + 1:index + 1
                ]

                # Every value in the window must be valid.
                if any(value is None for value in window_values):
                    output_records[index][column] = None
                    continue

                output_records[index][column] = float(
                    np.std(
                        window_values,
                        ddof=1,
                    )
                )

        # --------------------------------------------------------------
        # Return early if insufficient valid data
        # --------------------------------------------------------------

        if len(valid_returns) < 2:
            return AnalysisResult(
                records=output_records,
                metrics={},
                metric_names=[],
                metadata=metadata,
            )

        returns = np.asarray(valid_returns, dtype=float)

        # --------------------------------------------------------------
        # Core volatility metrics
        # --------------------------------------------------------------

        volatility = float(
            np.std(
                returns,
                ddof=1,
            )
        )

        annualized_volatility = float(
            volatility
            * math.sqrt(self.config.annualization_factor)
        )

        downside_returns = returns[returns < 0.0]

        if len(downside_returns) >= 2:
            downside_volatility = float(
                np.std(
                    downside_returns,
                    ddof=1,
                )
            )
        else:
            downside_volatility = 0.0

        # --------------------------------------------------------------
        # Risk-adjusted metrics
        # --------------------------------------------------------------

        daily_risk_free_rate = (
            self.config.risk_free_rate
            / self.config.annualization_factor
        )

        excess_returns = returns - daily_risk_free_rate

        mean_excess_return = float(
            np.mean(excess_returns)
        )

        if volatility > 0.0:
            sharpe_ratio = float(
                mean_excess_return
                / volatility
                * math.sqrt(
                    self.config.annualization_factor
                )
            )
        else:
            sharpe_ratio = 0.0

        downside_excess_returns = excess_returns[
            excess_returns < 0.0
        ]

        if len(downside_excess_returns) >= 2:
            downside_deviation = float(
                np.std(
                    downside_excess_returns,
                    ddof=1,
                )
            )

            if downside_deviation > 0.0:
                sortino_ratio = float(
                    mean_excess_return
                    / downside_deviation
                    * math.sqrt(
                        self.config.annualization_factor
                    )
                )
            else:
                sortino_ratio = 0.0
        else:
            sortino_ratio = 0.0

        # --------------------------------------------------------------
        # Final metrics
        # --------------------------------------------------------------

        metrics = {
            "analysis__volatility": volatility,
            "analysis__annualized_volatility": annualized_volatility,
            "analysis__downside_volatility": downside_volatility,
            "analysis__sharpe_ratio": sharpe_ratio,
            "analysis__sortino_ratio": sortino_ratio,
        }

        return AnalysisResult(
            records=output_records,
            metrics=metrics,
            metric_names=list(metrics.keys()),
            metadata=metadata,
        )

def test_risk_analyzer_calculates_sharpe_ratio() -> None:
    analyzer = RiskAnalyzer()

    records = [
        {"feature__return_1d": 0.01},
        {"feature__return_1d": -0.02},
        {"feature__return_1d": 0.03},
        {"feature__return_1d": -0.01},
    ]

    result = analyzer.analyze(records)

    returns = np.asarray(
        [0.01, -0.02, 0.03, -0.01],
        dtype=float,
    )

    expected_volatility = float(
        np.std(returns, ddof=1)
    )

    expected_sharpe = float(
        np.mean(returns)
        / expected_volatility
        * math.sqrt(analyzer.config.annualization_factor)
    )

    assert result.metrics["analysis__sharpe_ratio"] == pytest.approx(
        expected_sharpe
    )


def test_risk_analyzer_calculates_sortino_ratio() -> None:
    analyzer = RiskAnalyzer()

    records = [
        {"feature__return_1d": 0.01},
        {"feature__return_1d": -0.02},
        {"feature__return_1d": 0.03},
        {"feature__return_1d": -0.01},
    ]

    result = analyzer.analyze(records)

    returns = np.asarray(
        [0.01, -0.02, 0.03, -0.01],
        dtype=float,
    )

    downside_returns = returns[returns < 0.0]

    downside_deviation = float(
        np.std(downside_returns, ddof=1)
    )

    expected_sortino = float(
        np.mean(returns)
        / downside_deviation
        * math.sqrt(analyzer.config.annualization_factor)
    )

    assert result.metrics["analysis__sortino_ratio"] == pytest.approx(
        expected_sortino
    )


def test_risk_analyzer_uses_risk_free_rate() -> None:
    config = AnalysisConfig(
        risk_free_rate=0.05,
    )

    analyzer = RiskAnalyzer(config=config)

    records = [
        {"feature__return_1d": 0.01},
        {"feature__return_1d": -0.02},
        {"feature__return_1d": 0.03},
        {"feature__return_1d": -0.01},
    ]

    result = analyzer.analyze(records)

    returns = np.asarray(
        [0.01, -0.02, 0.03, -0.01],
        dtype=float,
    )

    daily_risk_free_rate = (
        config.risk_free_rate
        / config.annualization_factor
    )

    excess_returns = returns - daily_risk_free_rate

    expected_sharpe = float(
        np.mean(excess_returns)
        / np.std(returns, ddof=1)
        * math.sqrt(config.annualization_factor)
    )

    assert result.metrics["analysis__sharpe_ratio"] == pytest.approx(
        expected_sharpe
    )


def test_risk_analyzer_handles_zero_volatility() -> None:
    analyzer = RiskAnalyzer()

    records = [
        {"feature__return_1d": 0.01},
        {"feature__return_1d": 0.01},
        {"feature__return_1d": 0.01},
    ]

    result = analyzer.analyze(records)

    assert result.metrics["analysis__volatility"] == 0.0
    assert result.metrics["analysis__sharpe_ratio"] == 0.0
    assert result.metrics["analysis__sortino_ratio"] == 0.0