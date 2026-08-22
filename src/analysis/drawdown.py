from __future__ import annotations

import math
from typing import Any, Sequence

from src.analysis.base import BaseFinancialAnalyzer
from src.analysis.result import AnalysisResult


class DrawdownAnalyzer(BaseFinancialAnalyzer):
    """
    Calculates cumulative wealth, running peaks, drawdowns,
    drawdown durations, and recovery information.
    """

    @property
    def name(self) -> str:
        return "drawdown"

    def analyze(
        self,
        records: Sequence[dict[str, Any]],
    ) -> AnalysisResult:
        """
        Analyze drawdowns from sequential return data.
        """

        output_records = [dict(record) for record in records]

        metadata = {
            "return_column": self.config.return_column,
            "drawdown_threshold": self.config.drawdown_threshold,
            "valid_return_count": 0,
        }

        if not records:
            return AnalysisResult(
                records=output_records,
                metrics={},
                metric_names=[],
                metadata=metadata,
            )

        wealth = 1.0
        peak = 1.0

        valid_return_count = 0
        drawdowns: list[float] = []

        drawdown_duration = 0
        maximum_drawdown_duration = 0

        for index, record in enumerate(records):
            value = record.get(self.config.return_column)

            try:
                return_value = float(value)
            except (TypeError, ValueError):
                return_value = None

            if (
                return_value is None
                or not math.isfinite(return_value)
            ):
                output_records[index]["analysis__cumulative_wealth"] = None
                output_records[index]["analysis__running_peak"] = None
                output_records[index]["analysis__drawdown"] = None
                output_records[index]["analysis__drawdown_duration"] = None
                output_records[index]["analysis__recovered"] = None
                continue

            valid_return_count += 1

            previous_peak = peak

            wealth *= 1.0 + return_value
            peak = max(peak, wealth)

            drawdown = (wealth / peak) - 1.0

            # ----------------------------------------------------------
            # Drawdown duration
            # ----------------------------------------------------------

            recovered = False

            if wealth >= previous_peak:
                drawdown_duration = 0
                recovered = True
            elif drawdown < 0.0:
                drawdown_duration += 1
            else:
                drawdown_duration = 0

            maximum_drawdown_duration = max(
                maximum_drawdown_duration,
                drawdown_duration,
            )

            # ----------------------------------------------------------
            # Store record-level analysis
            # ----------------------------------------------------------

            output_records[index]["analysis__cumulative_wealth"] = wealth
            output_records[index]["analysis__running_peak"] = peak
            output_records[index]["analysis__drawdown"] = drawdown
            output_records[index][
                "analysis__drawdown_duration"
            ] = drawdown_duration
            output_records[index]["analysis__recovered"] = recovered

            drawdowns.append(drawdown)

        metadata["valid_return_count"] = valid_return_count

        if not drawdowns:
            return AnalysisResult(
                records=output_records,
                metrics={},
                metric_names=[],
                metadata=metadata,
            )

        maximum_drawdown = min(drawdowns)

        threshold_breach_count = sum(
            drawdown <= self.config.drawdown_threshold
            for drawdown in drawdowns
        )

        metrics = {
            "analysis__maximum_drawdown": maximum_drawdown,
            "analysis__drawdown_threshold_breach_count": (
                threshold_breach_count
            ),
            "analysis__maximum_drawdown_duration": (
                maximum_drawdown_duration
            ),
        }

        return AnalysisResult(
            records=output_records,
            metrics=metrics,
            metric_names=list(metrics.keys()),
            metadata=metadata,
        )