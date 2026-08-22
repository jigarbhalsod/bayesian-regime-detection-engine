from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from src.analysis.base import BaseFinancialAnalyzer
from src.analysis.result import AnalysisResult


class CrossAssetAnalyzer(BaseFinancialAnalyzer):
    """
    Calculates correlations between the primary return series and
    other asset return series.
    """

    @property
    def name(self) -> str:
        return "cross_asset"

    def analyze(
        self,
        records: Sequence[dict[str, Any]],
    ) -> AnalysisResult:
        output_records = [dict(record) for record in records]

        metadata = {
            "correlation_windows": list(
                self.config.correlation_windows
            ),
            "valid_pair_count": 0,
        }

        if not records:
            return AnalysisResult(
                records=[],
                metrics={},
                metric_names=[],
                metadata=metadata,
            )

        primary_column = self.config.return_column

        all_columns: list[str] = []
        for record in records:
            for column in record:
                if column not in all_columns:
                    all_columns.append(column)

        asset_columns = [
            column
            for column in all_columns
            if column != primary_column
            and "return" in column.lower()
        ]

        if primary_column not in all_columns or not asset_columns:
            return AnalysisResult(
                records=output_records,
                metrics={},
                metric_names=[],
                metadata=metadata,
            )

        metrics: dict[str, float] = {}
        max_valid_pair_count = 0

        for asset_column in asset_columns:
            valid_primary: list[float] = []
            valid_asset: list[float] = []

            parsed_pairs: list[
                tuple[float | None, float | None]
            ] = []

            for record in records:
                primary_value = self._parse_numeric(
                    record.get(primary_column)
                )
                asset_value = self._parse_numeric(
                    record.get(asset_column)
                )

                parsed_pairs.append(
                    (primary_value, asset_value)
                )

                if (
                    primary_value is not None
                    and asset_value is not None
                ):
                    valid_primary.append(primary_value)
                    valid_asset.append(asset_value)

            valid_pair_count = len(valid_primary)

            max_valid_pair_count = max(
                max_valid_pair_count,
                valid_pair_count,
            )

            correlation_key = (
                f"analysis__correlation__"
                f"{primary_column}__{asset_column}"
            )

            if valid_pair_count >= 2:
                primary_array = np.asarray(
                    valid_primary,
                    dtype=float,
                )
                asset_array = np.asarray(
                    valid_asset,
                    dtype=float,
                )

                if (
                    np.std(primary_array) > 0.0
                    and np.std(asset_array) > 0.0
                ):
                    metrics[correlation_key] = float(
                        np.corrcoef(
                            primary_array,
                            asset_array,
                        )[0, 1]
                    )

            # Add rolling correlations only when at least one
            # complete window can exist.
            for window in self.config.correlation_windows:
                if len(output_records) < window:
                    continue

                rolling_column = (
                    f"analysis__rolling_correlation_{window}__"
                    f"{primary_column}__{asset_column}"
                )

                for index in range(len(output_records)):
                    if index < window - 1:
                        output_records[index][rolling_column] = None
                        continue

                    window_pairs = parsed_pairs[
                        index - window + 1:index + 1
                    ]

                    if any(
                        primary is None or asset is None
                        for primary, asset in window_pairs
                    ):
                        output_records[index][rolling_column] = None
                        continue

                    window_primary = np.asarray(
                        [
                            primary
                            for primary, _ in window_pairs
                        ],
                        dtype=float,
                    )

                    window_asset = np.asarray(
                        [
                            asset
                            for _, asset in window_pairs
                        ],
                        dtype=float,
                    )

                    if (
                        np.std(window_primary) == 0.0
                        or np.std(window_asset) == 0.0
                    ):
                        output_records[index][rolling_column] = None
                        continue

                    output_records[index][rolling_column] = float(
                        np.corrcoef(
                            window_primary,
                            window_asset,
                        )[0, 1]
                    )

        metadata["valid_pair_count"] = max_valid_pair_count

        return AnalysisResult(
            records=output_records,
            metrics=metrics,
            metric_names=list(metrics.keys()),
            metadata=metadata,
        )

    @staticmethod
    def _parse_numeric(
        value: Any,
    ) -> float | None:
        if value is None:
            return None

        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return None

        if not math.isfinite(numeric_value):
            return None

        return numeric_value