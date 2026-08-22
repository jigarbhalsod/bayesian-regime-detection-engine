import math
from typing import Any

import numpy as np
from sklearn.cluster import KMeans

from .base import BaseRegimeDetector
from .config import RegimeConfig
from .result import RegimeResult


class ClusteringRegimeDetector(BaseRegimeDetector):
    """
    Unsupervised K-Means baseline for regime discovery.
    """

    def __init__(
        self,
        config: RegimeConfig | None = None,
    ) -> None:
        super().__init__(config=config)

        if not self.config.feature_columns:
            raise ValueError(
                "ClusteringRegimeDetector requires at least one "
                "feature column."
            )

    @property
    def name(self) -> str:
        return "clustering"

    def detect(
        self,
        records: list[dict[str, Any]],
    ) -> RegimeResult:
        output_records = [dict(record) for record in records]

        if not output_records:
            return RegimeResult(
                records=[],
                regime_labels=[],
                metadata={
                    "detector": self.name,
                    "feature_columns": self.config.feature_columns,
                    "n_clusters": self.config.n_regimes,
                    "n_valid_records": 0,
                },
            )

        valid_indices: list[int] = []
        feature_rows: list[list[float]] = []

        for index, record in enumerate(output_records):
            row = self._extract_feature_row(record)

            if row is not None:
                valid_indices.append(index)
                feature_rows.append(row)

        labels: list[str] = ["unknown"] * len(output_records)

        if len(feature_rows) < self.config.n_regimes:
            for record, label in zip(output_records, labels):
                record["regime"] = label

            return RegimeResult(
                records=output_records,
                regime_labels=labels,
                metadata={
                    "detector": self.name,
                    "feature_columns": self.config.feature_columns,
                    "n_clusters": self.config.n_regimes,
                    "n_valid_records": len(feature_rows),
                },
            )

        feature_matrix = np.asarray(feature_rows, dtype=float)

        model = KMeans(
            n_clusters=self.config.n_regimes,
            random_state=self.config.random_state,
            **self.config.model_parameters,
        )

        cluster_ids = model.fit_predict(feature_matrix)

        for index, cluster_id in zip(valid_indices, cluster_ids):
            labels[index] = str(int(cluster_id))

        for record, label in zip(output_records, labels):
            record["regime"] = label

        return RegimeResult(
            records=output_records,
            regime_labels=labels,
            metadata={
                "detector": self.name,
                "feature_columns": self.config.feature_columns,
                "n_clusters": self.config.n_regimes,
                "n_valid_records": len(feature_rows),
                "cluster_centers": model.cluster_centers_.tolist(),
            },
        )

    def _extract_feature_row(
        self,
        record: dict[str, Any],
    ) -> list[float] | None:
        row: list[float] = []

        for column in self.config.feature_columns:
            value = self._to_float(record.get(column))

            if value is None:
                return None

            row.append(value)

        return row

    @staticmethod
    def _to_float(value: Any) -> float | None:
        if value is None or isinstance(value, bool):
            return None

        try:
            number = float(value)
        except (TypeError, ValueError):
            return None

        if not math.isfinite(number):
            return None

        return number