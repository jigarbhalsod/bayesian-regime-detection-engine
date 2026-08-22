from typing import Any


class RegimeLabelMapper:
    """
    Maps raw cluster IDs to interpretable financial regime labels.

    Clusters are ranked by the selected score column, where lower values
    represent weaker market conditions and higher values represent stronger
    market conditions.
    """

    def __init__(
        self,
        score_column: str = "return",
    ) -> None:
        self.score_column = score_column

    def build_mapping(
        self,
        records: list[dict[str, Any]],
        cluster_labels: list[str | int],
    ) -> dict[str, str]:
        """
        Build a deterministic cluster-to-regime mapping.
        """
        if len(records) != len(cluster_labels):
            raise ValueError(
                "records and cluster_labels must have the same length."
            )

        cluster_scores: dict[str, list[float]] = {}

        for record, cluster_label in zip(records, cluster_labels):
            cluster_id = self._normalize_cluster_id(cluster_label)

            if cluster_id is None:
                continue

            score = self._to_float(record.get(self.score_column))

            if score is None:
                continue

            cluster_scores.setdefault(cluster_id, []).append(score)

        if not cluster_scores:
            return {}

        averages = {
            cluster_id: sum(scores) / len(scores)
            for cluster_id, scores in cluster_scores.items()
        }

        ranked_clusters = sorted(
            averages,
            key=lambda cluster_id: (
                averages[cluster_id],
                cluster_id,
            ),
        )

        if len(ranked_clusters) == 1:
            return {
                ranked_clusters[0]: "transitional",
            }

        mapping: dict[str, str] = {}

        for index, cluster_id in enumerate(ranked_clusters):
            if index == 0:
                mapping[cluster_id] = "risk_off"
            elif index == len(ranked_clusters) - 1:
                mapping[cluster_id] = "risk_on"
            else:
                mapping[cluster_id] = "transitional"

        return mapping

    def map_labels(
        self,
        cluster_labels: list[str | int],
        mapping: dict[str, str],
    ) -> list[str]:
        """
        Apply a cluster-to-regime mapping to raw cluster labels.
        """
        regimes: list[str] = []

        for cluster_label in cluster_labels:
            cluster_id = self._normalize_cluster_id(cluster_label)

            if cluster_id is None:
                regimes.append("unknown")
                continue

            regimes.append(
                mapping.get(cluster_id, "unknown")
            )

        return regimes

    @staticmethod
    def _normalize_cluster_id(
        value: str | int,
    ) -> str | None:
        if isinstance(value, bool) or value is None:
            return None

        if isinstance(value, int):
            return str(value)

        if isinstance(value, str):
            normalized = value.strip()

            if normalized.isdigit():
                return str(int(normalized))

        return None

    @staticmethod
    def _to_float(value: Any) -> float | None:
        if value is None or isinstance(value, bool):
            return None

        try:
            number = float(value)
        except (TypeError, ValueError):
            return None

        if number != number:  # NaN
            return None

        if number in (float("inf"), float("-inf")):
            return None

        return number