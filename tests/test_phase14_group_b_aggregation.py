import pytest

from src.explainability.aggregation import ExplanationAggregator
from src.explainability.models import (
    ExplanationResult,
    FeatureAttribution,
    FeatureImportance,
)


class TestExplanationAggregator:
    def test_aggregates_single_result(self):
        aggregator = ExplanationAggregator()

        result = ExplanationResult(
            attributions=[
                FeatureAttribution(
                    feature_name="volatility",
                    attribution=0.8,
                    rank=1,
                )
            ],
            importances=[
                FeatureImportance(
                    feature_name="volatility",
                    importance=0.9,
                    rank=1,
                )
            ],
            prediction="RISK_OFF",
            confidence=0.85,
            warnings=["High volatility"],
            metadata={"source": "model_a"},
        )

        aggregated = aggregator.aggregate([result])

        assert aggregated == result

    def test_combines_attributions_from_multiple_results(self):
        aggregator = ExplanationAggregator()

        results = [
            ExplanationResult(
                attributions=[
                    FeatureAttribution(
                        feature_name="feature_a",
                        attribution=0.5,
                    )
                ]
            ),
            ExplanationResult(
                attributions=[
                    FeatureAttribution(
                        feature_name="feature_b",
                        attribution=-0.3,
                    )
                ]
            ),
        ]

        aggregated = aggregator.aggregate(results)

        assert [
            item.feature_name
            for item in aggregated.attributions
        ] == ["feature_a", "feature_b"]

    def test_combines_importances_from_multiple_results(self):
        aggregator = ExplanationAggregator()

        results = [
            ExplanationResult(
                importances=[
                    FeatureImportance(
                        feature_name="feature_a",
                        importance=0.8,
                    )
                ]
            ),
            ExplanationResult(
                importances=[
                    FeatureImportance(
                        feature_name="feature_b",
                        importance=0.6,
                    )
                ]
            ),
        ]

        aggregated = aggregator.aggregate(results)

        assert [
            item.feature_name
            for item in aggregated.importances
        ] == ["feature_a", "feature_b"]

    def test_uses_last_available_prediction(self):
        aggregator = ExplanationAggregator()

        results = [
            ExplanationResult(prediction="RISK_ON"),
            ExplanationResult(),
            ExplanationResult(prediction="RISK_OFF"),
        ]

        aggregated = aggregator.aggregate(results)

        assert aggregated.prediction == "RISK_OFF"

    def test_uses_last_available_confidence(self):
        aggregator = ExplanationAggregator()

        results = [
            ExplanationResult(confidence=0.60),
            ExplanationResult(),
            ExplanationResult(confidence=0.85),
        ]

        aggregated = aggregator.aggregate(results)

        assert aggregated.confidence == 0.85

    def test_combines_warnings(self):
        aggregator = ExplanationAggregator()

        results = [
            ExplanationResult(
                warnings=["Warning A"]
            ),
            ExplanationResult(
                warnings=["Warning B"]
            ),
        ]

        aggregated = aggregator.aggregate(results)

        assert aggregated.warnings == [
            "Warning A",
            "Warning B",
        ]

    def test_merges_metadata_with_later_values_overriding(self):
        aggregator = ExplanationAggregator()

        results = [
            ExplanationResult(
                metadata={
                    "model": "model_a",
                    "version": "1",
                }
            ),
            ExplanationResult(
                metadata={
                    "model": "model_b",
                    "environment": "test",
                }
            ),
        ]

        aggregated = aggregator.aggregate(results)

        assert aggregated.metadata == {
            "model": "model_b",
            "version": "1",
            "environment": "test",
        }

    def test_preserves_duplicate_attributions(self):
        aggregator = ExplanationAggregator()

        results = [
            ExplanationResult(
                attributions=[
                    FeatureAttribution(
                        feature_name="feature_a",
                        attribution=0.5,
                    )
                ]
            ),
            ExplanationResult(
                attributions=[
                    FeatureAttribution(
                        feature_name="feature_a",
                        attribution=0.2,
                    )
                ]
            ),
        ]

        aggregated = aggregator.aggregate(results)

        assert len(aggregated.attributions) == 2

    def test_preserves_duplicate_importances(self):
        aggregator = ExplanationAggregator()

        results = [
            ExplanationResult(
                importances=[
                    FeatureImportance(
                        feature_name="feature_a",
                        importance=0.8,
                    )
                ]
            ),
            ExplanationResult(
                importances=[
                    FeatureImportance(
                        feature_name="feature_a",
                        importance=0.4,
                    )
                ]
            ),
        ]

        aggregated = aggregator.aggregate(results)

        assert len(aggregated.importances) == 2

    def test_preserves_input_order(self):
        aggregator = ExplanationAggregator()

        results = [
            ExplanationResult(
                attributions=[
                    FeatureAttribution(
                        feature_name="feature_a",
                        attribution=0.5,
                    )
                ],
                warnings=["Warning A"],
            ),
            ExplanationResult(
                attributions=[
                    FeatureAttribution(
                        feature_name="feature_b",
                        attribution=-0.7,
                    )
                ],
                warnings=["Warning B"],
            ),
        ]

        aggregated = aggregator.aggregate(results)

        assert [
            item.feature_name
            for item in aggregated.attributions
        ] == ["feature_a", "feature_b"]

        assert aggregated.warnings == [
            "Warning A",
            "Warning B",
        ]

    def test_empty_list_returns_empty_result(self):
        aggregator = ExplanationAggregator()

        result = aggregator.aggregate([])

        assert isinstance(result, ExplanationResult)
        assert result.attributions == []
        assert result.importances == []
        assert result.prediction is None
        assert result.confidence is None
        assert result.warnings == []
        assert result.metadata == {}

    def test_none_results_rejected(self):
        aggregator = ExplanationAggregator()

        with pytest.raises(
            ValueError,
            match="results must not be None",
        ):
            aggregator.aggregate(None)

    @pytest.mark.parametrize(
        "results",
        [
            (),
            {},
            "invalid",
            123,
        ],
    )
    def test_non_list_results_rejected(self, results):
        aggregator = ExplanationAggregator()

        with pytest.raises(
            TypeError,
            match="results must be a list",
        ):
            aggregator.aggregate(results)

    @pytest.mark.parametrize(
        "results",
        [
            ["invalid"],
            [123],
            [{}],
        ],
    )
    def test_invalid_result_instances_rejected(self, results):
        aggregator = ExplanationAggregator()

        with pytest.raises(
            TypeError,
            match="each result must be an ExplanationResult instance",
        ):
            aggregator.aggregate(results)