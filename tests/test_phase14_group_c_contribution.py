import math

import pytest

from src.explainability.config import ExplainabilityConfig
from src.explainability.regime_contribution import (
    RegimeContributionAnalyzer,
)
from src.explainability.regime_models import RegimeContribution


class TestRegimeContributionAnalyzer:
    def test_default_config_is_created(self):
        analyzer = RegimeContributionAnalyzer()

        assert isinstance(analyzer.config, ExplainabilityConfig)

    def test_custom_config_is_used(self):
        config = ExplainabilityConfig(
            attribution_threshold=0.2,
            top_k=2,
        )

        analyzer = RegimeContributionAnalyzer(config)

        assert analyzer.config is config

    @pytest.mark.parametrize(
        "config",
        [
            "invalid",
            123,
            True,
            {},
        ],
    )
    def test_invalid_config_rejected(self, config):
        with pytest.raises(TypeError):
            RegimeContributionAnalyzer(config)

    def test_returns_ranked_contributions(self):
        analyzer = RegimeContributionAnalyzer(
            ExplainabilityConfig(normalize=False)
        )

        results = analyzer.analyze(
            {
                "volatility": -0.9,
                "momentum": 0.6,
                "volume": 0.3,
            }
        )

        assert all(
            isinstance(item, RegimeContribution)
            for item in results
        )

        assert [item.feature_name for item in results] == [
            "volatility",
            "momentum",
            "volume",
        ]

        assert [item.contribution for item in results] == [
            -0.9,
            0.6,
            0.3,
        ]

        assert [item.rank for item in results] == [1, 2, 3]

    def test_sorts_by_absolute_contribution_strength(self):
        analyzer = RegimeContributionAnalyzer(
            ExplainabilityConfig(normalize=False)
        )

        results = analyzer.analyze(
            {
                "feature_a": 0.4,
                "feature_b": -0.9,
                "feature_c": 0.7,
            }
        )

        assert [item.feature_name for item in results] == [
            "feature_b",
            "feature_c",
            "feature_a",
        ]

    def test_equal_strength_preserves_input_order(self):
        analyzer = RegimeContributionAnalyzer(
            ExplainabilityConfig(normalize=False)
        )

        results = analyzer.analyze(
            {
                "feature_a": 0.5,
                "feature_b": -0.5,
                "feature_c": 0.3,
            }
        )

        assert [item.feature_name for item in results] == [
            "feature_a",
            "feature_b",
            "feature_c",
        ]

    def test_applies_attribution_threshold(self):
        analyzer = RegimeContributionAnalyzer(
            ExplainabilityConfig(
                attribution_threshold=0.5,
                normalize=False,
            )
        )

        results = analyzer.analyze(
            {
                "feature_a": 0.8,
                "feature_b": -0.5,
                "feature_c": 0.49,
            }
        )

        assert [item.feature_name for item in results] == [
            "feature_a",
            "feature_b",
        ]

    def test_applies_top_k(self):
        analyzer = RegimeContributionAnalyzer(
            ExplainabilityConfig(
                top_k=2,
                normalize=False,
            )
        )

        results = analyzer.analyze(
            {
                "feature_a": 0.9,
                "feature_b": 0.8,
                "feature_c": 0.7,
            }
        )

        assert len(results) == 2
        assert [item.feature_name for item in results] == [
            "feature_a",
            "feature_b",
        ]

    def test_empty_dictionary_returns_empty_list(self):
        analyzer = RegimeContributionAnalyzer()

        assert analyzer.analyze({}) == []

    def test_none_contributions_rejected(self):
        analyzer = RegimeContributionAnalyzer()

        with pytest.raises(ValueError):
            analyzer.analyze(None)

    @pytest.mark.parametrize(
        "contributions",
        [
            [],
            "invalid",
            123,
            True,
        ],
    )
    def test_non_dictionary_contributions_rejected(
        self,
        contributions,
    ):
        analyzer = RegimeContributionAnalyzer()

        with pytest.raises(TypeError):
            analyzer.analyze(contributions)

    @pytest.mark.parametrize(
        "feature_name",
        [
            "",
            "   ",
        ],
    )
    def test_empty_feature_name_rejected(self, feature_name):
        analyzer = RegimeContributionAnalyzer()

        with pytest.raises(ValueError):
            analyzer.analyze(
                {
                    feature_name: 0.5,
                }
            )

    @pytest.mark.parametrize(
        "feature_name",
        [
            123,
            True,
            None,
        ],
    )
    def test_invalid_feature_name_type_rejected(
        self,
        feature_name,
    ):
        analyzer = RegimeContributionAnalyzer()

        with pytest.raises(TypeError):
            analyzer.analyze(
                {
                    feature_name: 0.5,
                }
            )

    @pytest.mark.parametrize(
        "contribution",
        [
            math.inf,
            -math.inf,
            math.nan,
        ],
    )
    def test_non_finite_contribution_rejected(
        self,
        contribution,
    ):
        analyzer = RegimeContributionAnalyzer()

        with pytest.raises(ValueError):
            analyzer.analyze(
                {
                    "volatility": contribution,
                }
            )

    @pytest.mark.parametrize(
        "contribution",
        [
            "invalid",
            True,
            None,
        ],
    )
    def test_invalid_contribution_type_rejected(
        self,
        contribution,
    ):
        analyzer = RegimeContributionAnalyzer()

        with pytest.raises(TypeError):
            analyzer.analyze(
                {
                    "volatility": contribution,
                }
            )

    def test_feature_name_is_normalized(self):
        analyzer = RegimeContributionAnalyzer(
            ExplainabilityConfig(normalize=False)
        )

        results = analyzer.analyze(
            {
                "  volatility  ": 0.8,
            }
        )

        assert results[0].feature_name == "volatility"

    def test_negative_contributions_preserve_sign(self):
        analyzer = RegimeContributionAnalyzer(
            ExplainabilityConfig(normalize=False)
        )

        results = analyzer.analyze(
            {
                "feature_a": -0.8,
                "feature_b": 0.6,
            }
        )

        assert results[0].contribution == -0.8
        assert results[1].contribution == 0.6

    def test_ranks_are_reassigned_after_filtering(self):
        analyzer = RegimeContributionAnalyzer(
            ExplainabilityConfig(
                attribution_threshold=0.5,
                normalize=False,
            )
        )

        results = analyzer.analyze(
            {
                "feature_a": 0.9,
                "feature_b": 0.1,
                "feature_c": 0.8,
            }
        )

        assert [item.rank for item in results] == [1, 2]
        assert [item.feature_name for item in results] == [
            "feature_a",
            "feature_c",
        ]