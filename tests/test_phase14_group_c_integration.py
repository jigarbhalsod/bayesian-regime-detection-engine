import math

import pytest

from src.explainability.config import ExplainabilityConfig
from src.explainability.regime_integration import (
    RegimeExplanationPipeline,
)
from src.explainability.regime_models import (
    RegimeContribution,
    RegimeExplanation,
)


class TestRegimeExplanationPipeline:
    def test_default_config_is_created(self):
        pipeline = RegimeExplanationPipeline()

        assert isinstance(
            pipeline.config,
            ExplainabilityConfig,
        )

    def test_custom_config_is_used_across_components(self):
        config = ExplainabilityConfig(
            attribution_threshold=0.2,
            top_k=2,
        )

        pipeline = RegimeExplanationPipeline(config)

        assert pipeline.config is config
        assert pipeline.contribution_analyzer.config is config
        assert pipeline.generator.config is config

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
            RegimeExplanationPipeline(config)

    def test_complete_regime_explanation_flow(self):
        pipeline = RegimeExplanationPipeline(
            ExplainabilityConfig(normalize=False)
        )

        result = pipeline.explain(
            regime="RISK_OFF",
            probability=0.87,
            raw_contributions={
                "volatility": -0.9,
                "momentum": 0.6,
                "volume": 0.3,
            },
            metadata={"model": "bayesian"},
        )

        assert isinstance(result, RegimeExplanation)
        assert result.regime == "RISK_OFF"
        assert result.probability == 0.87
        assert result.metadata == {"model": "bayesian"}

        assert all(
            isinstance(item, RegimeContribution)
            for item in result.contributions
        )

        assert [
            item.feature_name
            for item in result.contributions
        ] == [
            "volatility",
            "momentum",
            "volume",
        ]

        assert [
            item.rank
            for item in result.contributions
        ] == [1, 2, 3]

        assert result.key_drivers == [
            "volatility",
            "momentum",
            "volume",
        ]

    def test_threshold_filtering_flows_through_pipeline(self):
        pipeline = RegimeExplanationPipeline(
            ExplainabilityConfig(
                attribution_threshold=0.5,
                normalize=False,
            )
        )

        result = pipeline.explain(
            regime="RISK_ON",
            raw_contributions={
                "feature_a": 0.8,
                "feature_b": 0.5,
                "feature_c": 0.49,
            },
        )

        assert result.key_drivers == [
            "feature_a",
            "feature_b",
        ]

    def test_top_k_flows_through_pipeline(self):
        pipeline = RegimeExplanationPipeline(
            ExplainabilityConfig(
                top_k=2,
                normalize=False,
            )
        )

        result = pipeline.explain(
            regime="RISK_ON",
            raw_contributions={
                "feature_a": 0.9,
                "feature_b": 0.8,
                "feature_c": 0.7,
            },
        )

        assert len(result.contributions) == 2
        assert result.key_drivers == [
            "feature_a",
            "feature_b",
        ]
        assert [item.rank for item in result.contributions] == [
            1,
            2,
        ]

    def test_negative_contributions_preserve_sign(self):
        pipeline = RegimeExplanationPipeline(
            ExplainabilityConfig(normalize=False)
        )

        result = pipeline.explain(
            regime="RISK_OFF",
            raw_contributions={
                "volatility": -0.9,
                "momentum": 0.6,
            },
        )

        assert result.contributions[0].contribution == -0.9
        assert result.contributions[1].contribution == 0.6

    def test_equal_strength_preserves_input_order(self):
        pipeline = RegimeExplanationPipeline(
            ExplainabilityConfig(normalize=False)
        )

        result = pipeline.explain(
            regime="TRANSITIONAL",
            raw_contributions={
                "feature_a": 0.5,
                "feature_b": -0.5,
                "feature_c": 0.3,
            },
        )

        assert result.key_drivers == [
            "feature_a",
            "feature_b",
            "feature_c",
        ]

    def test_empty_contributions_flow_through_pipeline(self):
        pipeline = RegimeExplanationPipeline()

        result = pipeline.explain(
            regime="POST_SHOCK",
            raw_contributions={},
        )

        assert result.contributions == []
        assert result.key_drivers == []
        assert result.summary == (
            "The market is classified as POST_SHOCK "
            "with no probability available."
        )

    def test_feature_names_are_normalized_through_pipeline(self):
        pipeline = RegimeExplanationPipeline(
            ExplainabilityConfig(normalize=False)
        )

        result = pipeline.explain(
            regime="  RISK_ON  ",
            raw_contributions={
                "  volatility  ": 0.8,
            },
        )

        assert result.regime == "RISK_ON"
        assert result.key_drivers == ["volatility"]

    @pytest.mark.parametrize(
        "regime",
        [
            "",
            "   ",
        ],
    )
    def test_empty_regime_rejected(self, regime):
        pipeline = RegimeExplanationPipeline()

        with pytest.raises(ValueError):
            pipeline.explain(
                regime=regime,
                raw_contributions={},
            )

    @pytest.mark.parametrize(
        "regime",
        [
            123,
            True,
            None,
            [],
        ],
    )
    def test_invalid_regime_type_rejected(self, regime):
        pipeline = RegimeExplanationPipeline()

        with pytest.raises(TypeError):
            pipeline.explain(
                regime=regime,
                raw_contributions={},
            )

    @pytest.mark.parametrize(
        "probability",
        [
            -0.1,
            1.1,
            math.inf,
            -math.inf,
            math.nan,
        ],
    )
    def test_invalid_probability_value_rejected(
        self,
        probability,
    ):
        pipeline = RegimeExplanationPipeline()

        with pytest.raises(ValueError):
            pipeline.explain(
                regime="RISK_ON",
                probability=probability,
                raw_contributions={},
            )

    @pytest.mark.parametrize(
        "probability",
        [
            "invalid",
            True,
            [],
        ],
    )
    def test_invalid_probability_type_rejected(
        self,
        probability,
    ):
        pipeline = RegimeExplanationPipeline()

        with pytest.raises(TypeError):
            pipeline.explain(
                regime="RISK_ON",
                probability=probability,
                raw_contributions={},
            )

    def test_none_contributions_rejected(self):
        pipeline = RegimeExplanationPipeline()

        with pytest.raises(ValueError):
            pipeline.explain(
                regime="RISK_ON",
                raw_contributions=None,
            )

    @pytest.mark.parametrize(
        "raw_contributions",
        [
            [],
            "invalid",
            123,
            True,
        ],
    )
    def test_non_dictionary_contributions_rejected(
        self,
        raw_contributions,
    ):
        pipeline = RegimeExplanationPipeline()

        with pytest.raises(TypeError):
            pipeline.explain(
                regime="RISK_ON",
                raw_contributions=raw_contributions,
            )

    @pytest.mark.parametrize(
        "metadata",
        [
            [],
            "invalid",
            123,
            True,
        ],
    )
    def test_invalid_metadata_rejected(self, metadata):
        pipeline = RegimeExplanationPipeline()

        with pytest.raises(TypeError):
            pipeline.explain(
                regime="RISK_ON",
                raw_contributions={},
                metadata=metadata,
            )

    def test_none_metadata_defaults_to_empty_dictionary(self):
        pipeline = RegimeExplanationPipeline()

        result = pipeline.explain(
            regime="RISK_ON",
            raw_contributions={},
            metadata=None,
        )

        assert result.metadata == {}