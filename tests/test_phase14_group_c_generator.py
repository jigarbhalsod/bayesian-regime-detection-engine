import math

import pytest

from src.explainability.config import ExplainabilityConfig
from src.explainability.regime_generator import (
    RegimeExplanationGenerator,
)
from src.explainability.regime_models import (
    RegimeContribution,
    RegimeExplanation,
)


class TestRegimeExplanationGenerator:
    def test_default_config_is_created(self):
        generator = RegimeExplanationGenerator()

        assert isinstance(generator.config, ExplainabilityConfig)

    def test_custom_config_is_used(self):
        config = ExplainabilityConfig(top_k=2)

        generator = RegimeExplanationGenerator(config)

        assert generator.config is config

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
            RegimeExplanationGenerator(config)

    def test_generates_complete_explanation(self):
        generator = RegimeExplanationGenerator()

        contributions = [
            RegimeContribution(
                feature_name="volatility",
                contribution=-0.8,
                rank=1,
            ),
            RegimeContribution(
                feature_name="momentum",
                contribution=0.5,
                rank=2,
            ),
        ]

        result = generator.generate(
            regime="RISK_OFF",
            probability=0.87,
            contributions=contributions,
            metadata={"model": "bayesian"},
        )

        assert isinstance(result, RegimeExplanation)
        assert result.regime == "RISK_OFF"
        assert result.probability == 0.87
        assert result.contributions == contributions
        assert result.key_drivers == [
            "volatility",
            "momentum",
        ]
        assert result.summary == (
            "The market is classified as RISK_OFF "
            "with probability 0.87, driven by volatility "
            "and momentum."
        )
        assert result.metadata == {"model": "bayesian"}

    def test_generates_explanation_without_probability(self):
        generator = RegimeExplanationGenerator()

        result = generator.generate(
            regime="RISK_ON",
        )

        assert result.probability is None
        assert result.summary == (
            "The market is classified as RISK_ON "
            "with no probability available."
        )

    def test_generates_explanation_without_contributions(self):
        generator = RegimeExplanationGenerator()

        result = generator.generate(
            regime="TRANSITIONAL",
            probability=0.5,
        )

        assert result.contributions == []
        assert result.key_drivers == []
        assert result.summary == (
            "The market is classified as TRANSITIONAL "
            "with probability 0.50."
        )

    def test_preserves_contribution_order(self):
        generator = RegimeExplanationGenerator()

        contributions = [
            RegimeContribution(
                feature_name="feature_c",
                contribution=0.2,
            ),
            RegimeContribution(
                feature_name="feature_a",
                contribution=0.9,
            ),
        ]

        result = generator.generate(
            regime="RISK_ON",
            contributions=contributions,
        )

        assert result.key_drivers == [
            "feature_c",
            "feature_a",
        ]

    def test_regime_is_normalized(self):
        generator = RegimeExplanationGenerator()

        result = generator.generate(
            regime="  RISK_OFF  ",
        )

        assert result.regime == "RISK_OFF"

    @pytest.mark.parametrize(
        "regime",
        [
            "",
            "   ",
        ],
    )
    def test_empty_regime_rejected(self, regime):
        generator = RegimeExplanationGenerator()

        with pytest.raises(ValueError):
            generator.generate(regime=regime)

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
        generator = RegimeExplanationGenerator()

        with pytest.raises(TypeError):
            generator.generate(regime=regime)

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
        generator = RegimeExplanationGenerator()

        with pytest.raises(ValueError):
            generator.generate(
                regime="RISK_ON",
                probability=probability,
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
        generator = RegimeExplanationGenerator()

        with pytest.raises(TypeError):
            generator.generate(
                regime="RISK_ON",
                probability=probability,
            )

    @pytest.mark.parametrize(
        "contributions",
        [
            "invalid",
            {},
            123,
            True,
        ],
    )
    def test_non_list_contributions_rejected(
        self,
        contributions,
    ):
        generator = RegimeExplanationGenerator()

        with pytest.raises(TypeError):
            generator.generate(
                regime="RISK_ON",
                contributions=contributions,
            )

    @pytest.mark.parametrize(
        "contributions",
        [
            [None],
            ["invalid"],
            [123],
            [{}],
        ],
    )
    def test_invalid_contribution_instances_rejected(
        self,
        contributions,
    ):
        generator = RegimeExplanationGenerator()

        with pytest.raises(TypeError):
            generator.generate(
                regime="RISK_ON",
                contributions=contributions,
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
        generator = RegimeExplanationGenerator()

        with pytest.raises(TypeError):
            generator.generate(
                regime="RISK_ON",
                metadata=metadata,
            )

    def test_none_metadata_defaults_to_empty_dictionary(self):
        generator = RegimeExplanationGenerator()

        result = generator.generate(
            regime="RISK_ON",
            metadata=None,
        )

        assert result.metadata == {}