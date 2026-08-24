import math

import pytest
from pydantic import ValidationError

from src.explainability.regime_models import (
    RegimeContribution,
    RegimeExplanation,
)


class TestRegimeContribution:
    def test_creates_valid_contribution(self):
        result = RegimeContribution(
            feature_name="volatility",
            contribution=-0.75,
            rank=1,
        )

        assert result.feature_name == "volatility"
        assert result.contribution == -0.75
        assert result.rank == 1

    def test_feature_name_is_normalized(self):
        result = RegimeContribution(
            feature_name="  momentum  ",
            contribution=0.5,
        )

        assert result.feature_name == "momentum"

    def test_rank_is_optional(self):
        result = RegimeContribution(
            feature_name="volume",
            contribution=0.4,
        )

        assert result.rank is None

    @pytest.mark.parametrize(
        "feature_name",
        [
            "",
            "   ",
        ],
    )
    def test_empty_feature_name_rejected(self, feature_name):
        with pytest.raises(ValidationError):
            RegimeContribution(
                feature_name=feature_name,
                contribution=0.5,
            )

    @pytest.mark.parametrize(
        "contribution",
        [
            math.inf,
            -math.inf,
            math.nan,
        ],
    )
    def test_non_finite_contribution_rejected(self, contribution):
        with pytest.raises(ValidationError):
            RegimeContribution(
                feature_name="volatility",
                contribution=contribution,
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
        with pytest.raises(ValidationError):
            RegimeContribution(
                feature_name="volatility",
                contribution=contribution,
            )

    @pytest.mark.parametrize(
        "rank",
        [
            0,
            -1,
        ],
    )
    def test_invalid_rank_value_rejected(self, rank):
        with pytest.raises(ValidationError):
            RegimeContribution(
                feature_name="volatility",
                contribution=0.5,
                rank=rank,
            )

    @pytest.mark.parametrize(
        "rank",
        [
            1.5,
            True,
            "1",
        ],
    )
    def test_invalid_rank_type_rejected(self, rank):
        with pytest.raises(ValidationError):
            RegimeContribution(
                feature_name="volatility",
                contribution=0.5,
                rank=rank,
            )


class TestRegimeExplanation:
    def test_creates_complete_explanation(self):
        contribution = RegimeContribution(
            feature_name="volatility",
            contribution=0.8,
            rank=1,
        )

        result = RegimeExplanation(
            regime="RISK_OFF",
            probability=0.87,
            contributions=[contribution],
            key_drivers=["volatility"],
            summary="High volatility is driving a risk-off regime.",
            metadata={"model": "bayesian_regime_engine"},
        )

        assert result.regime == "RISK_OFF"
        assert result.probability == 0.87
        assert len(result.contributions) == 1
        assert result.key_drivers == ["volatility"]
        assert result.summary == (
            "High volatility is driving a risk-off regime."
        )
        assert result.metadata == {
            "model": "bayesian_regime_engine"
        }

    def test_all_fields_are_optional_or_defaulted(self):
        result = RegimeExplanation()

        assert result.regime is None
        assert result.probability is None
        assert result.contributions == []
        assert result.key_drivers == []
        assert result.summary is None
        assert result.metadata == {}

    def test_regime_is_normalized(self):
        result = RegimeExplanation(regime="  RISK_ON  ")

        assert result.regime == "RISK_ON"

    @pytest.mark.parametrize(
        "regime",
        [
            "",
            "   ",
        ],
    )
    def test_empty_regime_rejected(self, regime):
        with pytest.raises(ValidationError):
            RegimeExplanation(regime=regime)

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
        with pytest.raises(ValidationError):
            RegimeExplanation(probability=probability)

    @pytest.mark.parametrize(
        "probability",
        [
            "invalid",
            True,
        ],
    )
    def test_invalid_probability_type_rejected(
        self,
        probability,
    ):
        with pytest.raises(ValidationError):
            RegimeExplanation(probability=probability)

    def test_valid_probability_boundaries_accepted(self):
        assert RegimeExplanation(probability=0.0).probability == 0.0
        assert RegimeExplanation(probability=1.0).probability == 1.0

    def test_key_drivers_are_normalized(self):
        result = RegimeExplanation(
            key_drivers=[
                "  volatility  ",
                "momentum",
            ]
        )

        assert result.key_drivers == [
            "volatility",
            "momentum",
        ]

    def test_empty_key_driver_rejected(self):
        with pytest.raises(ValidationError):
            RegimeExplanation(
                key_drivers=["volatility", "   "]
            )

    @pytest.mark.parametrize(
        "drivers",
        [
            "volatility",
            {"volatility"},
            ("volatility",),
        ],
    )
    def test_non_list_key_drivers_rejected(self, drivers):
        with pytest.raises(ValidationError):
            RegimeExplanation(key_drivers=drivers)

    @pytest.mark.parametrize(
        "drivers",
        [
            [123],
            [True],
            [None],
        ],
    )
    def test_invalid_key_driver_item_rejected(self, drivers):
        with pytest.raises(ValidationError):
            RegimeExplanation(key_drivers=drivers)

    def test_summary_is_normalized(self):
        result = RegimeExplanation(
            summary="  Volatility increased sharply.  "
        )

        assert result.summary == (
            "Volatility increased sharply."
        )

    @pytest.mark.parametrize(
        "summary",
        [
            "",
            "   ",
        ],
    )
    def test_empty_summary_rejected(self, summary):
        with pytest.raises(ValidationError):
            RegimeExplanation(summary=summary)

    @pytest.mark.parametrize(
        "summary",
        [
            123,
            True,
            [],
        ],
    )
    def test_invalid_summary_type_rejected(self, summary):
        with pytest.raises(ValidationError):
            RegimeExplanation(summary=summary)

    @pytest.mark.parametrize(
        "metadata",
        [
            [],
            "invalid",
            123,
        ],
    )
    def test_invalid_metadata_rejected(self, metadata):
        with pytest.raises(ValidationError):
            RegimeExplanation(metadata=metadata)