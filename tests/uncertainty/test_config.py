import pytest

from src.uncertainty.config import UncertaintyConfig


def test_default_configuration():
    config = UncertaintyConfig()

    assert config.low_uncertainty_threshold == 0.15
    assert config.high_uncertainty_threshold == 0.40
    assert config.abstain_on_high_uncertainty is True


@pytest.mark.parametrize(
    "low_threshold, high_threshold",
    [
        (-0.1, 0.4),
        (0.2, 1.1),
    ],
)
def test_invalid_threshold_range_raises_error(
    low_threshold,
    high_threshold,
):
    with pytest.raises(ValueError):
        UncertaintyConfig(
            low_uncertainty_threshold=low_threshold,
            high_uncertainty_threshold=high_threshold,
        )


def test_low_threshold_cannot_exceed_high_threshold():
    with pytest.raises(ValueError):
        UncertaintyConfig(
            low_uncertainty_threshold=0.6,
            high_uncertainty_threshold=0.4,
        )


def test_custom_configuration():
    config = UncertaintyConfig(
        low_uncertainty_threshold=0.2,
        high_uncertainty_threshold=0.7,
        abstain_on_high_uncertainty=False,
    )

    assert config.low_uncertainty_threshold == 0.2
    assert config.high_uncertainty_threshold == 0.7
    assert config.abstain_on_high_uncertainty is False