import pytest

from src.regime import BaseRegimeDetector, RegimeConfig, RegimeResult


class DummyRegimeDetector(BaseRegimeDetector):

    @property
    def name(self) -> str:
        return "dummy"

    def detect(self, records):
        return RegimeResult(
            records=records,
            regime_labels=["dummy"] * len(records),
        )


def test_detector_uses_default_config():
    detector = DummyRegimeDetector()

    assert isinstance(detector.config, RegimeConfig)
    assert detector.config == RegimeConfig()


def test_detector_accepts_custom_config():
    config = RegimeConfig(n_regimes=5)

    detector = DummyRegimeDetector(config=config)

    assert detector.config is config
    assert detector.config.n_regimes == 5


def test_detector_name():
    detector = DummyRegimeDetector()

    assert detector.name == "dummy"


def test_detector_returns_regime_result():
    detector = DummyRegimeDetector()

    records = [
        {"return": 0.01},
        {"return": -0.02},
    ]

    result = detector.detect(records)

    assert isinstance(result, RegimeResult)
    assert result.records == records
    assert result.regime_labels == ["dummy", "dummy"]


def test_base_detector_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        BaseRegimeDetector()