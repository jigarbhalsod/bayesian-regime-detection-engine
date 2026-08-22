import pytest

from src.regime import (
    ClusteringRegimeDetector,
    RegimeConfig,
    RegimeDetectorFactory,
    RuleBasedRegimeDetector,
    StatisticalRegimeDetector,
)


def test_factory_lists_registered_detectors():
    available = RegimeDetectorFactory.available()

    assert "rule_based" in available
    assert "statistical" in available
    assert "clustering" in available


def test_factory_creates_rule_based_detector():
    detector = RegimeDetectorFactory.create("rule_based")

    assert isinstance(detector, RuleBasedRegimeDetector)


def test_factory_creates_statistical_detector():
    detector = RegimeDetectorFactory.create("statistical")

    assert isinstance(detector, StatisticalRegimeDetector)


def test_factory_creates_clustering_detector():
    config = RegimeConfig(
        feature_columns=("return_1d", "volatility"),
    )

    detector = RegimeDetectorFactory.create(
        "clustering",
        config=config,
    )

    assert isinstance(detector, ClusteringRegimeDetector)
    assert detector.config == config


def test_factory_passes_config_to_detector():
    config = RegimeConfig(
        n_regimes=4,
        feature_columns=("return_1d", "volatility"),
    )

    detector = RegimeDetectorFactory.create(
        "clustering",
        config=config,
    )

    assert detector.config == config


def test_factory_is_case_insensitive():
    detector = RegimeDetectorFactory.create("RULE_BASED")

    assert isinstance(detector, RuleBasedRegimeDetector)


def test_factory_normalizes_whitespace():
    detector = RegimeDetectorFactory.create("  statistical  ")

    assert isinstance(detector, StatisticalRegimeDetector)


def test_factory_rejects_unknown_detector():
    with pytest.raises(ValueError, match="Unknown regime detector"):
        RegimeDetectorFactory.create("hmm")


def test_factory_rejects_non_string_name():
    with pytest.raises(ValueError, match="Detector name must be a string"):
        RegimeDetectorFactory.create(123)


def test_factory_registers_custom_detector():
    class CustomDetector:
        def __init__(self, config=None):
            self.config = config

    RegimeDetectorFactory.register("custom_test", CustomDetector)

    detector = RegimeDetectorFactory.create("custom_test")

    assert isinstance(detector, CustomDetector)


def test_factory_rejects_duplicate_registration():
    class DuplicateDetector:
        pass

    with pytest.raises(ValueError, match="already registered"):
        RegimeDetectorFactory.register(
            "rule_based",
            DuplicateDetector,
        )


def test_factory_rejects_invalid_registration_name():
    class CustomDetector:
        pass

    with pytest.raises(ValueError, match="Detector name must be a string"):
        RegimeDetectorFactory.register(
            123,
            CustomDetector,
        )


def test_factory_rejects_non_class_registration():
    with pytest.raises(TypeError, match="Detector must be a class"):
        RegimeDetectorFactory.register(
            "invalid_detector",
            "not_a_class",
        )