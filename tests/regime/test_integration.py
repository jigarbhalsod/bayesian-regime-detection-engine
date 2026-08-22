from src.regime import (
    RegimeConfig,
    RegimeDetectorFactory,
    RegimeLabelMapper,
    RegimeValidator,
)


def test_rule_based_end_to_end():
    records = [
        {"return_1d": 0.03, "volatility": 0.10},
        {"return_1d": -0.02, "volatility": 0.35},
        {"return_1d": 0.001, "volatility": 0.20},
    ]

    detector = RegimeDetectorFactory.create("rule_based")
    result = detector.detect(records)

    validator = RegimeValidator()

    assert len(result.regime_labels) == len(records)
    assert validator.validate(result) is True


def test_statistical_end_to_end():
    records = [
        {"return_1d": -0.03, "volatility": 0.40},
        {"return_1d": -0.01, "volatility": 0.30},
        {"return_1d": 0.01, "volatility": 0.20},
        {"return_1d": 0.03, "volatility": 0.10},
    ]

    detector = RegimeDetectorFactory.create("statistical")
    result = detector.detect(records)

    validator = RegimeValidator()

    assert len(result.regime_labels) == len(records)
    assert validator.validate(result) is True


def test_clustering_end_to_end_with_label_mapping():
    records = [
        {"return_1d": -0.05},
        {"return_1d": -0.04},
        {"return_1d": -0.03},
        {"return_1d": 0.00},
        {"return_1d": 0.01},
        {"return_1d": 0.02},
        {"return_1d": 0.03},
        {"return_1d": 0.04},
        {"return_1d": 0.05},
    ]

    config = RegimeConfig(
        n_regimes=3,
        feature_columns=("return_1d",),
        min_samples=3,
        random_state=42,
    )

    detector = RegimeDetectorFactory.create(
        "clustering",
        config=config,
    )

    cluster_result = detector.detect(records)

    mapper = RegimeLabelMapper(score_column="return_1d")

    mapping = mapper.build_mapping(
        cluster_result.regime_labels,
        records,
    )

    final_labels = mapper.map_labels(
        cluster_result.regime_labels,
        mapping,
    )

    cluster_result.regime_labels = final_labels

    validator = RegimeValidator()

    assert len(cluster_result.regime_labels) == len(records)
    assert validator.validate(cluster_result) is True


def test_factory_detector_validator_with_invalid_records():
    records = [
        {"return_1d": 0.03, "volatility": 0.10},
        {"return_1d": "invalid", "volatility": 0.20},
        {"return_1d": None, "volatility": None},
    ]

    detector = RegimeDetectorFactory.create("rule_based")
    result = detector.detect(records)

    validator = RegimeValidator()

    assert len(result.regime_labels) == len(records)
    assert validator.validate(result) is True


def test_custom_config_end_to_end():
    config = RegimeConfig(
        feature_columns=("return_1d", "volatility"),
        min_samples=2,
        random_state=42,
    )

    records = [
        {"return_1d": 0.04, "volatility": 0.08},
        {"return_1d": -0.04, "volatility": 0.40},
    ]

    detector = RegimeDetectorFactory.create(
        "rule_based",
        config=config,
    )

    result = detector.detect(records)

    assert detector.config == config
    assert len(result.regime_labels) == len(records)

    validator = RegimeValidator()

    assert validator.validate(result) is True


def test_all_registered_detectors_are_available():
    available = RegimeDetectorFactory.available()

    assert set(
        ("rule_based", "statistical", "clustering")
    ).issubset(set(available))


def test_unknown_factory_detector_rejected():
    try:
        RegimeDetectorFactory.create("unknown_detector")
        assert False, "Expected ValueError"
    except ValueError:
        assert True