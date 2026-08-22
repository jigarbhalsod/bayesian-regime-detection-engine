from src.regime import RegimeConfig, RegimeResult
from src.regime.statistical import StatisticalRegimeDetector


def test_statistical_detector_name():
    detector = StatisticalRegimeDetector()

    assert detector.name == "statistical"


def test_statistical_detector_handles_empty_records():
    detector = StatisticalRegimeDetector()

    result = detector.detect([])

    assert result.records == []
    assert result.regime_labels == []
    assert result.metadata["return_threshold"] is None
    assert result.metadata["volatility_threshold"] is None


def test_statistical_detector_calculates_median_thresholds():
    detector = StatisticalRegimeDetector()

    result = detector.detect(
        [
            {"return": -0.02, "volatility": 0.03},
            {"return": 0.00, "volatility": 0.02},
            {"return": 0.02, "volatility": 0.01},
        ]
    )

    assert result.metadata["return_threshold"] == 0.00
    assert result.metadata["volatility_threshold"] == 0.02


def test_statistical_detector_classifies_risk_on():
    detector = StatisticalRegimeDetector()

    result = detector.detect(
        [
            {"return": -0.01, "volatility": 0.03},
            {"return": 0.01, "volatility": 0.01},
            {"return": 0.02, "volatility": 0.02},
        ]
    )

    assert result.regime_labels[1] == "risk_on"


def test_statistical_detector_classifies_risk_off():
    detector = StatisticalRegimeDetector()

    result = detector.detect(
        [
            {"return": -0.02, "volatility": 0.03},
            {"return": 0.00, "volatility": 0.02},
            {"return": 0.02, "volatility": 0.01},
        ]
    )

    assert result.regime_labels[0] == "risk_off"


def test_statistical_detector_classifies_transitional():
    detector = StatisticalRegimeDetector()

    result = detector.detect(
        [
            {"return": -0.01, "volatility": 0.01},
            {"return": 0.00, "volatility": 0.02},
            {"return": 0.01, "volatility": 0.03},
        ]
    )

    assert "transitional" in result.regime_labels


def test_statistical_detector_handles_missing_values():
    detector = StatisticalRegimeDetector()

    result = detector.detect(
        [
            {"return": None, "volatility": 0.01},
            {"return": 0.01, "volatility": None},
        ]
    )

    assert result.regime_labels == [
        "unknown",
        "unknown",
    ]


def test_statistical_detector_handles_invalid_values():
    detector = StatisticalRegimeDetector()

    result = detector.detect(
        [
            {"return": "invalid", "volatility": 0.01},
            {"return": float("nan"), "volatility": 0.01},
            {"return": 0.01, "volatility": float("inf")},
        ]
    )

    assert result.regime_labels == [
        "unknown",
        "unknown",
        "unknown",
    ]


def test_statistical_detector_handles_numeric_strings():
    detector = StatisticalRegimeDetector()

    result = detector.detect(
        [
            {"return": "-0.01", "volatility": "0.03"},
            {"return": "0.01", "volatility": "0.01"},
        ]
    )

    assert result.metadata["valid_return_count"] == 2
    assert result.metadata["valid_volatility_count"] == 2


def test_statistical_detector_does_not_mutate_input():
    detector = StatisticalRegimeDetector()

    records = [
        {"return": -0.01, "volatility": 0.03},
        {"return": 0.01, "volatility": 0.01},
    ]

    original = [dict(record) for record in records]

    detector.detect(records)

    assert records == original


def test_statistical_detector_supports_custom_columns():
    detector = StatisticalRegimeDetector(
        return_column="daily_return",
        volatility_column="realized_volatility",
    )

    result = detector.detect(
        [
            {
                "daily_return": -0.01,
                "realized_volatility": 0.03,
            },
            {
                "daily_return": 0.01,
                "realized_volatility": 0.01,
            },
        ]
    )

    assert len(result.regime_labels) == 2


def test_statistical_detector_returns_regime_result():
    detector = StatisticalRegimeDetector()

    result = detector.detect([])

    assert isinstance(result, RegimeResult)


def test_statistical_detector_accepts_config():
    config = RegimeConfig(n_regimes=4)

    detector = StatisticalRegimeDetector(config=config)

    assert detector.config is config