from src.regime import RegimeConfig, RegimeResult
from src.regime.rule_based import RuleBasedRegimeDetector

def test_rule_based_detector_name():
    detector = RuleBasedRegimeDetector()

    assert detector.name == "rule_based"


def test_rule_based_detector_classifies_risk_on():
    detector = RuleBasedRegimeDetector()

    result = detector.detect(
        [
            {
                "return": 0.01,
                "volatility": 0.01,
            }
        ]
    )

    assert result.regime_labels == ["risk_on"]
    assert result.records[0]["regime"] == "risk_on"


def test_rule_based_detector_classifies_risk_off():
    detector = RuleBasedRegimeDetector()

    result = detector.detect(
        [
            {
                "return": -0.01,
                "volatility": 0.03,
            }
        ]
    )

    assert result.regime_labels == ["risk_off"]
    assert result.records[0]["regime"] == "risk_off"


def test_rule_based_detector_classifies_transitional():
    detector = RuleBasedRegimeDetector()

    result = detector.detect(
        [
            {
                "return": 0.01,
                "volatility": 0.03,
            }
        ]
    )

    assert result.regime_labels == ["transitional"]


def test_rule_based_detector_handles_missing_values():
    detector = RuleBasedRegimeDetector()

    result = detector.detect(
        [
            {
                "return": None,
                "volatility": 0.01,
            },
            {
                "return": 0.01,
                "volatility": None,
            },
        ]
    )

    assert result.regime_labels == [
        "unknown",
        "unknown",
    ]


def test_rule_based_detector_handles_invalid_values():
    detector = RuleBasedRegimeDetector()

    result = detector.detect(
        [
            {
                "return": "invalid",
                "volatility": 0.01,
            },
            {
                "return": float("nan"),
                "volatility": 0.01,
            },
            {
                "return": 0.01,
                "volatility": float("inf"),
            },
        ]
    )

    assert result.regime_labels == [
        "unknown",
        "unknown",
        "unknown",
    ]


def test_rule_based_detector_handles_numeric_strings():
    detector = RuleBasedRegimeDetector()

    result = detector.detect(
        [
            {
                "return": "0.01",
                "volatility": "0.01",
            }
        ]
    )

    assert result.regime_labels == ["risk_on"]


def test_rule_based_detector_does_not_mutate_input():
    detector = RuleBasedRegimeDetector()

    records = [
        {
            "return": 0.01,
            "volatility": 0.01,
        }
    ]

    original = [dict(record) for record in records]

    detector.detect(records)

    assert records == original


def test_rule_based_detector_supports_custom_columns():
    detector = RuleBasedRegimeDetector(
        return_column="daily_return",
        volatility_column="realized_volatility",
    )

    result = detector.detect(
        [
            {
                "daily_return": 0.01,
                "realized_volatility": 0.01,
            }
        ]
    )

    assert result.regime_labels == ["risk_on"]


def test_rule_based_detector_uses_custom_thresholds():
    detector = RuleBasedRegimeDetector(
        positive_return_threshold=0.02,
        high_volatility_threshold=0.05,
    )

    result = detector.detect(
        [
            {
                "return": 0.03,
                "volatility": 0.04,
            }
        ]
    )

    assert result.regime_labels == ["risk_on"]


def test_rule_based_detector_returns_regime_result():
    detector = RuleBasedRegimeDetector()

    result = detector.detect([])

    assert isinstance(result, RegimeResult)
    assert result.records == []
    assert result.regime_labels == []


def test_rule_based_detector_accepts_config():
    config = RegimeConfig(n_regimes=4)

    detector = RuleBasedRegimeDetector(config=config)

    assert detector.config is config