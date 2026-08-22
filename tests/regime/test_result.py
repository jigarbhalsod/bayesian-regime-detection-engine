from src.regime.result import RegimeResult


def test_result_defaults():
    result = RegimeResult()

    assert result.records == []
    assert result.regime_labels == []
    assert result.regime_probabilities == []
    assert result.confidence_scores == []
    assert result.metrics == {}
    assert result.metadata == {}


def test_result_accepts_custom_values():
    result = RegimeResult(
        records=[{"return": 0.01}],
        regime_labels=["risk_on"],
        regime_probabilities=[
            {
                "risk_on": 0.9,
                "risk_off": 0.1,
            }
        ],
        confidence_scores=[0.9],
        metrics={"log_likelihood": -12.5},
        metadata={"detector": "test"},
    )

    assert result.records == [{"return": 0.01}]
    assert result.regime_labels == ["risk_on"]
    assert result.regime_probabilities == [
        {
            "risk_on": 0.9,
            "risk_off": 0.1,
        }
    ]
    assert result.confidence_scores == [0.9]
    assert result.metrics == {"log_likelihood": -12.5}
    assert result.metadata == {"detector": "test"}


def test_result_mutable_defaults_are_independent():
    first = RegimeResult()
    second = RegimeResult()

    first.records.append({"value": 1})
    first.regime_labels.append("risk_on")
    first.regime_probabilities.append({"risk_on": 1.0})
    first.confidence_scores.append(1.0)
    first.metrics["score"] = 10
    first.metadata["source"] = "test"

    assert second.records == []
    assert second.regime_labels == []
    assert second.regime_probabilities == []
    assert second.confidence_scores == []
    assert second.metrics == {}
    assert second.metadata == {}