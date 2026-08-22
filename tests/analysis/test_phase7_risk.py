import numpy as np
import pytest

from src.analysis.config import AnalysisConfig
from src.analysis.risk import RiskAnalyzer

def test_risk_analyzer_adds_rolling_volatility() -> None:
    config = AnalysisConfig(
        volatility_windows=(3,),
    )

    analyzer = RiskAnalyzer(config=config)

    records = [
        {"feature__return_1d": 0.01},
        {"feature__return_1d": -0.02},
        {"feature__return_1d": 0.03},
        {"feature__return_1d": 0.00},
    ]

    result = analyzer.analyze(records)

    column = "analysis__rolling_volatility_3"

    assert result.records[0][column] is None
    assert result.records[1][column] is None

    assert result.records[2][column] == pytest.approx(
        np.std([0.01, -0.02, 0.03], ddof=1)
    )

    assert result.records[3][column] == pytest.approx(
        np.std([-0.02, 0.03, 0.00], ddof=1)
    )


def test_risk_analyzer_supports_multiple_volatility_windows() -> None:
    config = AnalysisConfig(
        volatility_windows=(2, 3),
    )

    analyzer = RiskAnalyzer(config=config)

    records = [
        {"feature__return_1d": 0.01},
        {"feature__return_1d": -0.02},
        {"feature__return_1d": 0.03},
    ]

    result = analyzer.analyze(records)

    volatility_2 = "analysis__rolling_volatility_2"
    volatility_3 = "analysis__rolling_volatility_3"

    assert result.records[0][volatility_2] is None
    assert result.records[0][volatility_3] is None

    assert result.records[1][volatility_2] == pytest.approx(
        np.std([0.01, -0.02], ddof=1)
    )
    assert result.records[1][volatility_3] is None

    assert result.records[2][volatility_2] == pytest.approx(
        np.std([-0.02, 0.03], ddof=1)
    )
    assert result.records[2][volatility_3] == pytest.approx(
        np.std([0.01, -0.02, 0.03], ddof=1)
    )


def test_risk_analyzer_rolling_volatility_requires_valid_window() -> None:
    config = AnalysisConfig(
        volatility_windows=(2,),
    )

    analyzer = RiskAnalyzer(config=config)

    records = [
        {"feature__return_1d": 0.01},
        {"feature__return_1d": None},
        {"feature__return_1d": 0.03},
        {"feature__return_1d": -0.02},
    ]

    result = analyzer.analyze(records)

    column = "analysis__rolling_volatility_2"

    assert result.records[0][column] is None
    assert result.records[1][column] is None
    assert result.records[2][column] is None

    assert result.records[3][column] == pytest.approx(
        np.std([0.03, -0.02], ddof=1)
    )


def test_risk_analyzer_reports_volatility_windows_metadata() -> None:
    config = AnalysisConfig(
        volatility_windows=(3, 10),
    )

    analyzer = RiskAnalyzer(config=config)

    result = analyzer.analyze(
        [{"feature__return_1d": 0.01}]
    )

    assert result.metadata["volatility_windows"] == [3, 10]