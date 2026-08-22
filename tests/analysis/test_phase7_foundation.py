import pytest

from src.analysis.base import BaseFinancialAnalyzer
from src.analysis.config import AnalysisConfig
from src.analysis.pipeline import FinancialAnalysisPipeline
from src.analysis.registry import AnalyzerRegistry
from src.analysis.result import AnalysisResult

def test_analysis_result_required_records() -> None:
    records = [
        {"date": "2026-01-01", "close": 100.0},
        {"date": "2026-01-02", "close": 101.0},
    ]

    result = AnalysisResult(records=records)

    assert result.records == records
    assert result.metrics == {}
    assert result.metric_names == []
    assert result.metadata == {}


def test_analysis_result_with_values() -> None:
    records = [
        {"date": "2026-01-01", "close": 100.0},
    ]

    metrics = {
        "analysis__mean_return": 0.01,
        "analysis__cumulative_return": 0.05,
    }

    metric_names = [
        "analysis__mean_return",
        "analysis__cumulative_return",
    ]

    metadata = {
        "analyzer": "returns",
    }

    result = AnalysisResult(
        records=records,
        metrics=metrics,
        metric_names=metric_names,
        metadata=metadata,
    )

    assert result.records == records
    assert result.metrics == metrics
    assert result.metric_names == metric_names
    assert result.metadata == metadata


def test_analysis_result_defaults_are_independent() -> None:
    first = AnalysisResult(records=[])
    second = AnalysisResult(records=[])

    first.metrics["test"] = 1
    first.metric_names.append("test")
    first.metadata["source"] = "first"

    assert second.metrics == {}
    assert second.metric_names == []
    assert second.metadata == {}

def test_analysis_config_defaults() -> None:
    config = AnalysisConfig()

    assert config.return_column == "feature__return_1d"
    assert config.risk_free_rate == 0.0
    assert config.annualization_factor == 252

    assert tuple(config.return_periods) == (1, 5, 20)
    assert tuple(config.rolling_windows) == (5, 20, 60)
    assert tuple(config.volatility_windows) == (5, 20, 60)
    assert tuple(config.correlation_windows) == (5, 20, 60)

    assert config.drawdown_threshold == -0.10
    assert config.stress_enabled is True


def test_analysis_config_custom_values() -> None:
    config = AnalysisConfig(
        return_column="custom_return",
        risk_free_rate=0.05,
        annualization_factor=365,
        return_periods=(1, 10),
        rolling_windows=(10, 30),
        volatility_windows=(10, 30),
        drawdown_threshold=-0.20,
        correlation_windows=(10, 30),
        stress_enabled=False,
    )

    assert config.return_column == "custom_return"
    assert config.risk_free_rate == 0.05
    assert config.annualization_factor == 365

    assert tuple(config.return_periods) == (1, 10)
    assert tuple(config.rolling_windows) == (10, 30)
    assert tuple(config.volatility_windows) == (10, 30)
    assert tuple(config.correlation_windows) == (10, 30)

    assert config.drawdown_threshold == -0.20
    assert config.stress_enabled is False


def test_analysis_config_is_immutable() -> None:
    config = AnalysisConfig()

    with pytest.raises(Exception):
        config.risk_free_rate = 0.05

def test_base_financial_analyzer_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        BaseFinancialAnalyzer()


def test_base_financial_analyzer_uses_default_config() -> None:
    class DummyAnalyzer(BaseFinancialAnalyzer):
        @property
        def name(self) -> str:
            return "dummy"

        def analyze(self, records):
            return AnalysisResult(records=list(records))

    analyzer = DummyAnalyzer()

    assert analyzer.name == "dummy"
    assert isinstance(analyzer.config, AnalysisConfig)
    assert analyzer.config == AnalysisConfig()


def test_base_financial_analyzer_accepts_custom_config() -> None:
    class DummyAnalyzer(BaseFinancialAnalyzer):
        @property
        def name(self) -> str:
            return "dummy"

        def analyze(self, records):
            return AnalysisResult(records=list(records))

    config = AnalysisConfig(
        risk_free_rate=0.05,
        annualization_factor=365,
    )

    analyzer = DummyAnalyzer(config=config)

    assert analyzer.config is config
    assert analyzer.config.risk_free_rate == 0.05
    assert analyzer.config.annualization_factor == 365

class DummyFinancialAnalyzer(BaseFinancialAnalyzer):
    """
    Simple analyzer used for registry tests.
    """

    def __init__(
        self,
        analyzer_name: str,
        config: AnalysisConfig | None = None,
    ) -> None:
        super().__init__(config=config)
        self._analyzer_name = analyzer_name

    @property
    def name(self) -> str:
        return self._analyzer_name

    def analyze(self, records):
        return AnalysisResult(records=list(records))


def test_analyzer_registry_starts_empty() -> None:
    registry = AnalyzerRegistry()

    assert len(registry) == 0
    assert registry.names() == []
    assert registry.analyzers() == []


def test_analyzer_registry_registers_and_resolves_analyzer() -> None:
    registry = AnalyzerRegistry()
    analyzer = DummyFinancialAnalyzer("returns")

    registry.register(analyzer)

    assert len(registry) == 1
    assert registry.get("returns") is analyzer
    assert registry.names() == ["returns"]


def test_analyzer_registry_preserves_registration_order() -> None:
    registry = AnalyzerRegistry()

    returns = DummyFinancialAnalyzer("returns")
    risk = DummyFinancialAnalyzer("risk")
    drawdown = DummyFinancialAnalyzer("drawdown")

    registry.register(returns)
    registry.register(risk)
    registry.register(drawdown)

    assert registry.names() == [
        "returns",
        "risk",
        "drawdown",
    ]

    assert registry.analyzers() == [
        returns,
        risk,
        drawdown,
    ]


def test_analyzer_registry_rejects_duplicate_names() -> None:
    registry = AnalyzerRegistry()

    registry.register(DummyFinancialAnalyzer("returns"))

    with pytest.raises(ValueError, match="already registered"):
        registry.register(DummyFinancialAnalyzer("returns"))


def test_analyzer_registry_rejects_unknown_name() -> None:
    registry = AnalyzerRegistry()

    with pytest.raises(KeyError, match="not registered"):
        registry.get("unknown")


def test_analyzer_registry_is_iterable() -> None:
    registry = AnalyzerRegistry()

    returns = DummyFinancialAnalyzer("returns")
    risk = DummyFinancialAnalyzer("risk")

    registry.register(returns)
    registry.register(risk)

    assert list(registry) == [returns, risk]

def test_financial_analysis_pipeline_runs_without_analyzers() -> None:
    pipeline = FinancialAnalysisPipeline()

    records = [
        {"date": "2026-01-01", "value": 100.0},
        {"date": "2026-01-02", "value": 101.0},
    ]

    result = pipeline.run(records)

    assert result.records == records
    assert result.records is not records
    assert result.metrics == {}
    assert result.metric_names == []
    assert result.metadata == {}


def test_financial_analysis_pipeline_registers_and_runs_analyzer() -> None:
    class TestAnalyzer(BaseFinancialAnalyzer):
        @property
        def name(self) -> str:
            return "test"

        def analyze(self, records):
            updated_records = [
                {**record, "analyzed": True}
                for record in records
            ]

            return AnalysisResult(
                records=updated_records,
                metrics={"analysis__test": 1},
                metric_names=["analysis__test"],
                metadata={"status": "complete"},
            )

    pipeline = FinancialAnalysisPipeline()
    pipeline.register(TestAnalyzer())

    records = [{"value": 100.0}]

    result = pipeline.run(records)

    assert result.records == [
        {"value": 100.0, "analyzed": True}
    ]

    assert result.metrics == {
        "analysis__test": 1
    }

    assert result.metric_names == [
        "analysis__test"
    ]

    assert result.metadata == {
        "test": {"status": "complete"}
    }


def test_financial_analysis_pipeline_runs_analyzers_in_order() -> None:
    execution_order: list[str] = []

    class FirstAnalyzer(BaseFinancialAnalyzer):
        @property
        def name(self) -> str:
            return "first"

        def analyze(self, records):
            execution_order.append("first")

            updated_records = [
                {**record, "first": True}
                for record in records
            ]

            return AnalysisResult(
                records=updated_records,
                metrics={"analysis__first": 1},
                metric_names=["analysis__first"],
            )

    class SecondAnalyzer(BaseFinancialAnalyzer):
        @property
        def name(self) -> str:
            return "second"

        def analyze(self, records):
            execution_order.append("second")

            assert all(
                record.get("first") is True
                for record in records
            )

            updated_records = [
                {**record, "second": True}
                for record in records
            ]

            return AnalysisResult(
                records=updated_records,
                metrics={"analysis__second": 2},
                metric_names=["analysis__second"],
            )

    pipeline = FinancialAnalysisPipeline()

    pipeline.register(FirstAnalyzer())
    pipeline.register(SecondAnalyzer())

    result = pipeline.run([{"value": 100.0}])

    assert execution_order == ["first", "second"]

    assert result.records == [
        {
            "value": 100.0,
            "first": True,
            "second": True,
        }
    ]

    assert result.metrics == {
        "analysis__first": 1,
        "analysis__second": 2,
    }

    assert result.metric_names == [
        "analysis__first",
        "analysis__second",
    ]


def test_financial_analysis_pipeline_does_not_mutate_input() -> None:
    class MutatingAnalyzer(BaseFinancialAnalyzer):
        @property
        def name(self) -> str:
            return "mutating"

        def analyze(self, records):
            records[0]["analyzed"] = True

            return AnalysisResult(
                records=list(records),
            )

    pipeline = FinancialAnalysisPipeline()
    pipeline.register(MutatingAnalyzer())

    original_records = [
        {"value": 100.0},
    ]

    result = pipeline.run(original_records)

    assert original_records == [
        {"value": 100.0},
    ]

    assert result.records == [
        {"value": 100.0, "analyzed": True},
    ]