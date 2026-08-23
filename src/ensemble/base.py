from abc import ABC, abstractmethod
from typing import Any, Iterable

from .config import EnsembleConfig
from .result import EnsembleResult


class BaseEnsembleStrategy(ABC):
    """
    Abstract base class for ensemble strategies.

    Every ensemble strategy must implement a common execution
    contract so that strategies can be registered, selected,
    and executed consistently.
    """

    def __init__(self, config: EnsembleConfig | None = None) -> None:
        self.config = config or EnsembleConfig()
        self.config.validate()

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """
        Unique name identifying the ensemble strategy.
        """
        raise NotImplementedError

    @abstractmethod
    def combine(self, model_outputs: Iterable[Any]) -> EnsembleResult:
        """
        Combine model outputs into one standardized ensemble result.

        Parameters
        ----------
        model_outputs:
            Iterable of prepared and validated model outputs.

        Returns
        -------
        EnsembleResult
            Standardized ensemble output.
        """
        raise NotImplementedError

    def validate_result(self, result: EnsembleResult) -> EnsembleResult:
        """
        Validate and return an ensemble result.
        """

        if not isinstance(result, EnsembleResult):
            raise TypeError(
                "Ensemble strategies must return an EnsembleResult."
            )

        result.validate()
        return result