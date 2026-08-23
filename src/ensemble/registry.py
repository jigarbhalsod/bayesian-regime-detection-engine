from typing import Dict, List, Type

from .base import BaseEnsembleStrategy


class EnsembleRegistry:
    """
    Registry for managing available ensemble strategy classes.

    Strategies are registered by their unique strategy name and can
    later be retrieved or instantiated by higher-level components.
    """

    def __init__(self) -> None:
        self._strategies: Dict[str, Type[BaseEnsembleStrategy]] = {}

    def register(
        self,
        strategy_class: Type[BaseEnsembleStrategy],
    ) -> None:
        """
        Register an ensemble strategy class.
        """

        if not isinstance(strategy_class, type):
            raise TypeError(
                "strategy_class must be a class inheriting "
                "from BaseEnsembleStrategy."
            )

        if not issubclass(strategy_class, BaseEnsembleStrategy):
            raise TypeError(
                "strategy_class must inherit from BaseEnsembleStrategy."
            )

        strategy_name = self._get_strategy_name(strategy_class)

        if strategy_name in self._strategies:
            raise ValueError(
                f"Ensemble strategy '{strategy_name}' is already registered."
            )

        self._strategies[strategy_name] = strategy_class

    def unregister(self, strategy_name: str) -> None:
        """
        Remove a registered ensemble strategy.
        """

        self._validate_strategy_name(strategy_name)

        if strategy_name not in self._strategies:
            raise KeyError(
                f"Ensemble strategy '{strategy_name}' is not registered."
            )

        del self._strategies[strategy_name]

    def get(
        self,
        strategy_name: str,
    ) -> Type[BaseEnsembleStrategy]:
        """
        Retrieve a registered ensemble strategy class.
        """

        self._validate_strategy_name(strategy_name)

        if strategy_name not in self._strategies:
            raise KeyError(
                f"Ensemble strategy '{strategy_name}' is not registered."
            )

        return self._strategies[strategy_name]

    def has(self, strategy_name: str) -> bool:
        """
        Check whether a strategy is registered.
        """

        self._validate_strategy_name(strategy_name)

        return strategy_name in self._strategies

    def list_strategies(self) -> List[str]:
        """
        Return registered strategy names in registration order.
        """

        return list(self._strategies.keys())

    def clear(self) -> None:
        """
        Remove all registered strategies.
        """

        self._strategies.clear()

    @property
    def count(self) -> int:
        """
        Number of registered strategies.
        """

        return len(self._strategies)

    @staticmethod
    def _validate_strategy_name(strategy_name: str) -> None:
        """
        Validate a strategy name.
        """

        if not isinstance(strategy_name, str):
            raise TypeError(
                "strategy_name must be a string."
            )

        if not strategy_name.strip():
            raise ValueError(
                "strategy_name must be a non-empty string."
            )

    @staticmethod
    def _get_strategy_name(
        strategy_class: Type[BaseEnsembleStrategy],
    ) -> str:
        """
        Extract and validate the strategy name from a strategy class.
        """

        strategy_name = getattr(strategy_class, "STRATEGY_NAME", None)

        if not isinstance(strategy_name, str) or not strategy_name.strip():
            raise ValueError(
                "Registered ensemble strategy classes must define a "
                "non-empty STRATEGY_NAME string."
            )

        return strategy_name.strip()