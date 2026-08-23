from __future__ import annotations

from typing import Any, Callable

from src.integration.config import IntegrationConfig
from src.integration.registry import ModelRegistry
from src.integration.result import IntegrationResult


class IntegrationService:
    """
    Execute registered models through a common integration workflow.

    The service does not require models to implement a particular
    interface. Instead, an executor function is supplied at execution
    time and receives each registered model plus the provided inputs.
    """

    def __init__(
        self,
        registry: ModelRegistry | None = None,
        config: IntegrationConfig | None = None,
    ) -> None:
        if registry is not None and not isinstance(
            registry,
            ModelRegistry,
        ):
            raise TypeError(
                "registry must be a ModelRegistry or None."
            )

        if config is not None and not isinstance(
            config,
            IntegrationConfig,
        ):
            raise TypeError(
                "config must be an IntegrationConfig or None."
            )

        self._registry = (
            registry
            if registry is not None
            else ModelRegistry()
        )

        self._config = (
            config
            if config is not None
            else IntegrationConfig()
        )

    @property
    def registry(self) -> ModelRegistry:
        """
        Return the model registry used by this service.
        """
        return self._registry

    @property
    def config(self) -> IntegrationConfig:
        """
        Return the integration configuration.
        """
        return self._config

    def execute(
        self,
        executor: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> IntegrationResult:
        """
        Execute all registered models.

        Parameters
        ----------
        executor:
            Callable invoked as ``executor(model, *args, **kwargs)``.
        *args:
            Positional arguments forwarded to every model execution.
        **kwargs:
            Keyword arguments forwarded to every model execution.

        Returns
        -------
        IntegrationResult
            Aggregated outputs and errors from the execution run.

        Raises
        ------
        TypeError
            If executor is not callable.
        ValueError
            If fewer than ``min_models`` are registered.
        Exception
            Any model execution exception when ``fail_fast=True``.
        """
        if not callable(executor):
            raise TypeError(
                "executor must be callable."
            )

        registered_models = self._registry.count

        if registered_models < self._config.min_models:
            raise ValueError(
                "insufficient registered models: "
                f"required at least "
                f"{self._config.min_models}, "
                f"got {registered_models}."
            )

        outputs: dict[str, Any] = {}
        errors: dict[str, str] = {}

        for name in self._registry.names:
            model = self._registry.require(name)

            try:
                outputs[name] = executor(
                    model,
                    *args,
                    **kwargs,
                )
            except Exception as exc:
                if self._config.fail_fast:
                    raise

                errors[name] = str(exc) or exc.__class__.__name__

        metadata: dict[str, Any] = {}

        if self._config.include_metadata:
            metadata = {
                "integration_name": (
                    self._config.integration_name
                ),
                "registered_models": registered_models,
                "successful_models": len(outputs),
                "failed_models": len(errors),
            }

        return IntegrationResult(
            outputs=outputs,
            errors=errors,
            metadata=metadata,
        )