from __future__ import annotations

from typing import Any, Iterable

from src.integration.adapter import ModelOutputAdapter
from src.integration.input import AdvancedModelInput
from src.integration.registry import ModelRegistry
from src.integration.result import IntegrationResult


class ModelOrchestrator:
    """
    Coordinate registered advanced models using the shared
    input and output contracts.
    """

    def __init__(
        self,
        registry: ModelRegistry,
        adapter: ModelOutputAdapter | None = None,
    ) -> None:
        if not isinstance(registry, ModelRegistry):
            raise TypeError(
                "registry must be a ModelRegistry."
            )

        if adapter is not None and not isinstance(
            adapter,
            ModelOutputAdapter,
        ):
            raise TypeError(
                "adapter must be a ModelOutputAdapter or None."
            )

        self._registry = registry
        self._adapter = (
            adapter
            if adapter is not None
            else ModelOutputAdapter()
        )

    @property
    def registry(self) -> ModelRegistry:
        """Return the orchestrator model registry."""
        return self._registry

    @property
    def adapter(self) -> ModelOutputAdapter:
        """Return the orchestrator output adapter."""
        return self._adapter

    def run(
        self,
        model_input: AdvancedModelInput,
        model_names: Iterable[str] | None = None,
    ) -> IntegrationResult:
        """
        Execute registered models.

        Model failures are captured in the result rather than
        preventing other requested models from executing.
        """
        if not isinstance(
            model_input,
            AdvancedModelInput,
        ):
            raise TypeError(
                "model_input must be an AdvancedModelInput."
            )

        names = self._resolve_model_names(
            model_names
        )

        outputs: dict[str, Any] = {}
        errors: dict[str, str] = {}

        for model_name in names:
            try:
                model = self._registry.get(
                    model_name
                )

                output = self._execute_model(
                    model,
                    model_input,
                )

                outputs[model_name] = (
                    self._adapter.adapt(
                        model_name,
                        output,
                    )
                )

            except Exception as exc:
                errors[model_name] = (
                    f"{type(exc).__name__}: {exc}"
                )

        return IntegrationResult(
            outputs=outputs,
            errors=errors,
            metadata={
                "requested_models": list(names),
                "successful_models": list(
                    outputs.keys()
                ),
                "failed_models": list(
                    errors.keys()
                ),
            },
        )

    def _resolve_model_names(
        self,
        model_names: Iterable[str] | None,
    ) -> tuple[str, ...]:
        """
        Resolve the models to execute.
        """
        if model_names is None:
            return tuple(
                self._registry.names
            )

        if isinstance(
            model_names,
            (str, bytes),
        ):
            raise TypeError(
                "model_names must be an iterable "
                "of model names or None."
            )

        try:
            names = tuple(model_names)
        except TypeError as exc:
            raise TypeError(
                "model_names must be an iterable "
                "of model names or None."
            ) from exc

        if not names:
            raise ValueError(
                "model_names cannot be empty."
            )

        for name in names:
            if not isinstance(name, str):
                raise TypeError(
                    "each model name must be a string."
                )

            if not name.strip():
                raise ValueError(
                    "model names cannot be empty."
                )

        return names

    @staticmethod
    def _execute_model(
        model: Any,
        model_input: AdvancedModelInput,
    ) -> Any:
        """
        Execute a model through its supported prediction method.

        Preferred order:
        1. predict(AdvancedModelInput)
        2. run(AdvancedModelInput)
        3. callable(AdvancedModelInput)
        """
        predict = getattr(
            model,
            "predict",
            None,
        )

        if callable(predict):
            return predict(model_input)

        run = getattr(
            model,
            "run",
            None,
        )

        if callable(run):
            return run(model_input)

        if callable(model):
            return model(model_input)

        raise TypeError(
            "registered model must provide predict(), "
            "run(), or be callable."
        )