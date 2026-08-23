from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class ValidationPipelineResult:
    """
    Result produced by the validation pipeline.
    """

    results: dict[str, Any]
    executed_validators: tuple[str, ...]
    failed_validators: tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        return len(self.failed_validators) == 0

    @property
    def validator_count(self) -> int:
        return len(self.executed_validators)


class ValidationPipeline:
    """
    Execute registered validation functions in order.
    """

    def __init__(self) -> None:
        self._validators: list[tuple[str, Callable[..., Any]]] = []

    def register(
        self,
        name: str,
        validator: Callable[..., Any],
    ) -> None:
        self._validate_name(name)
        self._validate_validator(validator)

        if self.has_validator(name):
            raise ValueError(
                f"Validator '{name}' is already registered"
            )

        self._validators.append((name, validator))

    def unregister(self, name: str) -> None:
        self._validate_name(name)

        for index, (registered_name, _) in enumerate(
            self._validators
        ):
            if registered_name == name:
                del self._validators[index]
                return

        raise KeyError(
            f"Validator '{name}' is not registered"
        )

    def has_validator(self, name: str) -> bool:
        self._validate_name(name)

        return any(
            registered_name == name
            for registered_name, _ in self._validators
        )

    @property
    def validator_names(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, _ in self._validators
        )

    def run(
        self,
        *args: Any,
        raise_on_error: bool = False,
        **kwargs: Any,
    ) -> ValidationPipelineResult:
        if not isinstance(raise_on_error, bool):
            raise TypeError(
                "raise_on_error must be a boolean"
            )

        results: dict[str, Any] = {}
        executed: list[str] = []
        failed: list[str] = []

        for name, validator in self._validators:
            executed.append(name)

            try:
                results[name] = validator(
                    *args,
                    **kwargs,
                )
            except Exception:
                failed.append(name)

                if raise_on_error:
                    raise

        return ValidationPipelineResult(
            results=results,
            executed_validators=tuple(executed),
            failed_validators=tuple(failed),
        )

    @staticmethod
    def _validate_name(name: str) -> None:
        if not isinstance(name, str):
            raise TypeError(
                "Validator name must be a string"
            )

        if not name.strip():
            raise ValueError(
                "Validator name cannot be empty"
            )

    @staticmethod
    def _validate_validator(
        validator: Callable[..., Any],
    ) -> None:
        if not callable(validator):
            raise TypeError(
                "Validator must be callable"
            )