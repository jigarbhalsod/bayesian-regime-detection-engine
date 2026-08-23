from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class IntegrationResult:
    """
    Result produced by the cross-model integration layer.

    Parameters
    ----------
    outputs:
        Mapping of model names to their successful outputs.
    errors:
        Mapping of model names to error messages for failed models.
    metadata:
        Optional metadata describing the integration run.
    """

    outputs: dict[str, Any] = field(
        default_factory=dict
    )

    errors: dict[str, str] = field(
        default_factory=dict
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self._validate_mapping(
            self.outputs,
            "outputs",
        )

        self._validate_mapping(
            self.errors,
            "errors",
        )

        self._validate_mapping(
            self.metadata,
            "metadata",
        )

        for name in self.outputs:
            self._validate_model_name(
                name
            )

        for name, error in self.errors.items():
            self._validate_model_name(
                name
            )

            if not isinstance(error, str):
                raise TypeError(
                    "error messages must be strings."
                )

            if not error.strip():
                raise ValueError(
                    "error messages must not be empty."
                )

        duplicate_names = (
            set(self.outputs)
            & set(self.errors)
        )

        if duplicate_names:
            raise ValueError(
                "a model cannot exist in both "
                "outputs and errors."
            )

        object.__setattr__(
            self,
            "outputs",
            dict(self.outputs),
        )

        object.__setattr__(
            self,
            "errors",
            dict(self.errors),
        )

        object.__setattr__(
            self,
            "metadata",
            dict(self.metadata),
        )

    @staticmethod
    def _validate_mapping(
        value: Any,
        field_name: str,
    ) -> None:
        if not isinstance(value, dict):
            raise TypeError(
                f"{field_name} must be a dictionary."
            )

    @staticmethod
    def _validate_model_name(
        value: Any,
    ) -> None:
        if not isinstance(value, str):
            raise TypeError(
                "model names must be strings."
            )

        if not value.strip():
            raise ValueError(
                "model names must not be empty."
            )

    @property
    def successful_models(self) -> int:
        """
        Return the number of successful models.
        """
        return len(self.outputs)

    @property
    def failed_models(self) -> int:
        """
        Return the number of failed models.
        """
        return len(self.errors)

    @property
    def total_models(self) -> int:
        """
        Return the total number of processed models.
        """
        return (
            self.successful_models
            + self.failed_models
        )

    @property
    def success(self) -> bool:
        """
        Return True when no model failures occurred.
        """
        return self.failed_models == 0

    def get_output(
        self,
        model_name: str,
        default: Any = None,
    ) -> Any:
        """
        Return a model output or the supplied default.
        """
        return self.outputs.get(
            model_name,
            default,
        )

    def get_error(
        self,
        model_name: str,
        default: Any = None,
    ) -> Any:
        """
        Return a model error or the supplied default.
        """
        return self.errors.get(
            model_name,
            default,
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Return the integration result as a dictionary.
        """
        return {
            "outputs": dict(self.outputs),
            "errors": dict(self.errors),
            "metadata": dict(self.metadata),
            "successful_models": self.successful_models,
            "failed_models": self.failed_models,
            "total_models": self.total_models,
            "success": self.success,
        }