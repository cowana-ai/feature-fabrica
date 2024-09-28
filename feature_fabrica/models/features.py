from collections.abc import Callable
from typing import Any, Literal

import numpy as np
from beartype import beartype
from omegaconf import DictConfig
from pydantic import BaseModel, ConfigDict, Field, validator

from feature_fabrica._internal.compute import compute_all_transformations
from feature_fabrica.models.arrays import ArrayLike


class FeatureSpec(BaseModel):
    description: str =  Field(min_length=5)
    data_type: str
    group: str  | None = None
    dependencies: list[str] | None = Field(default_factory=list)
    transformation: dict[str, Any] | None = Field(default_factory=dict)

    @validator("data_type")
    def validate_data_type(cls, v):
        try:
            # Check if the data_type is a valid type
            if not (getattr(np, v, None)):
                raise ValueError(
                    f"Invalid data_type specified: {v}, it should be in numpy"
                )
        except Exception as e:
            raise ValueError(f"Invalid data_type: {v}, error: {str(e)}")
        return v

class PromiseValue(ArrayLike, BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    value: np.ndarray | None = Field(default=None, exclude=True)
    data_type: str | None = Field(default=None)
    cast: bool = Field(default=True)
    casting: Literal['safe', 'unsafe'] = Field(default="unsafe")
    transformation: Callable | DictConfig | None = Field(default=None)

    def _get_value(self):
        return self.value

    def _set_value(self, value: np.ndarray | None):
        """Internal method to set value and transform to FeatureValue."""
        if self.data_type is not None:
            self._validate_and_cast_value(value)
        self.value = value

    @beartype
    def __call__(self, data: np.ndarray | None = None):
        if self.transformation is not None:
            result = compute_all_transformations(self.transformation)
            self._set_value(result.value)
        elif data is not None:
            self._set_value(data)
        else:
            raise ValueError("Either transfromation or data should be set!")

    def _validate_and_cast_value(self, value: np.ndarray | None):
        """Validates the value's data type and shape."""
        # Ensure data type is set before validation
        if self.data_type is None:
            raise ValueError("data_type must be specified for validation.")

        # Get the expected data type from numpy
        try:
            expected_dtype = getattr(np, self.data_type)
        except AttributeError:
            raise ValueError(f"Unsupported data type '{self.data_type}', use valid numpy dtype!")

        # Check if the value is a NumPy array
        if not isinstance(value, np.ndarray):
            raise ValueError(f"Value must be a NumPy array, got {type(value).__name__} instead.")

        if value.dtype.type is expected_dtype:
            return

        if self.cast:
            # Validate that the array dtype matches or is compatible with the expected dtype
            if not (np.issubdtype(value.dtype.type, expected_dtype) or np.can_cast(value.dtype.type, expected_dtype, casting=self.casting)):
                raise ValueError(
                    f"Array dtype '{value.dtype}' does not match or is not compatible with expected type '{self.data_type}'"
                )
            value = value.astype(expected_dtype)

    def __repr__(self):
        return f"PromiseValue(value={self._value})"
