# models.py
import hashlib
from collections.abc import Callable
from typing import Any, Optional

import numpy as np
from beartype import beartype
from pydantic import BaseModel, ConfigDict, Field, validator

from feature_fabrica.utils import compute_all_transformations


class FeatureSpec(BaseModel):
    description: str
    data_type: str
    group: str  | None = None
    dependencies: list[str] | None = Field(default_factory=list)
    transformation: dict | None = Field(default_factory=dict)

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

class ArrayLike(np.lib.mixins.NDArrayOperatorsMixin):
    def __getattr__(self, name):
        return getattr(self.value, name)

    def __array__(self, dtype=None, copy=None):
        # Automatically converts to np.ndarray when passed to a function that expects an array
        if dtype:
            return self.value.astype(dtype)
        return self.value.copy() if copy else self.value

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # Convert inputs to their underlying values if they are FeatureValue instances
        inputs = tuple(x.value if isinstance(x, ArrayLike) else x for x in inputs)

        # Perform the operation using the ufunc
        result = getattr(ufunc, method)(*inputs, **kwargs)
        return result

    def __array_function__(self, func, types, args, kwargs):
        # Convert args to their underlying values if they are FeatureValue instances
        args = tuple(x.value if isinstance(x, ArrayLike) else x for x in args)
        # Perform the operation using the function
        result = func(*args, **kwargs)
        return result

    def __getitem__(self, idx):
        return self.value[idx]


class PromiseValue(ArrayLike, BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    value: np.ndarray = Field(default=None)
    data_type: str | None = None
    transformation: Callable | dict[str, Callable] | None = None

    @beartype
    def __call__(self, data: np.ndarray | None = None):
        if self.transformation is not None:
            result = compute_all_transformations(self.transformation)
            self._set_value(result.value)
        else:

            self._set_value(data)

    def _set_value(self, value: np.ndarray | None):
        """Internal method to set value and transform to FeatureValue."""
        self.value = value

    def __repr__(self):
        return f"PromiseValue(value={self._value})"


class TNode(BaseModel):
    transformation_name: str
    start_time: float
    end_time: float
    shape: tuple | None = None
    time_taken: float | None = None
    output_hash: str | None = None
    next: Optional["TNode"] = None  # Forward reference

    def to_dict(self) -> dict[str, Any]:
        # Convert the node and its next nodes to a dictionary
        node_dict = {
            "value": self.value,
            "transformation_name": self.transformation_name,
        }
        if self.next:
            node_dict["next"] = self.next.to_dict()
        else:
            node_dict["next"] = None
        return node_dict

    def compute_hash(self, data: np.ndarray) -> str:
        data_bytes = data.tobytes()
        return hashlib.sha256(data_bytes).hexdigest()

    def store_hash_and_shape(self, output_data: np.ndarray):
        self.shape = output_data.shape
        self.output_hash = self.compute_hash(output_data)

    def finilize_metrics(self) -> None:
        self.time_taken = self.end_time - self.start_time


class THead(BaseModel):
    next: TNode | None = None
