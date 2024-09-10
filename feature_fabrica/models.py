# models.py
import hashlib
import re
from typing import Any, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, root_validator, validator


class PromiseValue:
    value: np.ndarray | None = None


class FeatureSpec(BaseModel):
    description: str
    data_type: str
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


class FeatureValue(np.lib.mixins.NDArrayOperatorsMixin, BaseModel, validate_assignment=True):  # type: ignore[call-arg]
    model_config = ConfigDict(arbitrary_types_allowed=True)

    value: np.ndarray | PromiseValue = Field(default=None)
    data_type: str

    @root_validator(pre=True)
    def validate_value(cls, values):
        v = values.get("value")
        data_type = values.get("data_type")
        # allow PromiseValue only
        if isinstance(v, PromiseValue):
            return values
        # Get the expected data type from builtins or numpy
        expected_type = getattr(np, data_type, None)

        if expected_type is None:
            raise ValueError(f"Unsupported data type '{data_type}', use numpy typing!")

        # Validate that the array has the correct data type
        actual_type_name = v.dtype.name
        actual_type, actual_precision = re.findall(r'\D+|\d+', actual_type_name)

        expected_type_list = re.findall(r'\D+|\d+', data_type)

        if actual_type not in data_type:
            raise ValueError(
                f"Array dtype '{v.dtype}' does not match expected type '{data_type}'"
            )
        elif len(expected_type_list) > 1 and actual_precision != expected_type_list[1]:
            # convert to desired type if compatible
            v = v.astype(expected_type)

        # Update the value in the values dictionary
        values["value"] = v
        return values

    def __array__(self, dtype=None, copy=None):
        # Automatically converts to np.ndarray when passed to a function that expects an array
        if dtype:
            return self.value.astype(dtype)
        return self.value.copy() if copy else self.value

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # Convert inputs to their underlying values if they are FeatureValue instances
        inputs = tuple(x.value if isinstance(x, FeatureValue) else x for x in inputs)

        # Perform the operation using the ufunc
        result = getattr(ufunc, method)(*inputs, **kwargs)
        return result

    def __array_function__(self, func, types, args, kwargs):
        # Convert args to their underlying values if they are FeatureValue instances
        args = tuple(x.value if isinstance(x, FeatureValue) else x for x in args)
        # Perform the operation using the function
        result = func(*args, **kwargs)
        return result

    def __getitem__(self, idx):
        return self.value[idx]

    def __getattr__(self, name):
        return getattr(self.value, name)

    def __repr__(self):
        return f"FeatureValue(value={self.value})"

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
