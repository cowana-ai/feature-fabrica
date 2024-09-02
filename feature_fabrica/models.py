# models.py
import hashlib
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


class FeatureValue(BaseModel, validate_assignment=True):  # type: ignore[call-arg]
    model_config = ConfigDict(arbitrary_types_allowed=True)

    value: np.ndarray | PromiseValue | None = Field(default=None)
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
            raise ValueError(f"Unsupported data type '{data_type}'")

        # Validate that the array has the correct data type
        if v.dtype.type != expected_type:
            raise ValueError(
                f"Array dtype '{v.dtype}' does not match expected type '{data_type}'"
            )

        # Update the value in the values dictionary
        values["value"] = v
        return values

    def __add__(self, other):
        if isinstance(other, FeatureValue):
            return self.value + other.value
        return self.value + other

    def __sub__(self, other):
        if isinstance(other, FeatureValue):
            return self.value - other.value
        return self.value - other

    def __mul__(self, other):
        if isinstance(other, FeatureValue):
            return self.value * other.value
        return self.value * other

    def __truediv__(self, other):
        if isinstance(other, FeatureValue):
            return self.value / other.value
        return self.value / other

    def __getattr__(self, name):
        # Delegate attribute access to the `value` attribute
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
