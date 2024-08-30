# models.py
from pydantic import BaseModel, Field, validator, root_validator
from typing import Any, Optional
import builtins
import numpy as np


class FeatureSpec(BaseModel):
    description: str
    data_type: str
    dependencies: list[str] | None = Field(default_factory=list)
    transformation: dict | None = Field(default_factory=dict)

    @validator("data_type")
    def validate_data_type(cls, v):
        try:
            # Check if the data_type is a valid Python type
            if v not in dir(builtins):
                raise ValueError(
                    f"Invalid data_type specified: {v}, it should be in builtins"
                )
        except Exception as e:
            raise ValueError(f"Invalid data_type: {v}, error: {str(e)}")
        return v


class FeatureValue(BaseModel):
    value: Any
    data_type: str

    @root_validator(pre=True)
    def validate_value(cls, values):
        v = values.get("value")
        data_type = values.get("data_type")

        # Get the expected data type from builtins or numpy
        expected_type = getattr(builtins, data_type, None)
        if not expected_type:
            expected_type = getattr(np, data_type, None)

        if expected_type is None:
            raise ValueError(f"Unsupported data type '{data_type}'")

        # Convert value to a numpy array if it's not already an array
        if not isinstance(v, np.ndarray):
            try:
                v = np.array(v, dtype=expected_type)
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"Value '{v}' cannot be converted to array of type '{data_type}': {e}"
                )
        else:
            # Validate that the array has the correct data type
            if v.dtype != expected_type:
                raise ValueError(
                    f"Array dtype '{v.dtype}' does not match expected type '{data_type}'"
                )

        # Update the value in the values dictionary
        values["value"] = v
        return values


class TNode(BaseModel):
    value: Any
    transformation_name: str
    start_time: float
    end_time: float
    time_taken: float | None = None
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

    def finilize_metrics(self) -> None:
        self.time_taken = self.end_time - self.start_time


class THead(BaseModel):
    next: TNode | None = None
