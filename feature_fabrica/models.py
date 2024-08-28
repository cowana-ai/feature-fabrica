# models.py
from pydantic import BaseModel, Field, validator, root_validator
from typing import Any, Optional
import builtins


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
        # Validate that the value matches the specified data type
        expected_type = getattr(builtins, values["data_type"], None)
        if expected_type and not isinstance(v, expected_type):
            raise ValueError(
                f"Value '{v}' does not match data type '{values['data_type']}'"
            )
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
