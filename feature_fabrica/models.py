# models.py
from pydantic import BaseModel, Field, validator, root_validator
from typing import Any
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
                raise ValueError(f"Invalid data_type specified: {v}")
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
