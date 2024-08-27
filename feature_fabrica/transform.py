# transform.py
from abc import ABC
from typing import Any
from collections.abc import Iterable, Mapping
import inspect
from loguru import logger
from .core import Feature


class Transformation(ABC):
    def __init__(self) -> None:
        self.expects_data = False

    def compile(self, features: dict[str, Feature] | None) -> bool:
        if features is not None:
            for attr_name, attr_value in self.__dict__.items():
                if attr_name == "expects_data":
                    continue

                if isinstance(attr_value, str) and attr_value in features:
                    setattr(self, attr_name, features[attr_value].feature_value.value)  # type: ignore[attr-defined]
                elif isinstance(attr_value, Iterable):
                    setattr(
                        self,
                        attr_name,
                        [
                            features[item].feature_value.value  # type: ignore[attr-defined]
                            if isinstance(item, str) and item in features
                            else item
                            for item in attr_value
                        ],
                    )
                elif isinstance(attr_value, Mapping):
                    setattr(
                        self,
                        attr_name,
                        {
                            key: features[val].feature_value.value  # type: ignore[attr-defined]
                            if isinstance(val, str) and val in features
                            else val
                            for key, val in attr_value.items()
                        },
                    )
        # Check the signature of the execute method
        execute_signature = inspect.signature(self.execute)
        execute_params = execute_signature.parameters

        # Raise an error if execute expects more than one argument (excluding 'self')
        if len(execute_params) > 2:  # 'self' and one additional argument
            raise TypeError(
                f"{self.__class__.__name__}.execute expects too many arguments. "
                f"Expected 1 argument, but got {len(execute_params) - 1}."
            )
        assert hasattr(self, "expects_data")

        self.expects_data = len(execute_params) == 1
        return self.expects_data

    def execute(self, *args):
        raise NotImplementedError()

    @logger.catch
    def __call__(self, *args):
        return self.execute(*args)


class SumFn(Transformation):
    def __init__(self, iterable: list[Any] | str):
        super().__init__()
        self.iterable = iterable

    def execute(self):
        return sum(self.iterable)


class ScaleFeature(Transformation):
    def __init__(self, factor: float):
        super().__init__()
        self.factor = factor

    def execute(self, data: float):
        return data * self.factor
