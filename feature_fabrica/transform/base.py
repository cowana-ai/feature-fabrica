from __future__ import annotations

import inspect
import time
from abc import ABC
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

from easydict import EasyDict as edict

from feature_fabrica.models import PromiseValue
from feature_fabrica.utils import get_logger, get_promise_manager

if TYPE_CHECKING:
    from feature_fabrica.core import Feature

logger = get_logger()
promise_manager = get_promise_manager()

class Transformation(ABC):
    def __init__(self) -> None:
        self.expects_data = False

    def compile(self, features: dict[str, Feature] | None = None) -> bool:
        if features is not None:
            for attr_name, attr_value in self.__dict__.items():
                if attr_name == "expects_data":
                    continue

                if isinstance(attr_value, str) and attr_value in features:
                    setattr(self, attr_name, features[attr_value].feature_value)
                elif isinstance(attr_value, Transformation):
                    attr_value.compile(features)
                elif isinstance(attr_value, PromiseValue) and isinstance(attr_value.apply_transform, Transformation):
                    attr_value.apply_transform.compile(features) # type: ignore
                elif isinstance(attr_value, Iterable):
                    setattr(
                        self,
                        attr_name,
                        [
                            features[item].feature_value
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
                            key: features[val].feature_value
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
        # assert hasattr(self, "expects_data")

        self.expects_data = len(execute_params) == 1
        return self.expects_data

    def execute(self, *args):
        raise NotImplementedError()

    @promise_manager
    def __call__(self, *args):
        # Start time
        start_time = time.time()

        value = self.execute(*args)
        # End time
        end_time = time.time()
        return edict(
            start_time=start_time,
            value=value,
            end_time=end_time,
        )
