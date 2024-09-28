from __future__ import annotations

import inspect
import time
from abc import ABC
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from easydict import EasyDict as edict
from omegaconf import OmegaConf

from feature_fabrica.models import PromiseValue
from feature_fabrica.promise_manager import get_promise_manager
from feature_fabrica.transform.registry import TransformationRegistry

if TYPE_CHECKING:
    from feature_fabrica.core import Feature

promise_manager = get_promise_manager()

class Transformation(ABC):
    _name_: str | None = None

    def __init__(self) -> None:
        self.feature_name: str | None = None # type: ignore
        self.expects_data = False
        self.expects_executable_promise = 0

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        TransformationRegistry.register(cls)

    def compile(self, feature_name: str, feature_dependencies: dict[str, Feature] | None = None) -> bool:
        # bind feature name
        self.feature_name = feature_name

        executable_promise_count = 0
        if feature_dependencies is not None:
            memo: dict[str, int] = defaultdict(int)
            memo["expects_data"] = 1
            memo["expects_promise"] = 1
            for attr_name, attr_value in self.__dict__.items():
                if memo[attr_name] == 1:
                    continue
                stack: list[tuple[Any, Any, Any]] = [(self, attr_name, attr_value)]
                while stack:
                    cur_obj, cur_attr, cur_value = stack.pop()
                    if not OmegaConf.is_list(cur_obj) and not OmegaConf.is_dict(cur_obj):
                        key = cur_obj.__class__.__name__ + '.' + cur_attr
                        memo[key] = 1

                    # If cur_value is str and in features -> resolved immediately
                    if isinstance(cur_value, str):
                        if cur_value in feature_dependencies:
                            cur_value = feature_dependencies[cur_value].feature_value
                    # If cur_value is Transformation -> compile it -> resolved immediately
                    elif isinstance(cur_value, Transformation):
                        cur_value.compile(feature_name, feature_dependencies)
                        continue

                    # If cur_value is PromiseValue -> resolve recursively
                    elif isinstance(cur_value, PromiseValue):
                        if cur_value.transformation is not None:
                            stack.append((cur_value, 'transformation', cur_value.transformation))
                            promise_manager.set_promise_value(cur_value, base_name=str(id(self)), suffix=str(executable_promise_count))
                            executable_promise_count += 1
                        continue

                    # If cur_value is Iterable -> iterate recursively
                    elif OmegaConf.is_list(cur_value):
                        for idx, item in enumerate(cur_value):
                            stack.append((cur_value, idx, item))
                        continue

                    # If cur_value is Mapping -> iterate over key-value pairs
                    elif OmegaConf.is_dict(cur_value):
                        for key, val in cur_value.items():
                            stack.append((cur_value, key, val))
                        continue
                    # Resolve here
                    # Set resolved values appropriately
                    if OmegaConf.is_list(cur_obj) or OmegaConf.is_dict(cur_obj):
                        cur_obj[cur_attr] = cur_value
                    else:
                        setattr(cur_obj, cur_attr, cur_value)



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
        self.expects_executable_promise = executable_promise_count
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
