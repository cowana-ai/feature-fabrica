from __future__ import annotations

import inspect
import time
from abc import ABC
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from easydict import EasyDict as edict

from feature_fabrica.models import PromiseValue
from feature_fabrica.promise_manager import get_promise_manager
from feature_fabrica.utils import get_logger, is_dict_like, is_list_like

if TYPE_CHECKING:
    from feature_fabrica.core import Feature

logger = get_logger()
promise_manager = get_promise_manager()

class Transformation(ABC):
    def __init__(self) -> None:
        self.expects_data = False
        self.expects_promise = 0

    def compile(self, features: dict[str, Feature] | None = None) -> bool:
        promise_count = 0
        if features is not None:
            memo: dict[str, int] = defaultdict(int)
            memo["expects_data"] = 1
            memo["expects_promise"] = 1
            for attr_name, attr_value in self.__dict__.items():
                if memo[attr_name] == 1:
                    continue
                stack: list[tuple[Any, Any, Any]] = [(self, attr_name, attr_value)]
                while stack:
                    cur_obj, cur_name, cur_value = stack.pop()
                    if not is_list_like(cur_obj) and not is_dict_like(cur_obj):
                        key = cur_obj.__class__.__name__ + '.' + cur_name
                        memo[key] = 1

                    # If cur_value is str and in features -> resolved immediately
                    if isinstance(cur_value, str):
                        if cur_value in features:
                            cur_value = features[cur_value].feature_value
                    # If cur_value is Transformation -> compile it -> resolved immediately
                    elif isinstance(cur_value, Transformation):
                        cur_value.compile(features)
                        continue

                    # If cur_value is PromiseValue -> resolve recursively
                    elif isinstance(cur_value, PromiseValue):
                        if cur_value.transformation is not None:
                            stack.append((cur_value, 'transformation', cur_value.transformation))
                            promise_manager.set_promise_value(cur_value, base_name=str(id(self)), suffix=str(promise_count))
                            promise_count += 1
                        continue

                    # If cur_value is Iterable -> iterate recursively
                    elif is_list_like(cur_value):
                        for idx, item in enumerate(cur_value):
                            stack.append((cur_value, idx, item))
                        continue

                    # If cur_value is Mapping -> iterate over key-value pairs
                    elif is_dict_like(cur_value):
                        for key, val in cur_value.items():
                            stack.append((cur_value, key, val))
                        continue
                    # Resolve here
                    # Set resolved values appropriately
                    if is_list_like(cur_obj) or is_dict_like(cur_obj):
                        cur_obj[cur_name] = cur_value
                    else:
                        setattr(cur_obj, cur_name, cur_value)



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
        self.expects_promise = promise_count
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
