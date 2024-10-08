from abc import ABC
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from beartype import beartype

from feature_fabrica.models import PromiseValue, get_execution_config


class PromiseManager(ABC):
    def __init__(self, parallel_execution: bool = True, max_workers: int = 5):
        super().__init__()
        self.execution_config = get_execution_config(parallel_execution=parallel_execution, max_workers=max_workers)
        self.promised_memo: dict[str, PromiseValue] = {}


    @beartype
    def get_promise_value(self, base_name: str, suffix: str | None = None) -> PromiseValue:
        key = self._generate_key(base_name, suffix)
        if key not in self.promised_memo:
            self.promised_memo[key] = PromiseValue()
        return self.promised_memo[key]

    @beartype
    def set_promise_value(self, promise_value: PromiseValue, base_name: str, suffix: str | None = None):
        key = self._generate_key(base_name, suffix)
        self.promised_memo[key] = promise_value

    @beartype
    def is_promised(self, base_name: str, suffix: str | None = None) -> bool:
        key = self._generate_key(base_name, suffix)
        return key in self.promised_memo

    @beartype
    def is_promised_any(self, base_name: str) -> bool:
        return any(base_name == key.split(':')[0] for key in self.promised_memo)

    @beartype
    def pass_data(self, data: np.ndarray, base_name: str, suffix: str | None = None, finally_delete_key: bool = False):
        key = self._generate_key(base_name, suffix)
        if key in self.promised_memo:
            self.promised_memo[key](data)
            if finally_delete_key:
                del self.promised_memo[key]

    @beartype
    def delete_all_related_keys(self, base_name: str):
        keys_to_delete = [key for key in self.promised_memo if base_name == key.split(':')[0]]
        for key in keys_to_delete:
            del self.promised_memo[key]

    def _generate_key(self, base_name: str, suffix: str | None) -> str:
        return base_name if suffix is None else f"{base_name}:{suffix}"

    def __call__(self, func):
        """Decorator to manage promises before executing the function."""
        def wrapper(transformation, *args, **kwargs):
            if func.__name__ == '__call__':
                if transformation.expects_executable_promise:
                    transformation_obj_id = str(id(transformation))

                    if self.execution_config.parallel_execution:
                        with ThreadPoolExecutor(max_workers=self.execution_config.max_workers) as executor:
                            keys = [
                                self._generate_key(transformation_obj_id, str(i))
                                for i in range(transformation.expects_executable_promise)
                            ]
                            promise_values = [self.promised_memo[key] for key in keys if key in self.promised_memo]
                            try:
                                list(executor.map(lambda pv: pv(), promise_values))
                            except Exception as e:
                                raise RuntimeError(f"Error executing promise: {e}")
                    else:
                        for i in range(transformation.expects_executable_promise):
                            key = self._generate_key(base_name=transformation_obj_id, suffix=str(i))
                            if key in self.promised_memo:
                                try:
                                    self.promised_memo[key]()
                                except Exception as e:
                                    raise RuntimeError(f"Error executing promise for index {i}: {e}")

                result = func(transformation, *args, **kwargs)

                if transformation._name_ and self.is_promised(transformation.feature_name, transformation._name_):
                    self.pass_data(result.value, transformation.feature_name, transformation._name_)

            return result

        return wrapper

def get_promise_manager(**kwargs):
    """Singleton pattern for PromiseManager."""
    if 'promise_memo' not in globals():
        globals()['promise_memo'] = PromiseManager(**kwargs)
    return globals()['promise_memo']
