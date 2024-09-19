from abc import ABC

import numpy as np
from beartype import beartype

from feature_fabrica.models import PromiseValue


class PromiseManager(ABC):
    def __init__(self):
        super().__init__()
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
        return any(base_name in key for key in self.promised_memo)

    @beartype
    def pass_data(self, data: np.ndarray, base_name: str, suffix: str | None = None, finally_delete_key: bool = False):
        key = self._generate_key(base_name, suffix)
        if key in self.promised_memo:
            self.promised_memo[key](data)
            if finally_delete_key:
                del self.promised_memo[key]

    @beartype
    def delete_all_related_keys(self, base_name: str):
        keys_to_delete = [key for key in self.promised_memo if base_name in key]
        for key in keys_to_delete:
            del self.promised_memo[key]

    def _generate_key(self, base_name: str, suffix: str | None) -> str:
        return base_name if suffix is None else f"{base_name}:{suffix}"

    def __call__(self, func):
        """Decorator to manage promises before executing the function."""
        def wrapper(transformation, *args, **kwargs):
            # Resolve all PromiseValues in the transformation
            if transformation.expects_promise:
                for i in range(transformation.expects_promise):
                    key = self._generate_key(base_name=str(id(transformation)), suffix=str(i))
                    promise_value = self.promised_memo[key]
                    promise_value()
                    del self.promised_memo[key]
            # Call the original transformation method
            return func(transformation, *args, **kwargs)

        return wrapper
