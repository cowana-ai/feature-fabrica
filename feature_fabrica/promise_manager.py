from abc import ABC

import numpy as np
from beartype import beartype

from feature_fabrica.models import PromiseValue


class PromiseManager(ABC):
    def __init__(self):
        super().__init__()
        self.promised_memo: dict[str, PromiseValue] = {}

    @beartype
    def get_promise_value(self, feature: str, transform_stage: str | None = None) -> PromiseValue:
        key = self._generate_key(feature, transform_stage)
        if key not in self.promised_memo:
            self.promised_memo[key] = PromiseValue()
        return self.promised_memo[key]

    @beartype
    def is_promised(self, feature: str, transform_stage: str | None = None) -> bool:
        key = self._generate_key(feature, transform_stage)
        return key in self.promised_memo

    @beartype
    def is_promised_any(self, feature: str) -> bool:
        return any(feature in key for key in self.promised_memo)

    @beartype
    def pass_data(self, data: np.ndarray, feature: str, transform_stage: str | None = None, finally_delete_key: bool = False):
        key = self._generate_key(feature, transform_stage)
        if key in self.promised_memo:
            self.promised_memo[key](data)
            if finally_delete_key:
                del self.promised_memo[key]

    @beartype
    def delete_all_related_keys(self, feature: str):
        keys_to_delete = [key for key in self.promised_memo if feature in key]
        for key in keys_to_delete:
            del self.promised_memo[key]

    def _generate_key(self, feature: str, transform_stage: str | None) -> str:
        return feature if transform_stage is None else f"{feature}:{transform_stage}"

    def __call__(self, func):
        """Decorator to manage promises before executing the function."""
        def wrapper(transformation, *args, **kwargs):
            # Resolve all PromiseValues in the transformation
            for attr_name, attr_value in transformation.__dict__.items():
                if isinstance(attr_value, PromiseValue):
                    attr_value()
                    assert not isinstance(attr_value, PromiseValue)
            # Call the original transformation method
            return func(transformation, *args, **kwargs)

        return wrapper
