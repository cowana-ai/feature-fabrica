from abc import ABC

import numpy as np
from beartype import beartype

from feature_fabrica.models import PromiseValue


class PromiseManager(ABC):
    def __init__(self):
        super().__init__()
        self.promised_memo: dict[str, PromiseValue] = {}
    @beartype
    def get_promise_value(self, feature: str, transform_stage: str | None=None) -> PromiseValue:
        key = feature if transform_stage is None else feature + ':' + transform_stage
        if key in self.promised_memo:
            return self.promised_memo[key]
        new_promise_value = PromiseValue()
        self.promised_memo[key] = new_promise_value
        return new_promise_value
    @beartype
    def is_promised(self, feature: str, transform_stage: str | None=None) -> bool:
        key = feature if transform_stage is None else feature + ':' + transform_stage
        return key in self.promised_memo
    @beartype
    def is_promised_any(self, feature: str) -> bool:
        for key in self.promised_memo.keys():
            if feature in key:
                return True
        return False
    @beartype
    def pass_data(self, data: np.ndarray, feature: str, transform_stage: str | None=None, finally_delete_key: bool = False):
        key = feature if transform_stage is None else feature + ':' + transform_stage
        self.promised_memo[key](data)
        if finally_delete_key:
            del self.promised_memo[key]
    @beartype
    def delete_all_related_keys(self, feature: str):
        all_keys = list(self.promised_memo.keys())
        for key in all_keys:
            if feature in key:
                del self.promised_memo[key]
