from typing import Any

from beartype import beartype
from omegaconf import ListConfig

from feature_fabrica.models.features import PromiseValue
from feature_fabrica.transform.base import Transformation
from feature_fabrica.transform.utils import AnyArray, is_valid_numpy_dtype


class AsType(Transformation):
    _name_ = "astype"
    @beartype
    def __init__(self, dtype: str):
        if not is_valid_numpy_dtype(dtype):
            raise ValueError(f"dtype = {dtype} is not valid numpy data type!")

        self.dtype = dtype
    @beartype
    def execute(self, data: AnyArray) -> AnyArray:
        return data.astype(self.dtype)

class ListAggregation(Transformation):
    @beartype
    def __init__(self, iterable: ListConfig[Any]):
        self.iterable = iterable
    @beartype
    def execute(self) -> list[Any]:
        return [o.value if isinstance(o, PromiseValue) else o for o in self.iterable]
