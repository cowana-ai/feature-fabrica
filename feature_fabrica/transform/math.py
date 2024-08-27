from typing import Any
from .base import Transformation


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
