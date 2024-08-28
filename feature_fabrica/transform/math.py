from typing import Any
from .base import Transformation
import math


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

    def execute(self, data: float) -> float:
        return data * self.factor


class LogTransform(Transformation):
    def execute(self, data: float) -> float:
        return math.log(data)


class ExpTransform(Transformation):
    def execute(self, data: float) -> float:
        return math.exp(data)


class SqrtTransform(Transformation):
    def execute(self, data: float) -> float:
        return math.sqrt(data)


class PowerTransform(Transformation):
    def __init__(self, power: float):
        self.power = power

    def execute(self, data: float) -> float:
        return math.pow(data, self.power)


class ZScoreTransform(Transformation):
    def __init__(self, mean: float, std_dev: float):
        self.mean = mean
        self.std_dev = std_dev

    def execute(self, data: float) -> float:
        return (data - self.mean) / self.std_dev


class ClipTransform(Transformation):
    def __init__(self, min: float, max: float):
        self.min = min
        self.max = max

    def execute(self, data: float) -> float:
        return min(max(data, self.min), self.max)


class MinMaxTransform(Transformation):
    def __init__(self, min: float, max: float):
        self.min = min
        self.max = max

    def execute(self, data: float) -> float:
        return (data - self.min) / (self.max - self.min)
