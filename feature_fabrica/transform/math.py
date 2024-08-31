from collections.abc import Iterable
from typing import Union

import numpy as np
from beartype import beartype
from numpy.typing import NDArray

from feature_fabrica.transform.base import Transformation

NumericArray = Union[NDArray[np.floating], NDArray[np.int_]]
NumericValue = Union[np.floating, np.int_, float, int]


class SumFn(Transformation):
    def __init__(self, iterable: Iterable):
        super().__init__()
        self.iterable = iterable

    @beartype
    def execute(self) -> NumericArray | NumericValue:
        return np.sum(self.iterable, axis=0)


class ScaleFeature(Transformation):
    def __init__(self, factor: float):
        super().__init__()
        self.factor = factor

    @beartype
    def execute(self, data: NumericArray | NumericValue) -> NumericArray | NumericValue:
        return np.multiply(data, self.factor)


class LogTransform(Transformation):
    @beartype
    def execute(self, data: NumericArray | NumericValue) -> NumericArray | NumericValue:
        return np.log(data)


class ExpTransform(Transformation):
    @beartype
    def execute(self, data: NumericArray | NumericValue) -> NumericArray | NumericValue:
        return np.exp(data)


class SqrtTransform(Transformation):
    @beartype
    def execute(self, data: NumericArray | NumericValue) -> NumericArray | NumericValue:
        return np.sqrt(data)


class PowerTransform(Transformation):
    def __init__(self, power: float):
        self.power = power

    @beartype
    def execute(self, data: NumericArray | NumericValue) -> NumericArray | NumericValue:
        return np.pow(data, self.power)


class ZScoreTransform(Transformation):
    def __init__(self, mean: float, std_dev: float):
        self.mean = mean
        self.std_dev = std_dev

    @beartype
    def execute(self, data: NumericArray | NumericValue) -> NumericArray | NumericValue:
        return (data - self.mean) / self.std_dev


class ClipTransform(Transformation):
    def __init__(self, min: float, max: float):
        self.min = min
        self.max = max

    @beartype
    def execute(self, data: NumericArray | NumericValue) -> NumericArray | NumericValue:
        return np.clip(data, self.min, self.max)


class MinMaxTransform(Transformation):
    def __init__(self, min: float, max: float):
        self.min = min
        self.max = max

    @beartype
    def execute(self, data: NumericArray | NumericValue) -> NumericArray | NumericValue:
        return (data - self.min) / (self.max - self.min)
