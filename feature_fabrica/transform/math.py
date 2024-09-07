from collections.abc import Iterable

import numpy as np
from beartype import beartype

from feature_fabrica.transform.base import Transformation
from feature_fabrica.transform.utils import (
    NumericArray, NumericValue, broadcast_and_normalize_numeric_array)


class BaseReduce(Transformation):
    ufunc = None
    def __init__(self, iterable: Iterable | None = None, expect_data: bool = False, axis: int = 0):
        super().__init__()

        assert iterable or expect_data, "Either expect_data or iterable should be set!"
        self.iterable = iterable
        self.axis = axis
        if not expect_data and self.iterable:
            self.execute = self.default  # type: ignore[method-assign]
        elif expect_data and not self.iterable:
            self.execute = self.with_data  # type: ignore[method-assign]
        elif expect_data and self.iterable:
            self.execute = self.with_data_and_iterable # type: ignore[method-assign]
    @beartype
    def default(self) -> NumericArray | NumericValue:
        if self.ufunc is None:
            raise NotImplementedError()
        iterable: NumericArray = broadcast_and_normalize_numeric_array(self.iterable)
        return self.ufunc.reduce(iterable, axis=self.axis)
    @beartype
    def with_data(self, data: NumericArray) -> NumericArray | NumericValue:
        if self.ufunc is None:
            raise NotImplementedError()
        return self.ufunc.reduce(data, axis=self.axis)
    @beartype
    def with_data_and_iterable(self, data: NumericArray) -> NumericArray | NumericValue:
        if self.ufunc is None:
            raise NotImplementedError()
        data_and_iterable: NumericArray = broadcast_and_normalize_numeric_array([data] + self.iterable)
        return self.ufunc.reduce(data_and_iterable, axis=self.axis)

class SumReduce(BaseReduce):
    ufunc = np.add

class MultiplyReduce(BaseReduce):
    ufunc = np.multiply

class SubtractReduce(BaseReduce):
    ufunc = np.subtract

class DivideTransform(Transformation):
    def __init__(
        self,
        numerator: str | float | None = None,
        denominator: str | float | None = None,
    ):
        assert (
            numerator or denominator
        ), "You have to pass either numerator or denominator for computation!"

        self.numerator = numerator
        self.denominator = denominator

        if numerator and denominator:
            self.execute = self.default  # type: ignore[method-assign]
        elif numerator:
            self.execute = self.with_numerator  # type: ignore[method-assign]
        else:
            self.execute = self.with_denominator  # type: ignore[method-assign]

    @beartype
    def with_numerator(
        self, data: NumericArray | NumericValue
    ) -> NumericArray | NumericValue:
        return self.numerator / data  # type: ignore[operator]

    @beartype
    def with_denominator(
        self, data: NumericArray | NumericValue
    ) -> NumericArray | NumericValue:
        return data / self.denominator  # type: ignore[operator]

    def default(self) -> NumericArray | NumericValue:
        return self.numerator / self.denominator  # type: ignore[operator]


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
        return data**self.power


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
